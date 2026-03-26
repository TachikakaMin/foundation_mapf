#include <Python.h>
#include <numpy/arrayobject.h>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <unordered_map>

const int NOT_FOUND_PATH = 2048;

// 预加载的距离图：在持有GIL时从Python dict中提取所有数据到纯C++结构
// Preloaded distance map: extracts all data from Python dict into pure C++ structures while holding GIL
struct MatrixInfo {
    const int32_t* data;  // 指向numpy数组原始数据的指针（由Python dict保持存活）
                          // Raw pointer into numpy array data (kept alive by the Python dict)
    npy_intp rows;
    npy_intp cols;
};

class PreloadedDistanceMap {
public:
    // 在持有GIL时调用：遍历Python dict，提取所有 (x,y)->矩阵 映射
    // Called while holding GIL: iterates Python dict and extracts all (x,y)->matrix mappings
    explicit PreloadedDistanceMap(PyObject* distance_map) {
        PyObject* dict = nullptr;
        bool dict_borrowed = false;

        if (PyDict_Check(distance_map)) {
            dict = distance_map;
            dict_borrowed = true;
        } else {
            PyObject* maybe_dict = PyObject_GetAttrString(distance_map, "distance_map");
            if (maybe_dict != nullptr && PyDict_Check(maybe_dict)) {
                dict = maybe_dict;
                dict_borrowed = false;
            } else {
                Py_XDECREF(maybe_dict);
                PyErr_Clear();
            }
        }

        if (dict != nullptr) {
            // 遍历dict，提取所有键值对
            // Iterate dict and extract all key-value pairs
            PyObject *key, *value;
            Py_ssize_t pos = 0;
            while (PyDict_Next(dict, &pos, &key, &value)) {
                if (!PyTuple_Check(key) || PyTuple_GET_SIZE(key) != 2) continue;
                if (!PyArray_Check(value)) continue;

                PyObject* px = PyTuple_GET_ITEM(key, 0);
                PyObject* py = PyTuple_GET_ITEM(key, 1);
                if (!PyLong_Check(px) || !PyLong_Check(py)) continue;

                int x = static_cast<int>(PyLong_AsLong(px));
                int y = static_cast<int>(PyLong_AsLong(py));
                if (PyErr_Occurred()) { PyErr_Clear(); continue; }

                auto* arr = reinterpret_cast<PyArrayObject*>(value);
                if (PyArray_NDIM(arr) != 2) continue;
                if (PyArray_TYPE(arr) != NPY_INT32) continue;

                MatrixInfo info;
                info.data = reinterpret_cast<const int32_t*>(PyArray_DATA(arr));
                info.rows = PyArray_DIM(arr, 0);
                info.cols = PyArray_DIM(arr, 1);
                map_.emplace(pack_key(x, y), info);
            }
            if (!dict_borrowed) Py_DECREF(dict);
        }
    }

    // GIL释放后可安全调用：仅访问C++数据结构和原始指针
    // Safe to call after GIL release: only accesses C++ data structures and raw pointers
    int get_distance(int agent_x, int agent_y, int goal_x, int goal_y) const {
        const uint64_t k = pack_key(agent_x, agent_y);
        const auto it = map_.find(k);
        if (it == map_.end()) return NOT_FOUND_PATH;
        const MatrixInfo& info = it->second;
        if (goal_x < 0 || goal_x >= info.rows || goal_y < 0 || goal_y >= info.cols) {
            return NOT_FOUND_PATH;
        }
        return static_cast<int>(info.data[goal_x * info.cols + goal_y]);
    }

private:
    static uint64_t pack_key(int x, int y) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) |
               static_cast<uint32_t>(y);
    }

    std::unordered_map<uint64_t, MatrixInfo> map_;
};

inline int choose_sign(std::mt19937& gen) {
    static thread_local std::uniform_int_distribution<int> binary(0, 1);
    return binary(gen) == 0 ? -1 : 1;
}

inline int choose_trit(std::mt19937& gen) {
    static thread_local std::uniform_int_distribution<int> trit(-1, 1);
    return trit(gen);
}

// 主要的C++实现函数
static PyObject* construct_input_feature_cpp(PyObject* self, PyObject* args) {
    PyArrayObject *map_data, *agent_locations, *goal_locations;
    PyObject* distance_map;
    int feature_dim;
    const char* feature_type;

    // 解析参数
    if (!PyArg_ParseTuple(args, "O!O!O!Ois",
                         &PyArray_Type, &map_data,
                         &PyArray_Type, &agent_locations,
                         &PyArray_Type, &goal_locations,
                         &distance_map,
                         &feature_dim,
                         &feature_type)) {
        return NULL;
    }

    // 获取维度信息（持有GIL）
    const int height = static_cast<int>(PyArray_DIM(map_data, 0));
    const int width = static_cast<int>(PyArray_DIM(map_data, 1));
    const int agent_num = static_cast<int>(PyArray_DIM(agent_locations, 0));
    const npy_intp plane_size = static_cast<npy_intp>(height) * static_cast<npy_intp>(width);

    // 在持有GIL时，将所有Python对象数据提取到C++变量
    // While holding GIL, extract all Python object data into C++ variables

    // 提取地图数据、智能体位置、目标位置的原始指针（numpy数组在函数调用期间保持存活）
    // Extract raw pointers for map, agent locations, goal locations (numpy arrays stay alive for call duration)
    const float*   map_data_ptr    = reinterpret_cast<float*>(PyArray_DATA(map_data));
    const int64_t* agent_loc_ptr   = reinterpret_cast<int64_t*>(PyArray_DATA(agent_locations));
    const int64_t* goal_loc_ptr    = reinterpret_cast<int64_t*>(PyArray_DATA(goal_locations));

    // 将feature_type复制到std::string，避免GIL释放后访问Python字符串缓冲区
    // Copy feature_type into std::string to avoid accessing Python string buffer after GIL release
    const std::string feature_type_str(feature_type);

    // 在持有GIL时预加载距离图
    // Preload distance map while holding GIL
    PreloadedDistanceMap distance_accessor(distance_map);

    // 创建输出数组（必须在持有GIL时进行）
    // Create output array (must be done while holding GIL)
    npy_intp dims[3] = {feature_dim, height, width};
    PyArrayObject* output = (PyArrayObject*)PyArray_ZEROS(3, dims, NPY_FLOAT32, 0);
    if (!output) return NULL;

    float* output_data = reinterpret_cast<float*>(PyArray_DATA(output));

    // 释放GIL，开始纯C++计算
    // Release GIL and begin pure C++ computation
    Py_BEGIN_ALLOW_THREADS

    float* agent_plane    = output_data + plane_size;
    float* goal_plane     = output_data + 2 * plane_size;

    // 第0层：复制地图数据
    std::memcpy(output_data, map_data_ptr, sizeof(float) * static_cast<size_t>(plane_size));

    // 第1层：智能体位置
    for (int i = 0; i < agent_num; ++i) {
        int x = agent_loc_ptr[i * 2];
        int y = agent_loc_ptr[i * 2 + 1];
        if (x >= 0 && x < height && y >= 0 && y < width) {
            const npy_intp flat_idx = static_cast<npy_intp>(x) * width + y;
            agent_plane[flat_idx] = i + 1;  // 智能体ID
        }
    }

    // 第2层：目标位置
    for (int i = 0; i < agent_num; ++i) {
        int x = goal_loc_ptr[i * 2];
        int y = goal_loc_ptr[i * 2 + 1];
        if (x >= 0 && x < height && y >= 0 && y < width) {
            const npy_intp flat_idx = static_cast<npy_intp>(x) * width + y;
            goal_plane[flat_idx] = i + 1;  // 智能体ID
        }
    }

    // 第3层及以上：距离和梯度特征
    if (feature_dim >= 4) {
        std::vector<float> distances(agent_num, static_cast<float>(NOT_FOUND_PATH));
        float* distance_plane = output_data + 3 * plane_size;

        // 计算距离
        for (int i = 0; i < agent_num; ++i) {
            int agent_x = agent_loc_ptr[i * 2];
            int agent_y = agent_loc_ptr[i * 2 + 1];
            int goal_x = goal_loc_ptr[i * 2];
            int goal_y = goal_loc_ptr[i * 2 + 1];

            distances[i] = static_cast<float>(
                distance_accessor.get_distance(agent_x, agent_y, goal_x, goal_y)
            );
        }

        // 设置距离特征
        for (int i = 0; i < agent_num; ++i) {
            int x = agent_loc_ptr[i * 2];
            int y = agent_loc_ptr[i * 2 + 1];
            if (x >= 0 && x < height && y >= 0 && y < width) {
                const npy_intp flat_idx = static_cast<npy_intp>(x) * width + y;
                distance_plane[flat_idx] = distances[i];
            }
        }

        // 梯度特征（feature_dim >= 5）
        if (feature_dim >= 5 && feature_type_str == "gradient") {
            static thread_local std::mt19937 gen(std::random_device{}());
            float* dx_plane = output_data + 4 * plane_size;
            float* dy_plane = (feature_dim >= 6) ? (output_data + 5 * plane_size) : nullptr;

            for (int i = 0; i < agent_num; ++i) {
                int agent_x = agent_loc_ptr[i * 2];
                int agent_y = agent_loc_ptr[i * 2 + 1];
                int goal_x = goal_loc_ptr[i * 2];
                int goal_y = goal_loc_ptr[i * 2 + 1];
                float current_dist = distances[i];
                float dx_value = 0;
                float dy_value = 0;

                // 计算四个方向的距离
                float left_dist = NOT_FOUND_PATH, right_dist = NOT_FOUND_PATH;
                float up_dist = NOT_FOUND_PATH, down_dist = NOT_FOUND_PATH;

                // 检查边界和碰撞
                const npy_intp flat_idx = static_cast<npy_intp>(agent_x) * width + agent_y;
                bool left_valid = (agent_x > 0) &&
                    (agent_plane[(static_cast<npy_intp>(agent_x - 1) * width) + agent_y] == 0.0f);
                bool right_valid = (agent_x < height - 1) &&
                    (agent_plane[(static_cast<npy_intp>(agent_x + 1) * width) + agent_y] == 0.0f);
                bool up_valid = (agent_y > 0) &&
                    (agent_plane[(static_cast<npy_intp>(agent_x) * width) + (agent_y - 1)] == 0.0f);
                bool down_valid = (agent_y < width - 1) &&
                    (agent_plane[(static_cast<npy_intp>(agent_x) * width) + (agent_y + 1)] == 0.0f);

                // 计算有效方向的距离
                if (left_valid) {
                    left_dist = static_cast<float>(
                        distance_accessor.get_distance(agent_x - 1, agent_y, goal_x, goal_y)
                    );
                }
                if (right_valid) {
                    right_dist = static_cast<float>(
                        distance_accessor.get_distance(agent_x + 1, agent_y, goal_x, goal_y)
                    );
                }
                if (up_valid) {
                    up_dist = static_cast<float>(
                        distance_accessor.get_distance(agent_x, agent_y - 1, goal_x, goal_y)
                    );
                }
                if (down_valid) {
                    down_dist = static_cast<float>(
                        distance_accessor.get_distance(agent_x, agent_y + 1, goal_x, goal_y)
                    );
                }

                // 计算梯度
                float delta_left = left_dist - current_dist;
                float delta_right = right_dist - current_dist;
                float delta_up = up_dist - current_dist;
                float delta_down = down_dist - current_dist;

                // X方向梯度
                if (delta_left > 0 && delta_right > 0) {
                    dx_value = 0;
                } else if (delta_left >= 0 && delta_right < 0) {
                    dx_value = 1;
                } else if (delta_left < 0 && delta_right >= 0) {
                    dx_value = -1;
                } else if (delta_left < 0 && delta_right < 0) {
                    dx_value = choose_sign(gen);
                } else if (delta_left == 0 && delta_right == 0) {
                    dx_value = choose_trit(gen);
                } else if (delta_left == 0 && delta_right > 0) {
                    dx_value = (choose_sign(gen) < 0) ? -1 : 0;
                } else if (delta_left > 0 && delta_right == 0) {
                    dx_value = (choose_sign(gen) < 0) ? 0 : 1;
                } else {
                    dx_value = choose_sign(gen);
                }

                // Y方向梯度
                if (delta_down > 0 && delta_up > 0) {
                    dy_value = 0;
                } else if (delta_down >= 0 && delta_up < 0) {
                    dy_value = 1;
                } else if (delta_down < 0 && delta_up >= 0) {
                    dy_value = -1;
                } else if (delta_down < 0 && delta_up < 0) {
                    dy_value = choose_sign(gen);
                } else if (delta_down == 0 && delta_up == 0) {
                    dy_value = choose_trit(gen);
                } else if (delta_down == 0 && delta_up > 0) {
                    dy_value = (choose_sign(gen) < 0) ? -1 : 0;
                } else if (delta_down > 0 && delta_up == 0) {
                    dy_value = (choose_sign(gen) < 0) ? 0 : 1;
                } else {
                    dy_value = choose_sign(gen);
                }

                dx_plane[flat_idx] = dx_value;
                if (dy_plane != nullptr) {
                    dy_plane[flat_idx] = dy_value;
                }
            }
        }
    }

    // 重新获取GIL
    // Re-acquire GIL
    Py_END_ALLOW_THREADS

    return (PyObject*)output;
}

// 模块方法定义
static PyMethodDef module_methods[] = {
    {"construct_input_feature", construct_input_feature_cpp, METH_VARARGS,
     "Construct input features using C++ implementation"},
    {NULL, NULL, 0, NULL}
};

// 模块定义
static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "construct_features_native",
    "Native C++ implementation of construct_input_feature",
    -1,
    module_methods
};

// 模块初始化
PyMODINIT_FUNC PyInit_construct_features_native(void) {
    import_array();  // 初始化numpy C API
    return PyModule_Create(&module_definition);
}
