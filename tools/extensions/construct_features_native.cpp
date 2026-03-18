#include <Python.h>
#include <numpy/arrayobject.h>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <unordered_map>

const int NOT_FOUND_PATH = 2048;

class DistanceMapAccessor {
public:
    explicit DistanceMapAccessor(PyObject* distance_map)
        : distance_dict_(nullptr), get_distance_method_(nullptr) {
        if (PyDict_Check(distance_map)) {
            distance_dict_ = distance_map;
            Py_INCREF(distance_dict_);
            return;
        }

        PyObject* maybe_dict = PyObject_GetAttrString(distance_map, "distance_map");
        if (maybe_dict != nullptr) {
            if (PyDict_Check(maybe_dict)) {
                distance_dict_ = maybe_dict;
                return;
            }
            Py_DECREF(maybe_dict);
        } else {
            PyErr_Clear();
        }

        if (PyObject_HasAttrString(distance_map, "get_distance")) {
            get_distance_method_ = PyObject_GetAttrString(distance_map, "get_distance");
            if (get_distance_method_ == nullptr) {
                PyErr_Clear();
            }
        }
    }

    ~DistanceMapAccessor() {
        Py_XDECREF(distance_dict_);
        Py_XDECREF(get_distance_method_);
    }

    int get_distance(int agent_x, int agent_y, int goal_x, int goal_y) const {
        if (distance_dict_ != nullptr) {
            PyArrayObject* arr = get_matrix(agent_x, agent_y);
            if (arr != nullptr) {
                if (goal_x >= 0 && goal_x < PyArray_DIM(arr, 0) &&
                    goal_y >= 0 && goal_y < PyArray_DIM(arr, 1)) {
                    const int32_t* data = reinterpret_cast<int32_t*>(
                        PyArray_GETPTR2(arr, goal_x, goal_y)
                    );
                    return *data;
                }
            }
            return NOT_FOUND_PATH;
        }

        if (get_distance_method_ != nullptr) {
            PyObject* agent_pos = Py_BuildValue("(ii)", agent_x, agent_y);
            PyObject* goal_pos = Py_BuildValue("(ii)", goal_x, goal_y);
            if (!agent_pos || !goal_pos) {
                Py_XDECREF(agent_pos);
                Py_XDECREF(goal_pos);
                PyErr_Clear();
                return NOT_FOUND_PATH;
            }

            PyObject* result = PyObject_CallFunctionObjArgs(
                get_distance_method_, agent_pos, goal_pos, NULL
            );
            Py_DECREF(agent_pos);
            Py_DECREF(goal_pos);

            if (result && PyLong_Check(result)) {
                const int distance = static_cast<int>(PyLong_AsLong(result));
                Py_DECREF(result);
                return distance;
            }

            Py_XDECREF(result);
            PyErr_Clear();
        }

        return NOT_FOUND_PATH;
    }

private:
    static uint64_t pack_key(int agent_x, int agent_y) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(agent_x)) << 32) |
               static_cast<uint32_t>(agent_y);
    }

    PyArrayObject* get_matrix(int agent_x, int agent_y) const {
        const uint64_t packed_key = pack_key(agent_x, agent_y);
        const auto cached = matrix_cache_.find(packed_key);
        if (cached != matrix_cache_.end()) {
            return cached->second;
        }

        PyObject* key = Py_BuildValue("(ii)", agent_x, agent_y);
        if (!key) {
            PyErr_Clear();
            return nullptr;
        }
        PyObject* matrix = PyDict_GetItemWithError(distance_dict_, key);
        Py_DECREF(key);
        if (matrix == nullptr) {
            PyErr_Clear();
            return nullptr;
        }
        if (!PyArray_Check(matrix)) {
            return nullptr;
        }
        auto* arr = reinterpret_cast<PyArrayObject*>(matrix);
        matrix_cache_.emplace(packed_key, arr);
        return arr;
    }

    PyObject* distance_dict_;
    PyObject* get_distance_method_;
    mutable std::unordered_map<uint64_t, PyArrayObject*> matrix_cache_;
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
    
    // 获取维度信息
    const int height = static_cast<int>(PyArray_DIM(map_data, 0));
    const int width = static_cast<int>(PyArray_DIM(map_data, 1));
    const int agent_num = static_cast<int>(PyArray_DIM(agent_locations, 0));
    const npy_intp plane_size = static_cast<npy_intp>(height) * static_cast<npy_intp>(width);
    
    // 创建输出数组
    npy_intp dims[3] = {feature_dim, height, width};
    PyArrayObject* output = (PyArrayObject*)PyArray_ZEROS(3, dims, NPY_FLOAT32, 0);
    if (!output) return NULL;
    
    // 获取数据指针
    float* output_data = reinterpret_cast<float*>(PyArray_DATA(output));
    float* map_data_ptr = reinterpret_cast<float*>(PyArray_DATA(map_data));
    const int64_t* agent_loc_ptr = reinterpret_cast<int64_t*>(PyArray_DATA(agent_locations));
    const int64_t* goal_loc_ptr = reinterpret_cast<int64_t*>(PyArray_DATA(goal_locations));
    float* agent_plane = output_data + plane_size;
    float* goal_plane = output_data + 2 * plane_size;
    
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
        DistanceMapAccessor distance_accessor(distance_map);
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
        if (feature_dim >= 5 && strcmp(feature_type, "gradient") == 0) {
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
