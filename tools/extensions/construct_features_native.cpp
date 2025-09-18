#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <random>
#include <vector>
#include <unordered_map>

const int NOT_FOUND_PATH = 2048;

// 获取距离的辅助函数
int get_distance_from_map(PyObject* distance_map, int agent_x, int agent_y, int goal_x, int goal_y) {
    // 检查是否是C++距离地图读取器
    if (PyObject_HasAttrString(distance_map, "get_distance")) {
        PyObject* get_distance_method = PyObject_GetAttrString(distance_map, "get_distance");
        PyObject* agent_pos = PyTuple_Pack(2, PyLong_FromLong(agent_x), PyLong_FromLong(agent_y));
        PyObject* goal_pos = PyTuple_Pack(2, PyLong_FromLong(goal_x), PyLong_FromLong(goal_y));
        PyObject* result = PyObject_CallFunctionObjArgs(get_distance_method, agent_pos, goal_pos, NULL);
        
        if (result && PyLong_Check(result)) {
            int distance = PyLong_AsLong(result);
            Py_DECREF(result);
            Py_DECREF(agent_pos);
            Py_DECREF(goal_pos);
            Py_DECREF(get_distance_method);
            return distance;
        }
        
        Py_XDECREF(result);
        Py_DECREF(agent_pos);
        Py_DECREF(goal_pos);
        Py_DECREF(get_distance_method);
    }
    
    // 回退到Python字典格式
    PyObject* agent_key = PyTuple_Pack(2, PyLong_FromLong(agent_x), PyLong_FromLong(agent_y));
    if (PyDict_Contains(distance_map, agent_key)) {
        PyObject* dist_matrix = PyDict_GetItem(distance_map, agent_key);
        if (PyArray_Check(dist_matrix)) {
            PyArrayObject* arr = (PyArrayObject*)dist_matrix;
            if (goal_x >= 0 && goal_x < PyArray_DIM(arr, 0) && 
                goal_y >= 0 && goal_y < PyArray_DIM(arr, 1)) {
                int* data = (int*)PyArray_GETPTR2(arr, goal_x, goal_y);
                int distance = *data;
                Py_DECREF(agent_key);
                return distance;
            }
        }
    }
    
    Py_DECREF(agent_key);
    return NOT_FOUND_PATH;
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
    int height = PyArray_DIM(map_data, 0);
    int width = PyArray_DIM(map_data, 1);
    int agent_num = PyArray_DIM(agent_locations, 0);
    
    // 创建输出数组
    npy_intp dims[3] = {feature_dim, height, width};
    PyArrayObject* output = (PyArrayObject*)PyArray_ZEROS(3, dims, NPY_FLOAT32, 0);
    if (!output) return NULL;
    
    // 获取数据指针
    float* output_data = (float*)PyArray_DATA(output);
    float* map_data_ptr = (float*)PyArray_DATA(map_data);
    long* agent_loc_ptr = (long*)PyArray_DATA(agent_locations);
    long* goal_loc_ptr = (long*)PyArray_DATA(goal_locations);
    
    // 第0层: 复制地图数据
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * width + j;
            output_data[idx] = map_data_ptr[idx];
        }
    }
    
    // 第1层: 智能体位置
    for (int i = 0; i < agent_num; ++i) {
        int x = agent_loc_ptr[i * 2];
        int y = agent_loc_ptr[i * 2 + 1];
        if (x >= 0 && x < height && y >= 0 && y < width) {
            int idx = height * width + x * width + y;  // 第1层的索引
            output_data[idx] = i + 1;  // 智能体ID
        }
    }
    
    // 第2层: 目标位置
    for (int i = 0; i < agent_num; ++i) {
        int x = goal_loc_ptr[i * 2];
        int y = goal_loc_ptr[i * 2 + 1];
        if (x >= 0 && x < height && y >= 0 && y < width) {
            int idx = 2 * height * width + x * width + y;  // 第2层的索引
            output_data[idx] = i + 1;  // 智能体ID
        }
    }
    
    // 第3层及以上: 距离和梯度特征
    if (feature_dim >= 4) {
        std::vector<float> distances(agent_num);
        
        // 计算距离
        for (int i = 0; i < agent_num; ++i) {
            int agent_x = agent_loc_ptr[i * 2];
            int agent_y = agent_loc_ptr[i * 2 + 1];
            int goal_x = goal_loc_ptr[i * 2];
            int goal_y = goal_loc_ptr[i * 2 + 1];
            
            distances[i] = get_distance_from_map(distance_map, agent_x, agent_y, goal_x, goal_y);
        }
        
        // 设置距离特征
        for (int i = 0; i < agent_num; ++i) {
            int x = agent_loc_ptr[i * 2];
            int y = agent_loc_ptr[i * 2 + 1];
            if (x >= 0 && x < height && y >= 0 && y < width) {
                int idx = 3 * height * width + x * width + y;  // 第3层的索引
                output_data[idx] = distances[i];
            }
        }
        
        // 梯度特征（feature_dim >= 5）
        if (feature_dim >= 5 && strcmp(feature_type, "gradient") == 0) {
            std::vector<float> dx(agent_num, 0);
            std::vector<float> dy(agent_num, 0);
            
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> choice(-1, 1);
            std::uniform_int_distribution<> binary(0, 1);
            
            for (int i = 0; i < agent_num; ++i) {
                int agent_x = agent_loc_ptr[i * 2];
                int agent_y = agent_loc_ptr[i * 2 + 1];
                int goal_x = goal_loc_ptr[i * 2];
                int goal_y = goal_loc_ptr[i * 2 + 1];
                float current_dist = distances[i];
                
                // 计算四个方向的距离
                float left_dist = NOT_FOUND_PATH, right_dist = NOT_FOUND_PATH;
                float up_dist = NOT_FOUND_PATH, down_dist = NOT_FOUND_PATH;
                
                // 检查边界和碰撞
                bool left_valid = (agent_x > 0);
                bool right_valid = (agent_x < height - 1);
                bool up_valid = (agent_y > 0);
                bool down_valid = (agent_y < width - 1);
                
                // 检查是否与其他智能体碰撞
                for (int j = 0; j < agent_num && (left_valid || right_valid || up_valid || down_valid); ++j) {
                    if (i == j) continue;
                    int other_x = agent_loc_ptr[j * 2];
                    int other_y = agent_loc_ptr[j * 2 + 1];
                    
                    if (left_valid && other_x == agent_x - 1 && other_y == agent_y) left_valid = false;
                    if (right_valid && other_x == agent_x + 1 && other_y == agent_y) right_valid = false;
                    if (up_valid && other_x == agent_x && other_y == agent_y - 1) up_valid = false;
                    if (down_valid && other_x == agent_x && other_y == agent_y + 1) down_valid = false;
                }
                
                // 计算有效方向的距离
                if (left_valid) {
                    left_dist = get_distance_from_map(distance_map, agent_x - 1, agent_y, goal_x, goal_y);
                }
                if (right_valid) {
                    right_dist = get_distance_from_map(distance_map, agent_x + 1, agent_y, goal_x, goal_y);
                }
                if (up_valid) {
                    up_dist = get_distance_from_map(distance_map, agent_x, agent_y - 1, goal_x, goal_y);
                }
                if (down_valid) {
                    down_dist = get_distance_from_map(distance_map, agent_x, agent_y + 1, goal_x, goal_y);
                }
                
                // 计算梯度
                float delta_left = left_dist - current_dist;
                float delta_right = right_dist - current_dist;
                float delta_up = up_dist - current_dist;
                float delta_down = down_dist - current_dist;
                
                // X方向梯度
                if (delta_left > 0 && delta_right > 0) {
                    dx[i] = 0;
                } else if (delta_left >= 0 && delta_right < 0) {
                    dx[i] = 1;
                } else if (delta_left < 0 && delta_right >= 0) {
                    dx[i] = -1;
                } else if (delta_left < 0 && delta_right < 0) {
                    dx[i] = (binary(gen) == 0) ? -1 : 1;
                } else if (delta_left == 0 && delta_right == 0) {
                    dx[i] = choice(gen);
                } else if (delta_left == 0 && delta_right > 0) {
                    dx[i] = (binary(gen) == 0) ? 0 : -1;
                } else if (delta_left > 0 && delta_right == 0) {
                    dx[i] = (binary(gen) == 0) ? 0 : 1;
                } else {
                    dx[i] = (binary(gen) == 0) ? -1 : 1;
                }
                
                // Y方向梯度
                if (delta_down > 0 && delta_up > 0) {
                    dy[i] = 0;
                } else if (delta_down >= 0 && delta_up < 0) {
                    dy[i] = 1;
                } else if (delta_down < 0 && delta_up >= 0) {
                    dy[i] = -1;
                } else if (delta_down < 0 && delta_up < 0) {
                    dy[i] = (binary(gen) == 0) ? -1 : 1;
                } else if (delta_down == 0 && delta_up == 0) {
                    dy[i] = choice(gen);
                } else if (delta_down == 0 && delta_up > 0) {
                    dy[i] = (binary(gen) == 0) ? -1 : 0;
                } else if (delta_down > 0 && delta_up == 0) {
                    dy[i] = (binary(gen) == 0) ? 0 : 1;
                } else {
                    dy[i] = (binary(gen) == 0) ? -1 : 1;
                }
            }
            
            // 设置梯度特征
            for (int i = 0; i < agent_num; ++i) {
                int x = agent_loc_ptr[i * 2];
                int y = agent_loc_ptr[i * 2 + 1];
                if (x >= 0 && x < height && y >= 0 && y < width) {
                    if (feature_dim >= 5) {
                        int idx = 4 * height * width + x * width + y;
                        output_data[idx] = dx[i];
                    }
                    if (feature_dim >= 6) {
                        int idx = 5 * height * width + x * width + y;
                        output_data[idx] = dy[i];
                    }
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