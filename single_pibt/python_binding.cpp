#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "single_step_pibt.hpp"

namespace py = pybind11;

PYBIND11_MODULE(single_step_pibt_py, m) {
    m.doc() = "Single Step PIBT - 只返回下一步动作的PIBT算法";
    
    // Position结构体
    py::class_<Position>(m, "Position")
        .def(py::init<int, int>(), "构造函数", py::arg("x") = 0, py::arg("y") = 0)
        .def_readwrite("x", &Position::x)
        .def_readwrite("y", &Position::y)
        .def("__eq__", &Position::operator==)
        .def("__repr__", [](const Position& p) {
            return "Position(" + std::to_string(p.x) + ", " + std::to_string(p.y) + ")";
        });
    
    // AgentState结构体
    py::class_<AgentState>(m, "AgentState")
        .def(py::init<int, Position, Position, double, int>(),
             "构造函数",
             py::arg("id"), py::arg("current_pos"), py::arg("goal_pos"),
             py::arg("priority") = 0.0, py::arg("elapsed_time") = 0)
        .def_readwrite("id", &AgentState::id)
        .def_readwrite("current_pos", &AgentState::current_pos)
        .def_readwrite("goal_pos", &AgentState::goal_pos)
        .def_readwrite("priority", &AgentState::priority)
        .def_readwrite("elapsed_time", &AgentState::elapsed_time);
    
    // SingleStepPIBT类
    py::class_<SingleStepPIBT>(m, "SingleStepPIBT")
        .def(py::init<int, int, int>(),
             "构造函数",
             py::arg("map_width"), py::arg("map_height"), py::arg("seed") = 0)
        .def("get_next_actions", &SingleStepPIBT::getNextActions,
             "获取所有agent的下一步动作",
             py::arg("agents"), py::arg("obstacle_map"),
             py::arg("action_preferences") = std::vector<std::vector<double>>(),
             R"pbdoc(
                获取所有agent的下一步动作

                参数:
                    agents: AgentState列表，包含每个agent的当前状态
                    obstacle_map: 障碍物地图，0表示可通行，1表示障碍物
                    action_preferences: 可选的动作偏好，每个agent有5个动作的偏好值

                返回:
                    int列表，每个agent的动作索引 (0=停留, 1=上, 2=下, 3=左, 4=右)
             )pbdoc");
}
