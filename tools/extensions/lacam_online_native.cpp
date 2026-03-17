#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <cstdint>
#include <exception>
#include <string>

#include <lacam.hpp>
#include <post_processing.hpp>

namespace {

uint8_t encode_action(int64_t cur_row, int64_t cur_col, int64_t next_row, int64_t next_col) {
    const int64_t dr = next_row - cur_row;
    const int64_t dc = next_col - cur_col;

    if (dr == 0 && dc == 1) {
        return 1;
    }
    if (dr == 0 && dc == -1) {
        return 2;
    }
    if (dr == -1 && dc == 0) {
        return 3;
    }
    if (dr == 1 && dc == 0) {
        return 4;
    }
    return 0;
}

void decref_and_null(PyObject*& obj) {
    Py_XDECREF(obj);
    obj = nullptr;
}

}  // namespace

static PyObject* generate_lacam_solution_cpp(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;
    const char* map_file_cstr = nullptr;
    int agent_num = 0;
    int seed = 0;
    int time_limit_sec = 5;
    int verbose = 0;

    static const char* kwlist[] = {
        "map_file",
        "agent_num",
        "seed",
        "time_limit_sec",
        "verbose",
        nullptr,
    };

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "sii|ii",
            const_cast<char**>(kwlist),
            &map_file_cstr,
            &agent_num,
            &seed,
            &time_limit_sec,
            &verbose)) {
        return nullptr;
    }

    if (agent_num <= 0) {
        PyErr_SetString(PyExc_ValueError, "agent_num must be > 0");
        return nullptr;
    }
    if (time_limit_sec <= 0) {
        PyErr_SetString(PyExc_ValueError, "time_limit_sec must be > 0");
        return nullptr;
    }

    try {
        const std::string map_file(map_file_cstr);
        const Instance ins(map_file, agent_num, seed);
        if (!ins.is_valid(0)) {
            PyErr_SetString(PyExc_RuntimeError, "Invalid MAPF instance for the given map/agent_num");
            return nullptr;
        }

        const Deadline deadline(static_cast<double>(time_limit_sec) * 1000.0);
        const Solution solution = solve(ins, verbose - 1, &deadline, seed);

        if (solution.empty()) {
            PyErr_SetString(PyExc_RuntimeError, "LACAM failed to find a solution");
            return nullptr;
        }
        if (!is_feasible_solution(ins, solution, 0)) {
            PyErr_SetString(PyExc_RuntimeError, "LACAM returned an infeasible solution");
            return nullptr;
        }

        const npy_intp steps = static_cast<npy_intp>(solution.size());
        const npy_intp n_agents = static_cast<npy_intp>(ins.N);

        PyObject* positions_obj = nullptr;
        PyObject* actions_obj = nullptr;
        PyObject* goals_obj = nullptr;
        PyObject* result = nullptr;
        PyObject* steps_obj = nullptr;
        PyObject* agent_num_obj = nullptr;

        npy_intp pos_dims[3] = {steps, n_agents, 2};
        npy_intp act_dims[2] = {steps, n_agents};
        npy_intp goals_dims[2] = {n_agents, 2};

        positions_obj = PyArray_SimpleNew(3, pos_dims, NPY_INT64);
        actions_obj = PyArray_SimpleNew(2, act_dims, NPY_UINT8);
        goals_obj = PyArray_SimpleNew(2, goals_dims, NPY_INT64);

        if (!positions_obj || !actions_obj || !goals_obj) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to allocate numpy arrays");
            decref_and_null(positions_obj);
            decref_and_null(actions_obj);
            decref_and_null(goals_obj);
            return nullptr;
        }

        auto* positions = reinterpret_cast<int64_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(positions_obj)));
        auto* actions = reinterpret_cast<uint8_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(actions_obj)));
        auto* goals = reinterpret_cast<int64_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(goals_obj)));

        for (npy_intp i = 0; i < n_agents; ++i) {
            const auto* goal_vertex = ins.goals[static_cast<size_t>(i)];
            goals[i * 2 + 0] = static_cast<int64_t>(goal_vertex->y);
            goals[i * 2 + 1] = static_cast<int64_t>(goal_vertex->x);
        }

        for (npy_intp t = 0; t < steps; ++t) {
            const auto& config = solution[static_cast<size_t>(t)];
            const auto& next_config = solution[static_cast<size_t>((t + 1 < steps) ? (t + 1) : t)];
            for (npy_intp i = 0; i < n_agents; ++i) {
                const auto* cur_v = config[static_cast<size_t>(i)];
                const auto* next_v = next_config[static_cast<size_t>(i)];

                const int64_t cur_row = static_cast<int64_t>(cur_v->y);
                const int64_t cur_col = static_cast<int64_t>(cur_v->x);
                const int64_t next_row = static_cast<int64_t>(next_v->y);
                const int64_t next_col = static_cast<int64_t>(next_v->x);

                const npy_intp pos_base = (t * n_agents + i) * 2;
                positions[pos_base + 0] = cur_row;
                positions[pos_base + 1] = cur_col;
                actions[t * n_agents + i] = encode_action(cur_row, cur_col, next_row, next_col);
            }
        }

        result = PyDict_New();
        if (!result) {
            decref_and_null(positions_obj);
            decref_and_null(actions_obj);
            decref_and_null(goals_obj);
            return nullptr;
        }

        steps_obj = PyLong_FromLongLong(static_cast<long long>(steps));
        agent_num_obj = PyLong_FromLongLong(static_cast<long long>(n_agents));
        if (!steps_obj || !agent_num_obj) {
            Py_DECREF(result);
            decref_and_null(positions_obj);
            decref_and_null(actions_obj);
            decref_and_null(goals_obj);
            decref_and_null(steps_obj);
            decref_and_null(agent_num_obj);
            return nullptr;
        }

        if (PyDict_SetItemString(result, "positions", positions_obj) != 0 ||
            PyDict_SetItemString(result, "actions", actions_obj) != 0 ||
            PyDict_SetItemString(result, "goals", goals_obj) != 0 ||
            PyDict_SetItemString(result, "steps", steps_obj) != 0 ||
            PyDict_SetItemString(result, "agent_num", agent_num_obj) != 0) {
            Py_DECREF(result);
            decref_and_null(positions_obj);
            decref_and_null(actions_obj);
            decref_and_null(goals_obj);
            decref_and_null(steps_obj);
            decref_and_null(agent_num_obj);
            return nullptr;
        }

        decref_and_null(positions_obj);
        decref_and_null(actions_obj);
        decref_and_null(goals_obj);
        decref_and_null(steps_obj);
        decref_and_null(agent_num_obj);
        return result;
    } catch (const std::exception& e) {
        PyErr_Format(PyExc_RuntimeError, "LACAM generation failed: %s", e.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "LACAM generation failed with unknown exception");
        return nullptr;
    }
}

static PyMethodDef module_methods[] = {
    {"generate_lacam_solution_cpp", reinterpret_cast<PyCFunction>(generate_lacam_solution_cpp), METH_VARARGS | METH_KEYWORDS, "Generate one MAPF scenario using LACAM and return numpy arrays."},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "lacam_online_native",
    "Native C++ online MAPF scenario generation via LACAM",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_lacam_online_native(void) {
    import_array();
    return PyModule_Create(&module_definition);
}
