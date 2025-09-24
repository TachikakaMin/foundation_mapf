cmake_minimum_required(VERSION 3.12)
project(single_step_pibt)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找Python和pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 QUIET)

if(NOT pybind11_FOUND)
    # 如果找不到pybind11，尝试使用pip安装的版本
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE pybind11_RESULT
    )
    
    if(pybind11_RESULT EQUAL 0)
        find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})
    else()
        message(FATAL_ERROR "pybind11 not found. Please install it with: pip install pybind11")
    endif()
endif()

# 添加源文件
set(SOURCES
    single_step_pibt.cpp
    python_binding.cpp
)

# 创建Python模块
pybind11_add_module(single_step_pibt_py ${SOURCES})

# 编译属性
target_compile_definitions(single_step_pibt_py PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_compile_features(single_step_pibt_py PRIVATE cxx_std_17)

# 优化设置
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(single_step_pibt_py PRIVATE -O3)
endif()

# 设置输出路径（可选）
set_target_properties(single_step_pibt_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)