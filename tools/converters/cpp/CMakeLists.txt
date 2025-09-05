cmake_minimum_required(VERSION 3.12)
project(MAPF_Tools VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# 默认为Release模式
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 查找线程库
find_package(Threads REQUIRED)

# 路径转换工具
add_executable(convert_lacam_path_to_bin 
    convert_lacam_path_to_bin.cpp
)
target_link_libraries(convert_lacam_path_to_bin 
    Threads::Threads
)

# 距离地图预计算工具
add_executable(precompute_distance_maps 
    precompute_distance_maps.cpp
)
target_link_libraries(precompute_distance_maps 
    Threads::Threads
)

# 测试工具
add_executable(test_converter 
    ../../testing/test_converter.cpp
)
target_link_libraries(test_converter 
    Threads::Threads
)

# 设置输出目录
set_target_properties(test_converter PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../testing
)

# 安装目标
install(TARGETS convert_lacam_path_to_bin precompute_distance_maps
    RUNTIME DESTINATION bin
)

install(TARGETS test_converter
    RUNTIME DESTINATION bin
)

# 自定义目标
add_custom_target(clean-tools
    COMMAND ${CMAKE_COMMAND} -E remove -f 
        ${CMAKE_CURRENT_BINARY_DIR}/convert_lacam_path_to_bin
        ${CMAKE_CURRENT_BINARY_DIR}/precompute_distance_maps
        ${CMAKE_CURRENT_SOURCE_DIR}/../../testing/test_converter
    COMMENT "Cleaning all executables"
)

# 测试目标
enable_testing()

add_test(NAME unit_test
    COMMAND test_converter
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../testing
)

add_test(NAME integration_test
    COMMAND python test_integration.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../testing
)

# 显示构建信息
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")

# 帮助信息
add_custom_target(show-tools-help
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "MAPF Tools - 可用目标:"
    COMMAND ${CMAKE_COMMAND} -E echo "  all                    - 编译所有工具 (默认)"
    COMMAND ${CMAKE_COMMAND} -E echo "  convert_lacam_path_to_bin - 路径转换工具"
    COMMAND ${CMAKE_COMMAND} -E echo "  precompute_distance_maps  - 距离地图预计算工具"
    COMMAND ${CMAKE_COMMAND} -E echo "  test_converter         - 测试验证工具"
    COMMAND ${CMAKE_COMMAND} -E echo "  clean-all              - 清理所有可执行文件"
    COMMAND ${CMAKE_COMMAND} -E echo "  test                   - 运行所有测试"
    COMMAND ${CMAKE_COMMAND} -E echo "  install                - 安装到系统"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "使用方法:"
    COMMAND ${CMAKE_COMMAND} -E echo "  mkdir build && cd build"
    COMMAND ${CMAKE_COMMAND} -E echo "  cmake .."
    COMMAND ${CMAKE_COMMAND} -E echo "  make -j$(nproc)"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMENT "显示帮助信息"
) 