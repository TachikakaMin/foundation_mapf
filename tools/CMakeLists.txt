cmake_minimum_required(VERSION 3.12)
project(MAPF_Toolsuite VERSION 2.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# 默认为Release模式
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 选项
option(BUILD_EXTENSIONS "Build C++ Python extensions" ON)
option(BUILD_TOOLS "Build standalone C++ tools" ON)
option(BUILD_TESTS "Build tests" ON)

# 添加子目录
if(BUILD_TOOLS)
    add_subdirectory(converters/cpp)
endif()

if(BUILD_EXTENSIONS)
    add_subdirectory(extensions)
endif()

# 全局目标
add_custom_target(build-all
    COMMENT "Building all components"
)

if(BUILD_TOOLS)
    add_dependencies(build-all convert_lacam_path_to_bin precompute_distance_maps)
endif()

if(BUILD_EXTENSIONS)
    add_dependencies(build-all construct_features_native)
endif()

# 测试目标
if(BUILD_TESTS)
    enable_testing()
    
    # 集成测试
    add_test(NAME integration_test
        COMMAND python testing/test_integration.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
    
    # 性能测试
    add_test(NAME performance_test
        COMMAND python benchmarks/cpp_feature_benchmark.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
endif()

# 安装脚本
install(PROGRAMS 
    core/mapf_tools.py
    DESTINATION bin
)

install(PROGRAMS 
    core/status_check.py
    DESTINATION bin
)

# 全局清理
add_custom_target(clean-global
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Cleaning all build files"
)

# 包信息
set(CPACK_PACKAGE_NAME "MAPF-Tools")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Multi-Agent Path Finding Tools")
set(CPACK_PACKAGE_VENDOR "MAPF Research Team")
include(CPack)

# 显示构建配置
message(STATUS "=== MAPF Tools Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Build tools: ${BUILD_TOOLS}")
message(STATUS "Build extensions: ${BUILD_EXTENSIONS}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "=====================================")

# 帮助信息
add_custom_target(show-help
    COMMAND ${CMAKE_COMMAND} -E echo "MAPF Tools v${PROJECT_VERSION} - Available targets:"
    COMMAND ${CMAKE_COMMAND} -E echo "  all                     - Build all components"
    COMMAND ${CMAKE_COMMAND} -E echo "  build-all               - Build all components"
    COMMAND ${CMAKE_COMMAND} -E echo "  convert_lacam_path_to_bin - Path converter tool"
    COMMAND ${CMAKE_COMMAND} -E echo "  precompute_distance_maps  - Distance map tool"
    COMMAND ${CMAKE_COMMAND} -E echo "  construct_features_native - Feature builder extension"
    COMMAND ${CMAKE_COMMAND} -E echo "  test                    - Run all tests"
    COMMAND ${CMAKE_COMMAND} -E echo "  clean-global            - Clean all files"
    COMMAND ${CMAKE_COMMAND} -E echo "  install                 - Install to system"
    COMMENT "Display help information"
) 