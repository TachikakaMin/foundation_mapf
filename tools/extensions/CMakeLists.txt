cmake_minimum_required(VERSION 3.12)
project(MAPF_Extensions VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找Python和NumPy
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy REQUIRED)

# 获取Python扩展后缀
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 创建Python扩展模块
add_library(construct_features_native MODULE
    construct_features_native.cpp
)

# 设置扩展属性
set_target_properties(construct_features_native PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_EXT_SUFFIX}"
    CXX_VISIBILITY_PRESET hidden
)

# 链接Python和NumPy
target_include_directories(construct_features_native PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

target_link_libraries(construct_features_native PRIVATE
    ${Python3_LIBRARIES}
)

# 编译选项
target_compile_options(construct_features_native PRIVATE
    -O3
    -ffast-math
    -march=native
)

# 定义宏
target_compile_definitions(construct_features_native PRIVATE
    NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
)

# 自定义目标
add_custom_target(build-extension
    DEPENDS construct_features_native
    COMMENT "Building Python extension"
)

add_custom_target(test-extension
    COMMAND ${Python3_EXECUTABLE} -c "import construct_features_native; print('✅ 扩展导入成功')"
    DEPENDS construct_features_native
    COMMENT "Testing Python extension"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(clean-extension
    COMMAND ${CMAKE_COMMAND} -E remove -f construct_features_native${PYTHON_EXT_SUFFIX}
    COMMENT "Cleaning Python extension"
)

# 安装
install(TARGETS construct_features_native
    LIBRARY DESTINATION ${Python3_SITELIB}
)

# 显示信息
message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "NumPy include dirs: ${Python3_NumPy_INCLUDE_DIRS}")
message(STATUS "Extension suffix: ${PYTHON_EXT_SUFFIX}")

# 帮助信息
add_custom_target(show-ext-help
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "MAPF C++ Extensions - 可用目标:"
    COMMAND ${CMAKE_COMMAND} -E echo "  construct_features_native - 编译特征构建扩展"
    COMMAND ${CMAKE_COMMAND} -E echo "  build-extension           - 构建所有扩展"
    COMMAND ${CMAKE_COMMAND} -E echo "  test-extension            - 测试扩展"
    COMMAND ${CMAKE_COMMAND} -E echo "  clean-extension           - 清理扩展"
    COMMAND ${CMAKE_COMMAND} -E echo "  install                   - 安装扩展"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "使用方法:"
    COMMAND ${CMAKE_COMMAND} -E echo "  mkdir build && cd build"
    COMMAND ${CMAKE_COMMAND} -E echo "  cmake .."
    COMMAND ${CMAKE_COMMAND} -E echo "  make build-extension"
    COMMAND ${CMAKE_COMMAND} -E echo "  make test-extension"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMENT "显示帮助信息"
) 