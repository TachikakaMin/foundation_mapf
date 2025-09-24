#!/bin/bash

# 编译单步PIBT Python模块的脚本

# 检查是否安装了pybind11
python -c "import pybind11" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: pybind11未安装，正在安装..."
    pip install pybind11
fi

# 创建build目录
mkdir -p build
cd build

# 运行cmake配置
echo "正在配置CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
echo "正在编译..."
make -j$(nproc)

# 检查编译是否成功并复制生成的模块
SO_FILE=$(find . -name "single_step_pibt_py*.so" | head -1)
if [ ! -z "$SO_FILE" ]; then
    cp "$SO_FILE" ../
    echo "✅ 编译成功！模块已复制到 $(pwd)/../"
    echo "生成的模块: $(basename $SO_FILE)"
    echo "现在您可以在Python中使用: from pibt_wrapper import pibt_solve_single_step"
else
    # 检查上级目录是否已经存在编译好的模块
    cd ..
    EXISTING_SO=$(find . -maxdepth 1 -name "single_step_pibt_py*.so" | head -1)
    if [ ! -z "$EXISTING_SO" ]; then
        echo "✅ 编译成功！模块已存在: $(basename $EXISTING_SO)"
        echo "现在您可以在Python中使用: from pibt_wrapper import pibt_solve_single_step"
    else
        echo "❌ 编译失败，请检查错误信息"
    fi
    cd build
fi

cd ..
