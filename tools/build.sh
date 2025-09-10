#!/bin/bash

# MAPF Tools CMake构建脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}🚀 MAPF Tools v2.0 构建脚本${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  build      - 构建所有工具 (默认)"
    echo "  clean      - 清理构建文件"
    echo "  rebuild    - 清理并重新构建"
    echo "  test       - 运行测试"
    echo "  install    - 安装到系统"
    echo "  converter  - 仅编译路径转换工具"
    echo "  help       - 显示此帮助信息"
    echo ""
    echo "CMake选项:"
    echo "  -DBUILD_EXTENSIONS=OFF  - 不构建C++扩展"
    echo "  -DBUILD_TOOLS=OFF       - 不构建独立工具"
    echo "  -DBUILD_TESTS=OFF       - 不构建测试"
    echo "  -DCMAKE_BUILD_TYPE=Debug - 调试模式"
    echo ""
    echo "工具说明:"
    echo "  🔄 路径转换工具: 高性能多线程.path到.mbin转换"
    echo "  ⚡ C++扩展: 高性能特征构建和数据处理"
}

build_path_converter() {
    echo -e "${YELLOW}🔄 编译路径转换工具...${NC}"
    
    # 检查是否有必要的编译器
    if ! command -v g++ &> /dev/null; then
        echo -e "${RED}错误: 未找到 g++ 编译器${NC}"
        return 1
    fi
    
    cd "$SCRIPT_DIR"
    
    # 编译路径转换工具
    g++ -std=c++17 -O3 -pthread -Wall -Wextra -o convert_path_to_mbin convert_path_to_mbin.cpp
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ 路径转换工具编译成功!${NC}"
        echo -e "${GREEN}可执行文件: $SCRIPT_DIR/convert_path_to_mbin${NC}"
        echo ""
        echo -e "${YELLOW}使用方法:${NC}"
        echo -e "  ./tools/convert_path_to_mbin data/path_files"
        echo ""
        echo -e "${YELLOW}性能提升:${NC}"
        echo -e "  - 多线程并行处理"
        echo -e "  - 内存高效管理" 
        echo -e "  - 比Python版本快5-10倍"
    else
        echo -e "${RED}❌ 路径转换工具编译失败${NC}"
        return 1
    fi
}

build_project() {
    echo -e "${YELLOW}📦 开始构建...${NC}"
    
    # 先编译路径转换工具
    if ! build_path_converter; then
        echo -e "${RED}❌ 路径转换工具编译失败，但继续构建其他工具...${NC}"
    fi
    
    echo ""
    
    # 创建构建目录
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # 配置
    echo -e "${YELLOW}⚙️ 配置CMake...${NC}"
    cmake -DPython3_EXECUTABLE=$(which python) .. "$@"
    
    # 构建
    echo -e "${YELLOW}🔨 编译中...${NC}"
    make -j$(nproc)
    
    echo -e "${GREEN}✅ 构建完成！${NC}"
    
    # 复制扩展文件到正确位置（如果存在）
    if [ -f "extensions/construct_features_native"*.so ]; then
        cp extensions/construct_features_native*.so ../extensions/
        echo -e "${GREEN}✅ C++扩展已复制到正确位置${NC}"
    fi
}

clean_project() {
    echo -e "${YELLOW}🧹 清理构建文件...${NC}"
    rm -rf "$BUILD_DIR"
    
    # 清理路径转换工具
    if [ -f "$SCRIPT_DIR/convert_path_to_mbin" ]; then
        rm "$SCRIPT_DIR/convert_path_to_mbin"
        echo -e "${YELLOW}🧹 已清理路径转换工具${NC}"
    fi
    
    echo -e "${GREEN}✅ 清理完成${NC}"
}

run_tests() {
    echo -e "${YELLOW}🧪 运行测试...${NC}"
    cd "$BUILD_DIR"
    make test
    echo -e "${GREEN}✅ 测试完成${NC}"
}

install_project() {
    echo -e "${YELLOW}📦 安装到系统...${NC}"
    cd "$BUILD_DIR"
    sudo make install
    echo -e "${GREEN}✅ 安装完成${NC}"
}

# 主逻辑
print_header

case "${1:-build}" in
    build)
        shift
        build_project "$@"
        ;;
    clean)
        clean_project
        ;;
    rebuild)
        clean_project
        shift
        build_project "$@"
        ;;
    test)
        if [ ! -d "$BUILD_DIR" ]; then
            echo -e "${RED}❌ 请先构建项目${NC}"
            exit 1
        fi
        run_tests
        ;;
    install)
        if [ ! -d "$BUILD_DIR" ]; then
            echo -e "${RED}❌ 请先构建项目${NC}"
            exit 1
        fi
        install_project
        ;;
    converter)
        build_path_converter
        ;;
    help)
        print_usage
        ;;
    *)
        echo -e "${RED}❌ 未知选项: $1${NC}"
        print_usage
        exit 1
        ;;
esac

echo -e "${GREEN}🎉 操作完成！${NC}" 