# C++版本路径转换工具

这是一个高性能的C++版本路径转换工具，用于将LACAM的路径文件转换为二进制格式。

## 特性

- 🚀 **高性能**: 使用C++17和优化编译
- 🔄 **并行处理**: 自动检测CPU核心数并使用多线程
- 📊 **实时进度条**: 显示处理进度、百分比和预计剩余时间
- 🛡️ **错误处理**: 完善的错误处理和文件验证
- 💾 **智能缓存**: 跳过已存在的输出文件
- 🎯 **内存优化**: 高效的内存管理和文件I/O

## 编译要求

- C++17兼容的编译器 (GCC 7+, Clang 5+, MSVC 2017+)
- 支持filesystem库
- 支持threading库

## 编译方法

### 使用Makefile (推荐)

```bash
# 编译优化版本
make

# 编译调试版本
make debug

# 编译性能分析版本
make profile

# 清理编译文件
make clean

# 查看所有可用目标
make help
```

### 手动编译

```bash
g++ -std=c++17 -O3 -Wall -Wextra -pthread -o convert_lacam_path_to_bin convert_lacam_path_to_bin.cpp
```

## 使用方法

```bash
./convert_lacam_path_to_bin <lacam结果文件目录路径>
```

### 示例

```bash
# 转换指定目录下的所有.path文件
./convert_lacam_path_to_bin /path/to/lacam/results/

# 转换当前目录下的所有.path文件
./convert_lacam_path_to_bin .
```

## 测试验证

### 运行所有测试

```bash
# 运行内置单元测试
make test

# 运行完整的集成测试（推荐）
python test_integration.py
```

### 验证转换结果

```bash
# 验证单个文件
./test_converter path/to/file.path

# 验证整个目录
./test_converter /path/to/directory/

# 仅运行单元测试
./test_converter
```

### 测试功能

- ✅ **单元测试**: 验证坐标解析和动作计算逻辑
- ✅ **文件一致性测试**: 确保转换前后数据完全一致
- ✅ **批量处理测试**: 验证多文件并行处理
- ✅ **性能测试**: 测量转换速度和内存使用
- ✅ **错误处理测试**: 验证异常情况的处理

## 输出格式

程序会：
1. 递归搜索指定目录下的所有`.path`文件
2. 将每个`.path`文件转换为对应的`.bin`文件
3. 输出文件保存在`input_data`目录中（替换原路径中的`path_files`）

### 二进制文件格式

每个`.bin`文件包含：
- 文件头：步骤数(2字节) + 智能体数(2字节)
- 每个时间步：
  - 所有智能体的当前位置 (x, y坐标，各1字节)
  - 所有智能体的动作 (1字节)

### 动作编码

- 0: 静止
- 1: 右
- 2: 左  
- 3: 上
- 4: 下

## 性能优化

### 并行处理
- 自动检测CPU核心数
- 智能分配文件到不同线程
- 使用原子操作确保线程安全

### 内存管理
- 流式处理大文件
- 避免不必要的内存分配
- 使用移动语义优化性能

### 编译优化
- `-O3`: 最高级别优化
- `-march=native`: 针对本地CPU优化
- `-pthread`: 启用多线程支持

## 错误处理

程序会处理以下情况：
- 无效的输入目录
- 损坏的路径文件
- 无solution标记的文件
- 输出目录创建失败
- 文件读写错误

## 与Python版本对比

| 特性 | Python版本 | C++版本 |
|------|------------|---------|
| 执行速度 | 基准 | 3-5x 更快 |
| 内存使用 | 基准 | 2-3x 更少 |
| 并行处理 | ProcessPoolExecutor | 原生线程 |
| 进度显示 | tqdm | 自定义进度条 |
| 依赖 | Python + 库 | 无外部依赖 |
| 部署 | 需要Python环境 | 独立可执行文件 |

## 故障排除

### 编译错误
```bash
# 如果遇到filesystem错误，可能需要链接lstdc++fs
g++ -std=c++17 -lstdc++fs -o convert_lacam_path_to_bin convert_lacam_path_to_bin.cpp
```

### 运行时错误
- 确保有足够的磁盘空间
- 检查文件权限
- 验证输入文件格式

## 许可证

与项目主许可证保持一致。 