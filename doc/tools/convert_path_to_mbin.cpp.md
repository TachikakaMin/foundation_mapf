# `tools/convert_path_to_mbin.cpp`

## 文件作用

这个文件实现了高性能多线程 `.path -> .mbin` 转换工具，是 Python 版 [tools/convert_lacam_path_to_bin.py](/home/yimin/research/RAILGUN/tools/convert_lacam_path_to_bin.py) 的 C++ 替代。

## 主要数据结构

### `Position`

保存单个坐标点：

- `x`
- `y`

### `ScenarioData`

保存一个场景的编码结果：

- `steps`
- `agent_num`
- `data`
- `file_name`

### `MBinFileInfo`

描述一个待生成 `.mbin` 文件的工作单元：

- `map_name`
- `agent_num`
- `path_files`

## `PathConverter` 类

### `getAction(cur_pos, next_pos)`

把相邻坐标差转换为动作编码。

### `parseCoordinates(coord_str)`

从一行字符串里提取全部 `(x,y)` 坐标。

### `convertPathToScenarioData(file_path)`

把单个 `.path` 文件转成 `ScenarioData`，内部会：

1. 找到 `solution=` 行
2. 读取后续每一步的位置
3. 写入步数、智能体数、位置和动作

### `createMBinFile(file_info)`

把一组同地图同智能体数量的路径文件合并成一个 `.mbin`：

- 创建输出目录
- 跳过已存在目标文件
- 写文件头
- 写索引表
- 写全部场景字节块

### `processDirectory(input_dir)`

公共入口。负责：

1. 递归收集全部 `.path`
2. 按 `(map_name, agent_num)` 分组
3. 生成 `MBinFileInfo`
4. 按硬件线程数多线程执行 `createMBinFile()`

## 程序入口

### `main(argc, argv)`

要求一个命令行参数：输入目录。会检查目录存在性，然后调用 `PathConverter::processDirectory()`。

## 接口情况

### 输入

- LACAM 生成的 `.path` 文件目录

### 输出

- `data/input_data/<map_name>/<map_name>-<agent_num>/<map_name>-<agent_num>.mbin`

### 典型优势

- 多线程
- 二进制顺序写入
- 相比 Python 版本更适合大批量路径转换

## 用法

```bash
./tools/convert_path_to_mbin data/path_files
```
