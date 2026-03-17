# `tools/convert_lacam_path_to_bin.py`

## 文件作用

这个脚本把 LACAM 输出的多个 `.path` 文本文件整理成项目训练使用的 `.mbin` 合并二进制格式。它是 Python 版本的路径转换工具，对应的高性能替代是 [tools/convert_path_to_mbin.cpp](/home/yimin/research/RAILGUN/tools/convert_path_to_mbin.cpp)。

## 主要函数

### `get_action(cur_pos, next_pos)`

把相邻两步坐标差转换为动作编码：

- `(0, 1)` -> `1`
- `(0, -1)` -> `2`
- `(-1, 0)` -> `3`
- `(1, 0)` -> `4`
- 其他 -> `0`

### `convert_path_to_scenario_data(file_name)`

把单个 `.path` 文件转换成一个场景的二进制字节块。

流程：

1. 读取文件
2. 找到 `solution=` 行
3. 调用 `parse_coordinates()` 解析每个时间步
4. 写入：
   - `steps`
   - `agent_num`
   - 每步的位置
   - 每步的动作

返回值是一个字典，包含：

- `steps`
- `agent_num`
- `data`
- `file_name`

### `group_path_files(path_files)`

按 `(map_name, agent_num)` 对 `.path` 文件分组，为后续合并同类场景做准备。

### `create_mbin_file(map_agent_key, path_files)`

为同一组场景生成一个 `.mbin` 文件。

写入内容包括：

- 16 字节文件头
- 场景索引表
- 所有场景数据块

## 命令行入口

脚本作为模块执行时会：

1. 递归搜索输入目录下全部 `.path`
2. 调用 `group_path_files()`
3. 对每个组执行 `create_mbin_file()`

命令：

```bash
python -m tools.convert_lacam_path_to_bin <path_to_lacam_result_file_dir>
```

## 接口情况

### 输入

- LACAM `.path` 文本文件

### 输出

- `data/input_data/<map_name>/<map_name>-<agent_num>/<map_name>-<agent_num>.mbin`

### 依赖

- `tools.utils.parse_coordinates(coord_str)`

## 用法

```bash
python -m tools.convert_lacam_path_to_bin data/path_files
```
