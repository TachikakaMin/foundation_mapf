# `tools/visualize_lacam_path.py`

## 文件作用

这个脚本用于解析 LACAM 生成的 `.path` 文本结果，并复用 [tools/visualize_path.py](/home/yimin/research/RAILGUN/tools/visualize_path.py) 进行可视化。

## 主要函数

### `parse_lacam_file(file_name)`

处理流程：

1. 读取整个 `.path` 文件
2. 定位 `solution=` 所在行
3. 对后续每一行调用 `parse_coordinates()`
4. 组装为 `all_paths`

返回值：

- 成功时返回完整路径序列
- 没找到 `solution=` 时打印错误并返回 `None`

## 命令行入口

入口命令：

```bash
python -m tools.visualize_lacam_path <path_to_lacam_result_file>
```

脚本会把最后一帧位置作为目标位置，随后调用 `visualize_path(..., show=True)`。

## 依赖接口

- `tools.utils.parse_coordinates(coord_str)`
- `tools.visualize_path.visualize_path(...)`

## 用法

```bash
python -m tools.visualize_lacam_path data/path_files/example/example.path
```
