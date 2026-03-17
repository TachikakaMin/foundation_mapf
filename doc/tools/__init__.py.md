# `tools/__init__.py`

## 文件作用

这个文件当前是空文件，作用是把 `tools/` 声明为 Python 包。

## 接口情况

- 没有导出的函数、类或常量
- 主要价值是让下面这些命令成立：

```bash
python -m tools.precompute_distance_maps ...
python -m tools.convert_lacam_path_to_bin ...
python -m tools.visualize_bin_path ...
```

## 用法

通常不直接导入它，而是通过 `tools.<module>` 使用具体模块。
