# `data_generation_ECBS/generate_data.py`

## 文件作用

这个脚本用于批量调用同目录下的 `ECBS` 可执行文件，把一组 `.yaml` 输入转换为求解结果文件。

它属于一个相对独立的数据生成入口，与主训练流程并不直接耦合。

## 主要函数

### `run_single_ecbs(yaml_file, input_dir, output_dir, weight, timeout)`

对单个 YAML 文件执行一次 ECBS。

主要逻辑：

1. 组装命令 `./ECBS -i ... -o ... -w ...`
2. 用 `subprocess.Popen(..., preexec_fn=os.setsid)` 启动新进程组
3. 在 `timeout` 限时内等待结束
4. 超时则杀掉整个进程组
5. 删除可能留下的不完整输出文件

### `run_ecbs(input_dir, output_dir, weight=1.2, timeout=5, max_workers=32)`

批量入口。

流程：

1. 列出 `input_dir` 下全部 `.yaml`
2. 创建输出目录
3. 用 `ProcessPoolExecutor` 并发执行 `run_single_ecbs()`
4. 捕获 `KeyboardInterrupt` 并尝试终止活跃子进程

## 接口情况

### 外部依赖

- 当前目录下的 `./ECBS` 二进制
- Python `subprocess`

### 输入输出

- 输入：`input_dir/*.yaml`
- 输出：`output_dir/*.yaml`，具体内容由 ECBS 决定

### 文件底部的直接执行逻辑

这个文件没有 `if __name__ == "__main__":` 保护。

也就是说，模块底部这几行会在导入时直接执行：

- 固定输入目录
- 固定输出目录
- 直接调用 `run_ecbs(...)`

如果把它作为库导入，需要特别注意这个副作用。

## 用法

当前文件的默认行为是直接运行底部示例配置；如果要手动执行，一般是：

```bash
python data_generation_ECBS/generate_data.py
```
