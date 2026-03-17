# `tools/visualize_path.py`

## 文件作用

这个文件提供 MAPF 路径可视化能力，支持：

- 交互查看每一步轨迹
- 键盘逐帧播放
- 导出 MP4 视频

它是其他可视化脚本的底层实现。

## 主要函数

### `revert_xy(paths)`

把内部使用的 `(x, y)` 或 `(row, col)` 顺序转换成绘图时的显示顺序。可以处理：

- 单条路径
- 多时间步路径序列

### `visualize_path(all_paths, all_goal_locations, file_name, video_path=None, show=False)`

主可视化入口。

主要流程：

1. 调用 `revert_xy()` 调整坐标顺序
2. 通过 `parse_file_name()` 找到地图文件
3. 调用 `read_map()` 绘制障碍地图
4. 为每个智能体建立：
   - 当前点散点
   - 历史轨迹线
   - 目标点散点
   - 当前到目标的虚线
   - 编号标注
5. 创建滑块和键盘事件
6. 在 `show=False` 时自动保存 MP4

## 交互接口

### 键盘控制

- `Right`: 下一帧
- `Left`: 上一帧
- `Space`: 播放或暂停
- `v`: 保存视频

### 参数说明

- `all_paths`: 每一帧的全部智能体位置
- `all_goal_locations`: 每一帧的全部目标位置
- `file_name`: 用来反推地图路径和输出视频名
- `video_path`: 视频输出目录
- `show`: 是否直接打开窗口

## 依赖接口

- `tools.utils.read_map(map_name)`
- `tools.utils.parse_file_name(file_name)`
- `matplotlib.animation.FFMpegWriter`

## 用法

```python
from tools.visualize_path import visualize_path

visualize_path(all_paths, all_goal_locations, file_name, video_path="evals", show=False)
```
