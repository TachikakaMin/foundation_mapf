# `gen_pathfile.sh`

## 文件作用

这是批量路径生成脚本。它会遍历已有地图，为不同智能体数量和随机种子调用 LACAM 可执行文件，生成 `.path` 轨迹文件。

## 脚本流程

1. 检查 `parallel` 和 `bc` 是否存在
2. 检查 `data_generation_LACAM/lacam3/build/main` 是否已经编译
3. 创建 `data/path_files/`
4. 遍历 `data/map_files/maze-*/*.map`
5. 针对每张地图计算不同智能体数量下应生成的轨迹数量
6. 只对缺失的输出文件生成任务
7. 把任务交给 GNU `parallel` 并行执行

## 接口情况

### 外部依赖

- `GNU parallel`
- `bc`
- `nproc`
- `data_generation_LACAM/lacam3/build/main`

### 生成的输出

输出目录结构为：

- `data/path_files/<map_pattern>/<map_name>-<N>/<map_name>-<N>-<seed>.path`

### 调用的 LACAM 接口

并行命令最终等价于：

```bash
data_generation_LACAM/lacam3/build/main \
  -m data/map_files/{map_pattern}/{map_name}.map \
  -N {agent_num} \
  -s {seed} \
  -v 1 \
  -o data/path_files/{map_pattern}/{map_name}-{agent_num}/{map_name}-{agent_num}-{seed}.path
```

### 轨迹数量策略

脚本按地图名中的密度字段和 `N in {128, 96, 64, 32, 16}` 计算 `num_paths`。

## 用法

```bash
bash gen_pathfile.sh
```
