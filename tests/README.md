# 测试说明

这个文件夹包含了 `foundation_mapf` 项目的测试脚本。

## 测试脚本

### test_maze_generator.py
测试迷宫生成器的基本功能，验证：
- 不同尺寸的迷宫生成
- 不同参数组合的迷宫生成
- 错误处理和异常情况

## 运行测试

```bash
# 在项目根目录下运行
cd foundation_mapf
python tests/test_maze_generator.py

# 或者直接运行
python -m tests.test_maze_generator
```

## 添加新测试

1. 在 `tests/` 文件夹中创建新的测试文件
2. 文件名以 `test_` 开头
3. 测试函数以 `test_` 开头
4. 确保测试可以独立运行 