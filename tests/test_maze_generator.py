#!/usr/bin/env python3
"""
测试迷宫生成器的脚本
"""
import os
import sys
sys.path.append('.')

from data_generation_LACAM.maze_generator import generate_maze

def test_maze_generation():
    """测试迷宫生成功能"""
    print("测试迷宫生成器...")
    
    # 测试参数
    test_params = [
        (10, 10, 0.2, 3, 0.75, 0),
        (12, 12, 0.3, 4, 0.8, 1),
        (8, 8, 0.1, 2, 0.7, 2)
    ]
    
    for width, height, density, components, go_straight, seed in test_params:
        print(f"\n生成迷宫: {width}x{height}, 密度={density}, 组件={components}, 直线概率={go_straight}, 种子={seed}")
        try:
            generate_maze(width, height, density, components, go_straight, seed)
            print(f"✓ 成功生成迷宫 {seed}")
        except Exception as e:
            print(f"✗ 生成迷宫失败: {e}")
            return False
    
    print("\n所有测试通过！")
    return True

if __name__ == "__main__":
    success = test_maze_generation()
    sys.exit(0 if success else 1) 