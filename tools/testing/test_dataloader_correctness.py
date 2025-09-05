#!/usr/bin/env python3
"""
数据加载器正确性测试

验证MAPF数据加载器的各个组件是否正确工作
"""

import sys
import os
import unittest
import tempfile
import struct
import numpy as np
import torch
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class TestDataLoaderCorrectness(unittest.TestCase):
    """数据加载器正确性测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_map_size = (32, 32)
        self.test_agents = 4
        
    def tearDown(self):
        """测试后的清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_mbin_file(self, filename, num_scenarios=3):
        """创建测试用的.mbin文件"""
        scenarios = []
        
        for i in range(num_scenarios):
            # 创建场景数据
            steps = 10 + i * 5  # 变化的步数
            agent_num = self.test_agents
            
            scenario_data = []
            for t in range(steps):
                # 位置数据
                for agent in range(agent_num):
                    x = (5 + agent + t) % self.test_map_size[0]
                    y = (5 + agent + t) % self.test_map_size[1]
                    scenario_data.extend([x, y])
                
                # 动作数据
                for agent in range(agent_num):
                    action = (t + agent) % 5  # 0-4的动作
                    scenario_data.append(action)
            
            scenarios.append({
                'steps': steps,
                'agent_num': agent_num,
                'data': bytes(scenario_data),
                'filename': f'scenario_{i}.path'
            })
        
        # 写入.mbin文件
        with open(filename, 'wb') as f:
            # 文件头部
            header = struct.pack('IIII', len(scenarios), 0, 0, 0)
            f.write(header)
            
            # 索引表
            data_offset = 16 + len(scenarios) * 272  # 头部 + 索引表
            for i, scenario in enumerate(scenarios):
                index_data = struct.pack('QI', data_offset, len(scenario['data']) + 4)
                index_data += struct.pack('HH', scenario['steps'], scenario['agent_num'])
                index_data += scenario['filename'].encode('utf-8').ljust(256, b'\0')
                f.write(index_data)
                data_offset += len(scenario['data']) + 4
            
            # 场景数据
            for scenario in scenarios:
                f.write(struct.pack('HH', scenario['steps'], scenario['agent_num']))
                f.write(scenario['data'])
    
    def create_test_map_file(self, map_name):
        """创建测试地图文件"""
        height, width = self.test_map_size
        
        map_content = f"""type octile
height {height}
width {width}
map
"""
        
        # 创建简单的地图
        for i in range(height):
            line = ""
            for j in range(width):
                if i == 0 or i == height-1 or j == 0 or j == width-1:
                    line += "@"  # 边界
                else:
                    line += "."  # 可通行
            map_content += line + "\n"
        
        map_file = os.path.join(self.temp_dir, "map_files", "test-32-32-10-1-75", f"{map_name}.map")
        os.makedirs(os.path.dirname(map_file), exist_ok=True)
        
        with open(map_file, 'w') as f:
            f.write(map_content)
        
        return map_file
    
    def create_test_distance_map(self, map_file):
        """创建测试距离地图"""
        from tools.data_processing.distance_map_reader import DistanceMapReader
        
        # 创建简化的距离地图
        height, width = self.test_map_size
        distance_data = {}
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                # 简单的曼哈顿距离
                distances = np.zeros((height, width), dtype=np.int32)
                for x in range(height):
                    for y in range(width):
                        if x == 0 or x == height-1 or y == 0 or y == width-1:
                            distances[x, y] = 2048  # 障碍物
                        else:
                            distances[x, y] = abs(x - i) + abs(y - j)
                distance_data[(i, j)] = distances
        
        return distance_data
    
    def test_mbin_file_structure(self):
        """测试.mbin文件结构的正确性"""
        print("\n🧪 测试.mbin文件结构...")
        
        mbin_file = os.path.join(self.temp_dir, "test.mbin")
        self.create_test_mbin_file(mbin_file, num_scenarios=3)
        
        # 验证文件结构
        with open(mbin_file, 'rb') as f:
            # 读取头部
            header = f.read(16)
            num_scenarios = struct.unpack('I', header[:4])[0]
            self.assertEqual(num_scenarios, 3, "场景数量应该为3")
            
            # 读取第一个索引
            index_data = f.read(272)
            offset = struct.unpack('Q', index_data[:8])[0]
            data_size = struct.unpack('I', index_data[8:12])[0]
            steps = struct.unpack('H', index_data[12:14])[0]
            agent_num = struct.unpack('H', index_data[14:16])[0]
            
            self.assertEqual(steps, 10, "第一个场景应该有10步")
            self.assertEqual(agent_num, self.test_agents, f"应该有{self.test_agents}个智能体")
            self.assertGreater(offset, 0, "偏移量应该大于0")
            self.assertGreater(data_size, 0, "数据大小应该大于0")
        
        print("✅ .mbin文件结构测试通过")
    
    def test_dataloader_basic_functionality(self):
        """测试数据加载器基本功能"""
        print("\n🧪 测试数据加载器基本功能...")
        
        # 创建测试文件
        mbin_file = os.path.join(self.temp_dir, "input_data", "test-32-32-10-1-75", "test-32-32-10-1-75-0-16.mbin")
        os.makedirs(os.path.dirname(mbin_file), exist_ok=True)
        self.create_test_mbin_file(mbin_file, num_scenarios=2)
        
        # 创建对应的地图文件
        map_file = self.create_test_map_file("test-32-32-10-1-75-0")
        
        # 创建距离地图
        distance_map = self.create_test_distance_map(map_file)
        
        # 模拟距离地图缓存
        import tools.data_processing.cached_distance_reader as cache_reader
        cache_reader._distance_map_cache[map_file.replace("map_files", "distance_maps").replace(".map", ".dmap")] = distance_map
        
        try:
            from MAPF_dataset_mbin import MAPFDataset
            
            # 创建数据集
            dataset = MAPFDataset([mbin_file], feature_dim=6, feature_type='gradient')
            
            # 基本检查
            self.assertGreater(len(dataset), 0, "数据集应该不为空")
            
            # 测试样本读取
            sample = dataset[0]
            
            # 验证样本结构
            self.assertIn("feature", sample, "样本应该包含feature")
            self.assertIn("action", sample, "样本应该包含action")
            self.assertIn("mask", sample, "样本应该包含mask")
            self.assertIn("file_name", sample, "样本应该包含file_name")
            
            # 验证tensor形状
            feature_shape = sample["feature"].shape
            action_shape = sample["action"].shape
            mask_shape = sample["mask"].shape
            
            self.assertEqual(len(feature_shape), 3, "特征应该是3D tensor")
            self.assertEqual(feature_shape[0], 6, "特征维度应该为6")
            self.assertEqual(feature_shape[1:], self.test_map_size, f"特征地图大小应该为{self.test_map_size}")
            self.assertEqual(action_shape, self.test_map_size, f"动作地图大小应该为{self.test_map_size}")
            self.assertEqual(mask_shape, self.test_map_size, f"掩码地图大小应该为{self.test_map_size}")
            
            print("✅ 数据加载器基本功能测试通过")
            
        except Exception as e:
            self.fail(f"数据加载器测试失败: {e}")
    
    def test_feature_construction_correctness(self):
        """测试特征构建的正确性"""
        print("\n🧪 测试特征构建正确性...")
        
        # 创建简单的测试数据
        map_data = np.zeros((16, 16), dtype=np.float32)
        agent_locations = torch.tensor([[5, 5], [10, 10]], dtype=torch.long)
        goal_locations = torch.tensor([[8, 8], [12, 12]], dtype=torch.long)
        
        # 简单的距离地图
        distance_map = {}
        for i in range(16):
            for j in range(16):
                distances = np.abs(np.arange(16)[:, None] - i) + np.abs(np.arange(16)[None, :] - j)
                distance_map[(i, j)] = distances.astype(np.int32)
        
        from tools.core.utils import construct_input_feature
        
        # 测试不同特征维度
        for feature_dim in [3, 4, 6]:
            feature_type = 'gradient' if feature_dim >= 5 else 'basic'
            
            result = construct_input_feature(
                map_data, agent_locations, goal_locations, distance_map,
                feature_dim, feature_type
            )
            
            # 验证结果
            self.assertEqual(result.shape, (feature_dim, 16, 16), f"特征维度{feature_dim}的形状不正确")
            
            # 验证第0层（地图）
            np.testing.assert_array_equal(result[0].numpy(), map_data, "地图层应该与输入地图相同")
            
            # 验证第1层（智能体位置）
            self.assertEqual(result[1, 5, 5].item(), 1, "智能体1应该在位置(5,5)")
            self.assertEqual(result[1, 10, 10].item(), 2, "智能体2应该在位置(10,10)")
            
            # 验证第2层（目标位置）
            self.assertEqual(result[2, 8, 8].item(), 1, "目标1应该在位置(8,8)")
            self.assertEqual(result[2, 12, 12].item(), 2, "目标2应该在位置(12,12)")
            
            if feature_dim >= 4:
                # 验证第3层（距离）
                dist1 = result[3, 5, 5].item()
                dist2 = result[3, 10, 10].item()
                self.assertGreater(dist1, 0, "智能体1的距离应该大于0")
                self.assertGreater(dist2, 0, "智能体2的距离应该大于0")
        
        print("✅ 特征构建正确性测试通过")
    
    def test_dataloader_batch_consistency(self):
        """测试数据加载器批次一致性"""
        print("\n🧪 测试数据加载器批次一致性...")
        
        # 创建测试文件
        mbin_file = os.path.join(self.temp_dir, "input_data", "test-32-32-10-1-75", "test-32-32-10-1-75-0-16.mbin")
        os.makedirs(os.path.dirname(mbin_file), exist_ok=True)
        self.create_test_mbin_file(mbin_file, num_scenarios=5)
        
        # 创建地图和距离地图
        map_file = self.create_test_map_file("test-32-32-10-1-75-0")
        distance_map = self.create_test_distance_map(map_file)
        
        # 设置缓存
        import tools.data_processing.cached_distance_reader as cache_reader
        cache_reader._distance_map_cache[map_file.replace("map_files", "distance_maps").replace(".map", ".dmap")] = distance_map
        
        try:
            from MAPF_dataset_mbin import MAPFDataset
            from torch.utils.data import DataLoader
            
            # 创建数据集和数据加载器
            dataset = MAPFDataset([mbin_file], feature_dim=4, feature_type='basic')
            dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
            
            # 测试批次加载
            batch_count = 0
            total_samples = 0
            
            for batch in dataloader:
                batch_count += 1
                batch_size = batch["feature"].shape[0]
                total_samples += batch_size
                
                # 验证批次结构
                self.assertEqual(len(batch["feature"].shape), 4, "批次特征应该是4D tensor")
                self.assertEqual(batch["feature"].shape[1], 4, "特征维度应该为4")
                self.assertEqual(batch["feature"].shape[2:], self.test_map_size, "地图大小应该正确")
                
                # 验证所有样本都有相同的形状
                for key in ["feature", "action", "mask"]:
                    shapes = [batch[key][i].shape for i in range(batch_size)]
                    self.assertTrue(all(s == shapes[0] for s in shapes), f"{key}的形状应该一致")
                
                if batch_count >= 3:  # 只测试前3个批次
                    break
            
            self.assertGreater(batch_count, 0, "应该至少有一个批次")
            self.assertGreater(total_samples, 0, "应该至少有一个样本")
            
            print(f"✅ 批次一致性测试通过 (测试了{batch_count}个批次，{total_samples}个样本)")
            
        except Exception as e:
            self.fail(f"批次一致性测试失败: {e}")
    
    def test_feature_layer_correctness(self):
        """测试特征层的正确性"""
        print("\n🧪 测试特征层正确性...")
        
        # 创建已知的测试数据
        map_data = np.zeros((8, 8), dtype=np.float32)
        map_data[0, :] = 1  # 顶部边界
        map_data[7, :] = 1  # 底部边界
        map_data[:, 0] = 1  # 左边界
        map_data[:, 7] = 1  # 右边界
        
        agent_locations = torch.tensor([[2, 2], [4, 4]], dtype=torch.long)
        goal_locations = torch.tensor([[6, 6], [5, 5]], dtype=torch.long)
        
        # 简单的距离地图
        distance_map = {}
        for i in range(1, 7):
            for j in range(1, 7):
                distances = np.full((8, 8), 2048, dtype=np.int32)  # 默认不可达
                # 只有内部区域可达
                for x in range(1, 7):
                    for y in range(1, 7):
                        distances[x, y] = abs(x - i) + abs(y - j)
                distance_map[(i, j)] = distances
        
        from tools.core.utils import construct_input_feature
        
        result = construct_input_feature(
            map_data, agent_locations, goal_locations, distance_map,
            feature_dim=4, feature_type='basic'
        )
        
        # 详细验证每一层
        
        # 第0层：地图数据
        np.testing.assert_array_equal(result[0].numpy(), map_data, "地图层不正确")
        
        # 第1层：智能体位置
        agent_layer = result[1].numpy()
        self.assertEqual(agent_layer[2, 2], 1, "智能体1应该在(2,2)")
        self.assertEqual(agent_layer[4, 4], 2, "智能体2应该在(4,4)")
        # 其他位置应该为0
        agent_layer[2, 2] = 0
        agent_layer[4, 4] = 0
        np.testing.assert_array_equal(agent_layer, np.zeros((8, 8)), "智能体层其他位置应该为0")
        
        # 第2层：目标位置
        goal_layer = result[2].numpy()
        self.assertEqual(goal_layer[6, 6], 1, "目标1应该在(6,6)")
        self.assertEqual(goal_layer[5, 5], 2, "目标2应该在(5,5)")
        
        # 第3层：距离
        distance_layer = result[3].numpy()
        dist1 = distance_layer[2, 2]  # 智能体1到目标1的距离
        dist2 = distance_layer[4, 4]  # 智能体2到目标2的距离
        self.assertGreater(dist1, 0, "距离应该大于0")
        self.assertGreater(dist2, 0, "距离应该大于0")
        
        print("✅ 特征层正确性测试通过")
    
    def test_dataloader_memory_efficiency(self):
        """测试数据加载器内存效率"""
        print("\n🧪 测试数据加载器内存效率...")
        
        # 创建较大的测试文件
        mbin_file = os.path.join(self.temp_dir, "input_data", "test-32-32-10-1-75", "large_test.mbin")
        os.makedirs(os.path.dirname(mbin_file), exist_ok=True)
        self.create_test_mbin_file(mbin_file, num_scenarios=10)
        
        # 创建地图和距离地图
        map_file = self.create_test_map_file("test-32-32-10-1-75-0")
        distance_map = self.create_test_distance_map(map_file)
        
        # 设置缓存
        import tools.data_processing.cached_distance_reader as cache_reader
        cache_reader._distance_map_cache[map_file.replace("map_files", "distance_maps").replace(".map", ".dmap")] = distance_map
        
        try:
            from MAPF_dataset_mbin import MAPFDataset
            import psutil
            import gc
            
            # 记录初始内存
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 创建数据集
            dataset = MAPFDataset([mbin_file], feature_dim=3, feature_type='basic')
            
            # 读取多个样本
            for i in range(min(20, len(dataset))):
                sample = dataset[i]
                
                # 验证样本不为空
                self.assertIsNotNone(sample["feature"], "特征不应该为None")
                self.assertIsNotNone(sample["action"], "动作不应该为None")
                self.assertIsNotNone(sample["mask"], "掩码不应该为None")
            
            # 强制垃圾回收
            gc.collect()
            
            # 检查内存使用
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # 内存增长应该在合理范围内（小于100MB）
            self.assertLess(memory_increase, 100, f"内存增长过多: {memory_increase:.1f}MB")
            
            print(f"✅ 内存效率测试通过 (内存增长: {memory_increase:.1f}MB)")
            
        except Exception as e:
            self.fail(f"内存效率测试失败: {e}")
    
    def test_dataloader_error_handling(self):
        """测试数据加载器错误处理"""
        print("\n🧪 测试数据加载器错误处理...")
        
        try:
            from MAPF_dataset_mbin import MAPFDataset
            
            # 测试不存在的文件
            with self.assertRaises(Exception):
                dataset = MAPFDataset(["/nonexistent/file.mbin"], feature_dim=3, feature_type='basic')
            
            # 测试空文件列表
            with self.assertRaises(Exception):
                dataset = MAPFDataset([], feature_dim=3, feature_type='basic')
            
            print("✅ 错误处理测试通过")
            
        except Exception as e:
            self.fail(f"错误处理测试失败: {e}")

def run_dataloader_tests():
    """运行所有数据加载器测试"""
    print("🧪 MAPF数据加载器正确性测试")
    print("=" * 60)
    
    # 创建测试套件
    suite = unittest.TestSuite()
    
    # 添加测试
    test_class = TestDataLoaderCorrectness
    suite.addTest(test_class('test_mbin_file_structure'))
    suite.addTest(test_class('test_dataloader_basic_functionality'))
    suite.addTest(test_class('test_feature_construction_correctness'))
    suite.addTest(test_class('test_dataloader_batch_consistency'))
    suite.addTest(test_class('test_dataloader_memory_efficiency'))
    suite.addTest(test_class('test_dataloader_error_handling'))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
    result = runner.run(suite)
    
    # 输出结果
    print(f"\n📊 测试结果:")
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ 失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n❌ 错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    
    if success_rate == 100:
        print(f"\n🎉 所有测试通过！数据加载器工作正常")
    elif success_rate >= 80:
        print(f"\n👍 大部分测试通过 ({success_rate:.1f}%)，有少量问题需要修复")
    else:
        print(f"\n⚠️ 测试通过率较低 ({success_rate:.1f}%)，需要检查代码")
    
    return result.testsRun == len(result.failures) + len(result.errors)

if __name__ == "__main__":
    success = run_dataloader_tests()
    sys.exit(0 if success else 1) 