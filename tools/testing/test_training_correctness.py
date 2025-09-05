#!/usr/bin/env python3
"""
训练函数正确性测试

验证训练过程中各个组件的正确性
"""

import sys
import os
import unittest
import tempfile
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class TestTrainingCorrectness(unittest.TestCase):
    """训练正确性测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.device = torch.device('cpu')  # 使用CPU进行测试
        self.map_size = (32, 32)
        self.feature_dim = 6
        self.action_dim = 5
        
    def create_mock_model(self):
        """创建模拟模型"""
        from models.unet import UNet
        model = UNet(
            n_channels=self.feature_dim,
            n_classes=self.action_dim,
            first_layer_channels=16,  # 减小以加快测试
            bilinear=False
        )
        return model.to(self.device)
    
    def create_mock_batch(self, batch_size=2):
        """创建模拟批次数据"""
        height, width = self.map_size
        
        # 创建特征
        features = torch.randn(batch_size, self.feature_dim, height, width, device=self.device)
        
        # 创建动作（只在有智能体的位置有动作）
        actions = torch.zeros(batch_size, height, width, dtype=torch.long, device=self.device)
        
        # 创建掩码（标记有智能体的位置）
        masks = torch.zeros(batch_size, height, width, dtype=torch.uint8, device=self.device)
        
        # 为每个样本添加一些智能体
        for b in range(batch_size):
            num_agents = np.random.randint(2, 6)
            for i in range(num_agents):
                x = np.random.randint(1, height-1)
                y = np.random.randint(1, width-1)
                actions[b, x, y] = np.random.randint(0, self.action_dim)
                masks[b, x, y] = 1
        
        return {
            "feature": features,
            "action": actions,
            "mask": masks,
            "file_name": [f"test_batch_{i}.mbin" for i in range(batch_size)]
        }
    
    def test_model_forward_pass(self):
        """测试模型前向传播"""
        print("\n🧪 测试模型前向传播...")
        
        model = self.create_mock_model()
        batch = self.create_mock_batch(batch_size=2)
        
        # 前向传播
        with torch.no_grad():
            logits, _ = model(batch["feature"])
        
        # 验证输出形状
        expected_shape = (2, self.action_dim, *self.map_size)
        self.assertEqual(logits.shape, expected_shape, f"模型输出形状应该为{expected_shape}")
        
        # 验证输出值的合理性
        self.assertFalse(torch.isnan(logits).any(), "模型输出不应该包含NaN")
        self.assertFalse(torch.isinf(logits).any(), "模型输出不应该包含Inf")
        
        print("✅ 模型前向传播测试通过")
    
    def test_loss_computation(self):
        """测试损失计算"""
        print("\n🧪 测试损失计算...")
        
        model = self.create_mock_model()
        batch = self.create_mock_batch(batch_size=2)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        
        # 前向传播
        logits, _ = model(batch["feature"])
        
        # 计算损失
        loss = loss_fn(logits, batch["action"])
        masked_loss = loss * batch["mask"].float()
        
        # 验证损失形状
        self.assertEqual(loss.shape, batch["action"].shape, "损失形状应该与动作形状相同")
        self.assertEqual(masked_loss.shape, batch["mask"].shape, "掩码损失形状应该与掩码形状相同")
        
        # 验证损失值
        self.assertFalse(torch.isnan(loss).any(), "损失不应该包含NaN")
        self.assertFalse(torch.isinf(loss).any(), "损失不应该包含Inf")
        self.assertTrue((loss >= 0).all(), "损失应该非负")
        
        # 验证掩码效果
        mask_sum = batch["mask"].sum()
        if mask_sum > 0:
            averaged_loss = masked_loss.sum() / mask_sum
            self.assertGreater(averaged_loss.item(), 0, "平均损失应该大于0")
        
        print("✅ 损失计算测试通过")
    
    def test_optimizer_step(self):
        """测试优化器步骤"""
        print("\n🧪 测试优化器步骤...")
        
        model = self.create_mock_model()
        batch = self.create_mock_batch(batch_size=2)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # 记录初始参数
        initial_params = [p.clone() for p in model.parameters()]
        
        # 前向传播
        logits, _ = model(batch["feature"])
        loss = loss_fn(logits, batch["action"])
        masked_loss = loss * batch["mask"].float()
        
        mask_sum = batch["mask"].sum()
        if mask_sum > 0:
            averaged_loss = masked_loss.sum() / mask_sum
            
            # 反向传播
            optimizer.zero_grad()
            averaged_loss.backward()
            
            # 检查梯度
            has_gradients = False
            for p in model.parameters():
                if p.grad is not None and p.grad.abs().sum() > 0:
                    has_gradients = True
                    break
            
            self.assertTrue(has_gradients, "应该有非零梯度")
            
            # 优化器步骤
            optimizer.step()
            
            # 检查参数是否更新
            params_changed = False
            for initial, current in zip(initial_params, model.parameters()):
                if not torch.equal(initial, current):
                    params_changed = True
                    break
            
            self.assertTrue(params_changed, "参数应该被更新")
        
        print("✅ 优化器步骤测试通过")
    
    def test_training_loop_stability(self):
        """测试训练循环稳定性"""
        print("\n🧪 测试训练循环稳定性...")
        
        model = self.create_mock_model()
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        losses = []
        
        # 运行几个训练步骤
        for step in range(5):
            batch = self.create_mock_batch(batch_size=2)
            
            # 训练步骤
            model.train()
            logits, _ = model(batch["feature"])
            loss = loss_fn(logits, batch["action"])
            masked_loss = loss * batch["mask"].float()
            
            mask_sum = batch["mask"].sum()
            if mask_sum > 0:
                averaged_loss = masked_loss.sum() / mask_sum
                
                optimizer.zero_grad()
                averaged_loss.backward()
                optimizer.step()
                
                losses.append(averaged_loss.item())
        
        # 验证训练稳定性
        self.assertGreater(len(losses), 0, "应该有损失记录")
        
        for loss in losses:
            self.assertFalse(np.isnan(loss), "损失不应该为NaN")
            self.assertFalse(np.isinf(loss), "损失不应该为Inf")
            self.assertGreater(loss, 0, "损失应该大于0")
        
        print(f"✅ 训练循环稳定性测试通过 (运行了{len(losses)}步)")
    
    def test_evaluation_mode(self):
        """测试评估模式"""
        print("\n🧪 测试评估模式...")
        
        model = self.create_mock_model()
        batch = self.create_mock_batch(batch_size=2)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        
        # 训练模式
        model.train()
        with torch.no_grad():
            train_output, _ = model(batch["feature"])
        
        # 评估模式
        model.eval()
        with torch.no_grad():
            eval_output, _ = model(batch["feature"])
        
        # 验证输出形状一致
        self.assertEqual(train_output.shape, eval_output.shape, "训练和评估模式输出形状应该一致")
        
        # 检查模型是否有BatchNorm层
        has_batchnorm = any(isinstance(module, torch.nn.BatchNorm2d) 
                           for name, module in model.named_modules())
        
        if has_batchnorm:
            # 如果有BatchNorm，训练和评估模式输出可能不同，这是正常的
            print("ℹ️ 模型包含BatchNorm层，训练和评估模式输出可能不同（这是正常行为）")
            
            # 验证输出在合理范围内
            self.assertFalse(torch.isnan(eval_output).any(), "评估输出不应该包含NaN")
            self.assertFalse(torch.isinf(eval_output).any(), "评估输出不应该包含Inf")
            
            # 验证输出形状和数值范围合理
            self.assertEqual(eval_output.shape, train_output.shape, "输出形状应该一致")
            
        else:
            # 如果没有BatchNorm等层，输出应该相同
            torch.testing.assert_close(train_output, eval_output, rtol=1e-5, atol=1e-5)
        
        print("✅ 评估模式测试通过")

def run_training_tests():
    """运行所有训练测试"""
    print("🧪 MAPF训练正确性测试")
    print("=" * 60)
    
    # 创建测试套件
    suite = unittest.TestSuite()
    
    # 添加测试
    test_class = TestTrainingCorrectness
    suite.addTest(test_class('test_model_forward_pass'))
    suite.addTest(test_class('test_loss_computation'))
    suite.addTest(test_class('test_optimizer_step'))
    suite.addTest(test_class('test_training_loop_stability'))
    suite.addTest(test_class('test_evaluation_mode'))
    
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
            print(f"  - {test}")
    
    if result.errors:
        print(f"\n❌ 错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    
    if success_rate == 100:
        print(f"\n🎉 所有训练测试通过！")
    elif success_rate >= 80:
        print(f"\n👍 大部分训练测试通过 ({success_rate:.1f}%)")
    else:
        print(f"\n⚠️ 训练测试通过率较低 ({success_rate:.1f}%)")
    
    return result.testsRun == len(result.failures) + len(result.errors)

if __name__ == "__main__":
    success = run_training_tests()
    sys.exit(0 if success else 1) 