"""
测试工具模块

包含：
- 正确性测试：验证数据加载器和训练函数的正确性
- 性能测试：benchmark和性能对比
- 集成测试：端到端功能验证
- 单元测试：各个组件的独立测试
"""

from .run_all_tests import main as run_all_tests
from .test_dataloader_correctness import run_dataloader_tests
from .test_training_correctness import run_training_tests 