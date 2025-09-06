"""
C++扩展模块

包含高性能的C++实现
"""

from .construct_features_native import construct_input_feature as construct_input_feature_cpp
__all__ = ['construct_input_feature_cpp']