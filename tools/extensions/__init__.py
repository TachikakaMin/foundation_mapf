"""
C++扩展模块

包含高性能的C++实现
"""

try:
    from .construct_features_native import construct_input_feature as construct_input_feature_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

__all__ = ['CPP_AVAILABLE']
if CPP_AVAILABLE:
    __all__.append('construct_input_feature_cpp') 