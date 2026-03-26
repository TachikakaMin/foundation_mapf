"""
C++扩展模块

包含高性能的C++实现
"""

from .construct_features_native import construct_input_feature as construct_input_feature_cpp
try:
    from .lacam_online_native import generate_lacam_solution_cpp
except ImportError as _lacam_import_error:
    def generate_lacam_solution_cpp(*args, **kwargs):
        raise ImportError(
            "lacam_online_native is not available. Build extensions via tools/build.sh build "
            "or CMake target build-extension."
        ) from _lacam_import_error

__all__ = ["construct_input_feature_cpp", "generate_lacam_solution_cpp"]
