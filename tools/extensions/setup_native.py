from distutils.core import setup, Extension
import numpy as np

# 定义扩展模块
ext_modules = [
    Extension(
        'construct_features_native',
        sources=['construct_features_native.cpp'],
        include_dirs=[np.get_include()],
        language='c++',
        extra_compile_args=['-O3', '-std=c++17'],
        extra_link_args=['-std=c++17'],
    )
]

setup(
    name='construct_features_native',
    ext_modules=ext_modules,
    zip_safe=False,
) 