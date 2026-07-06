#!/usr/bin/env python3
"""Build PyTorch binding for FPS operator"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='fps_ops',
    ext_modules=[
        CppExtension(
            'fps_ops',
            ['fps_binding.cpp'],
            extra_compile_args=['-std=c++17'],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
