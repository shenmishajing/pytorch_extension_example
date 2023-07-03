# coding:utf-8

import glob
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="default-project",  # 包名字
    version="1.0",  # 包版本
    packages=find_packages(),  # 包
    ext_modules=[
        CUDAExtension(
            name="ops",
            sources=glob.glob("ops/**/*.cpp", recursive=True)
            + glob.glob("ops/**/*.cu", recursive=True),
            include_dirs=[os.path.abspath("ops/include")],
        )  # 待编译文件，及编译函数
    ],
    cmdclass={"build_ext": BuildExtension},  # 执行编译命令设置
)
