"""
Setup script for arrange_library package
"""

from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent

# 读取 README（如不存在则忽略）
readme_file = here / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="arrange-library",
    version="0.1.0",
    author="yuyongpeng/caoxiaofang",
    author_email="caoxiaofang@novogene.com",
    description="Lane 排机 + 调用 prediction_delivery 的 arrange_library 封装",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://example.com/arrange_library",
    # 打包当前目录下的 Python 包（core、models 等）
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # 单文件模块：封装主逻辑的脚本
    py_modules=["test_full_pipeline_with_model3_copy_v6_copy"],
    include_package_data=True,
    package_data={
        # 打包配置和模型文件（排机用）
        "": [
            "config/*.yaml",
            "config/*.yml",
            "config/*.json",
            "models/**/*.pkl",
            "models/**/*.json",
        ],
    },
    python_requires=">=3.8",
    install_requires=[
        "numpy>=2.2.6",
        "pandas>=2.2.3",
        "loguru>=0.7.3",
        # 依赖已独立打包好的 prediction_delivery 包
        "prediction-delivery>=1.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    zip_safe=False,
)

