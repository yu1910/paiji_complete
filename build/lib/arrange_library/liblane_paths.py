"""
LibLane V2统一路径和导入管理系统
解决54个文件中的sys.path.append硬编码问题
创建时间：2025-08-21
更新时间：2025-12-01 00:00:00
"""

import sys
import os
from pathlib import Path
from typing import Optional


class LibLanePathManager:
    """LibLane V2项目路径统一管理器"""
    
    _instance: Optional['LibLanePathManager'] = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LibLanePathManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._setup_paths()
            LibLanePathManager._initialized = True
    
    def _setup_paths(self):
        """设置项目路径"""
        # 获取项目根目录
        current_file = Path(__file__).resolve()
        self.project_root = current_file.parent
        
        # 定义关键路径
        self.core_path = self.project_root / "core"
        self.models_path = self.project_root / "models"  
        self.config_path = self.project_root / "config"
        self.tests_path = self.project_root / "tests"
        self.docs_path = self.project_root / "docs"
        self.data_path = self.project_root / "data"
        self.output_path = self.project_root / "output"
        
        # 添加到Python路径（仅添加一次）
        paths_to_add = [
            str(self.project_root),
            str(self.core_path),
            str(self.models_path)
        ]
        
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
    
    @property
    def root(self) -> Path:
        """获取项目根目录"""
        return self.project_root
    
    def get_relative_path(self, target_path: str) -> Path:
        """获取相对于项目根目录的路径"""
        return self.project_root / target_path
    
    def ensure_path_exists(self, path: Path) -> None:
        """确保路径存在，不存在则创建"""
        path.mkdir(parents=True, exist_ok=True)


# 全局路径管理器实例
_path_manager = LibLanePathManager()


def # setup_liblane_paths():
    """设置LibLane项目路径（供其他模块调用）"""
    global _path_manager
    return _path_manager


def get_project_root() -> Path:
    """获取项目根目录路径"""
    return _path_manager.root


def get_core_path() -> Path:
    """获取core模块路径"""
    return _path_manager.core_path


def get_models_path() -> Path:
    """获取models模块路径"""
    return _path_manager.models_path


def get_config_path() -> Path:
    """获取config路径"""
    return _path_manager.config_path


def get_tests_path() -> Path:
    """获取tests路径"""
    return _path_manager.tests_path


# 自动初始化路径设置
# setup_liblane_paths()


# 便捷的导入别名，供其他模块使用
__all__ = [
    'setup_liblane_paths',
    'get_project_root', 
    'get_core_path',
    'get_models_path',
    'get_config_path',
    'get_tests_path',
    'LibLanePathManager'
]