"""
备注数据读取与清洗模块
创建时间：2025-11-17 00:00:00
更新时间：2025-12-01 17:20:00
"""
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger
import re

# 需要保持字符串类型的列名列表
STRING_COLUMNS: List[str] = ['ORIGREC', '上机备注', 'MACHINE_NOTE', 'machine_note', 'Index序列', 'INDEXSEQ']


def extract_and_clean_remarks(csv_path: str) -> Dict[str, str]:
    """
    从CSV文件提取并清洗上机备注
    
    Args:
        csv_path: CSV文件路径
    
    Returns:
        Dict[str, str]: {library_id: cleaned_remark}
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到CSV文件: {path}")
    
    logger.info(f"开始提取上机备注: {path}")
    
    # 读取CSV文件，仅对关键列使用字符串类型
    try:
        # 先读取列名
        df_cols = pd.read_csv(path, nrows=0)
        # 确定需要强制为字符串的列
        dtype_dict = {col: str for col in df_cols.columns if col in STRING_COLUMNS}
        df = pd.read_csv(path, dtype=dtype_dict, keep_default_na=False)
    except Exception as e:
        logger.error(f"读取CSV文件失败: {e}")
        raise
    
    # 检查必要的列
    if "ORIGREC" not in df.columns:
        raise ValueError("CSV文件缺少ORIGREC列")
    
    # 查找上机备注列（可能是"上机备注"或"MACHINE_NOTE"）
    remark_column = None
    for col in df.columns:
        if col in ["上机备注", "MACHINE_NOTE", "machine_note"]:
            remark_column = col
            break
    
    if remark_column is None:
        logger.warning("未找到上机备注列，返回空字典")
        return {}
    
    remarks = {}
    total_count = 0
    valid_count = 0
    
    for _, row in df.iterrows():
        library_id = str(row.get("ORIGREC", "")).strip()
        if not library_id:
            continue
        
        total_count += 1
        remark_text = str(row.get(remark_column, "")).strip()
        
        # 清洗备注内容
        cleaned_remark = clean_remark_text(remark_text)
        
        # 只保存非空且有效的备注
        if cleaned_remark:
            remarks[library_id] = cleaned_remark
            valid_count += 1
    
    logger.info(f"备注提取完成 - 总记录: {total_count}, 有效备注: {valid_count}")
    
    return remarks


def clean_remark_text(remark_text: str) -> str:
    """
    清洗备注文本
    
    Args:
        remark_text: 原始备注文本
    
    Returns:
        str: 清洗后的备注文本
    """
    if not remark_text or not isinstance(remark_text, str):
        return ""
    
    # 去除首尾空格
    cleaned = remark_text.strip()
    
    # 如果为空或只包含空白字符，返回空字符串
    if not cleaned or cleaned.isspace():
        return ""
    
    # 统一换行符为空格
    cleaned = cleaned.replace('\n', ' ').replace('\r', ' ')
    
    # 去除多余的空格（多个连续空格替换为单个空格）
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # 去除首尾空格（再次）
    cleaned = cleaned.strip()
    
    # 过滤明显无效的备注（如"nan", "None", "null"等）
    invalid_values = ['nan', 'none', 'null', 'n/a', 'na', '']
    if cleaned.lower() in invalid_values:
        return ""
    
    return cleaned


def extract_remarks_from_libraries(libraries: List) -> Dict[str, str]:
    """
    从文库对象列表中提取上机备注
    
    Args:
        libraries: 文库对象列表（EnhancedLibraryInfo）
    
    Returns:
        Dict[str, str]: {library_id: cleaned_remark}
    """
    remarks = {}
    
    for lib in libraries:
        if hasattr(lib, 'origrec') and hasattr(lib, 'machine_note'):
            library_id = str(lib.origrec).strip()
            remark_text = str(lib.machine_note) if lib.machine_note else ""
            
            cleaned_remark = clean_remark_text(remark_text)
            if cleaned_remark:
                remarks[library_id] = cleaned_remark
    
    logger.info(f"从{len(libraries)}个文库中提取了{len(remarks)}个有效备注")
    
    return remarks

