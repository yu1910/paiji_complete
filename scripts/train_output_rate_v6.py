"""
训练下单量预测模型 V6 - 使用XGBoost回归 + 特征权重 + 样本权重
创建时间：2026-02-04 11:30:00
更新时间：2026-02-09 10:00:00

业务逻辑：
  1. 直接预测下单量 = f(合同量, 质量特征)
  2. 目标：预测下单量使得产出 >= 合同，且不过度浪费
  3. wkcontractdata 是真实合同量（非产出值）

修复说明 (2026-02-09):
  1. 数据修复：wkcontractdata 保留真实合同量，不再用产出值替换
     - 旧版本训练时用产出值，推理时用合同量，导致分布不一致
  2. 交互特征修复：分位数阈值在训练集上计算后保存
     - 旧版本每个数据集独立计算分位数，训练/验证/测试阈值不同
     - 修复后训练集计算阈值并保存，推理时加载使用
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# 数据目录（预先划分好的训练集）
DATA_DIR = Path("/data/work/yuyongpeng/liblane_v2_deepseek/data/train_data_v6_lorderdata")
OUTPUT_DIR = Path("/data/work/yuyongpeng/liblane_v2_deepseek/models/output_rate_v6")

# ==================== 文库维度特征 ====================
NUMERIC_FEATURES: List[str] = [
    "wkqpcr",           # 13%
    "wkcontractdata",   # 22% - 真实合同量（训练和推理均为同一含义）
    "wkpeaksize",       # 6%
    "wkxpd",            # 6%
    "wkadaptorrate",    # 6% - 接头比值
]

CATEGORICAL_FEATURES: List[str] = [
    "wksampletype",             # 20% - 文库类型
    "sampletype_output_group",  # 新增：文库类型产出分组（高/中/低产出）
    "wkjkhj",                   # 10% - 文库构建背景
    "wkcomplexresult",          # 6% - 库建综合结果
    "wkpeakmap",                # 6% - 峰图描述
    "wk_jjbj",                  # 5% - 碱基不均
]

# ==================== Lane维度聚合特征 ====================
# 注意：Lane特征现在动态读取，不再硬编码
# 新的Lane特征包括：
#  - lane_jjbj_ratio: 碱基不均文库占比
#  - lane_single_index_ratio: 单端index占比
#  - lane_10bp_ratio: 10碱基数据量占比
#  - lane_addtest_ratio: 加测文库占比
#  - lane_sampletype_*_ratio: 各文库类型占比（动态）
#  - lane_sampletype_diversity: 文库类型多样性
#  - lane_contract_mean/median/std: 合同量统计
#  - lane_library_count: Lane内文库数量
# 这些都是排机后、测序前可知的特征（无数据泄露）
LANE_NUMERIC_FEATURES: List[str] = []  # 动态从数据中读取

LANE_CATEGORICAL_FEATURES: List[str] = []

# ==================== 特征权重 ====================
FEATURE_WEIGHTS: Dict[str, float] = {
    # 文库维度
    "wkcontractdata": 0.22,    # 真实合同量（训练和推理一致）
    "wksampletype": 0.20,
    "sampletype_output_group": 0.15,  # 新增：文库类型产出分组（高/中/低产出）
    "wkqpcr": 0.13,
    "wkjkhj": 0.10,
    "wkcomplexresult": 0.06,
    "wkpeakmap": 0.06,
    "wkpeaksize": 0.06,
    "wkxpd": 0.06,
    "wkadaptorrate": 0.06,
    "wk_jjbj": 0.05,
    # Lane维度特征（动态权重，默认0.05）
    # 因为有144个Lane特征，无法逐一指定权重
}

# 关键：预测下单量
TARGET_COL = "lorderdata"  # 下单量


def load_split_data(split_name: str, sample_frac: float = 1.0) -> pd.DataFrame:
    """
    加载预先划分好的数据集
    
    Args:
        split_name: 'train', 'val', 或 'test'
        sample_frac: 采样比例，默认1.0（全部数据）
    
    Returns:
        预处理后的DataFrame
    """
    file_path = DATA_DIR / f"{split_name}.csv"
    logger.info(f"加载{split_name}数据: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    logger.info(f"原始{split_name}数据: {len(df):,} 行")
    
    # 采样（仅对训练集采样，验证集和测试集保持完整）
    if split_name == "train" and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
        logger.info(f"训练集采样 {sample_frac*100:.0f}%: {len(df):,} 行")
    
    # 转换数值类型
    df["loutput"] = pd.to_numeric(df["loutput"], errors="coerce")
    df["lorderdata"] = pd.to_numeric(df["lorderdata"], errors="coerce")
    df["wkcontractdata"] = pd.to_numeric(df["wkcontractdata"], errors="coerce")
    
    # 过滤有效数据（下单量和产出都需要>0）
    df = df[(df["lorderdata"] > 0) & (df["loutput"] > 0) & (df["wkcontractdata"] > 0)].copy()
    logger.info(f"有效{split_name}数据: {len(df):,} 行")
    
    # 过滤下单量极值（保留合理范围：1-500G）
    min_order, max_order = 1.0, 500.0
    before = len(df)
    df = df[df[TARGET_COL].between(min_order, max_order)].copy()
    after = len(df)
    logger.info(f"{split_name}下单量过滤 [{min_order}, {max_order}]: {before:,} → {after:,} 行")
    
    # 下单量统计
    logger.info(f"{split_name}下单量统计: 均值={df[TARGET_COL].mean():.3f}G, "
                f"中位={df[TARGET_COL].median():.3f}G, "
                f"std={df[TARGET_COL].std():.3f}G")
    
    return df


def load_and_preprocess_data() -> pd.DataFrame:
    """加载并预处理数据，计算产率作为目标变量（兼容旧接口）"""
    return load_split_data("train")


def create_interaction_features(
    df: pd.DataFrame,
    quantile_thresholds: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    基于层级关系的完整交互特征生成

    层级结构（根据用户定义的缩进关系）：
      第1层：文库类型 (wksampletype) - 20%
        第2层：文库构建背景 (wkjkhj) - 10%
          第3层：库建综合结果 (wkcomplexresult) - 6%
            第4层：qpcr - 13% | 峰图 (wkpeakmap) - 6%
              第5层：peaksize - 6% | 小片段(xpd) - 6% | 接头比值(adaptorrate) - 6%
      独立分支A：合同数据量 (wkcontractdata) - 22%
      独立分支B：碱基不均 (wk_jjbj) - 5%

    分位数阈值机制 (2026-02-09 修复):
      旧版本：每个数据集（训练/验证/测试）各自独立计算 quantile，导致阈值不一致。
      修复后：训练时传 quantile_thresholds=None 会从训练数据计算并返回阈值，
             验证/测试/推理时传入训练集的阈值，确保一致性。

    Args:
        df: 输入数据
        quantile_thresholds: 分位数阈值字典。
            None = 训练模式（自动计算并返回）；
            非None = 推理模式（使用传入的阈值）。

    Returns:
        (处理后的DataFrame, 分位数阈值字典)
    """
    df = df.copy()
    is_training = (quantile_thresholds is None)
    thresholds = {} if is_training else dict(quantile_thresholds)

    # 确保数值类型
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if is_training:
        logger.info("开始生成交互特征（训练模式：计算分位数阈值）...")
    else:
        logger.info("开始生成交互特征（推理模式：使用已有阈值）...")

    # ================================================================
    # Phase 1: 连续派生特征（不依赖分位数）
    # ================================================================
    logger.info("  [层级5] 峰图子特征内部交互: peaksize, xpd, adaptorrate")

    # 峰图综合质量评分（peaksize高 + xpd低 + adaptorrate低 = 质量好）
    df["L5_peak_quality"] = df["wkpeaksize"] * (1 - df["wkxpd"]) * (1 - df["wkadaptorrate"])
    # 峰图风险评分
    df["L5_peak_risk"] = df["wkxpd"] + df["wkadaptorrate"]
    # 接头污染风险
    df["L5_adaptor_risk"] = df["wkadaptorrate"]
    # 小片段风险
    df["L5_xpd_risk"] = df["wkxpd"]

    logger.info("  [层级4] qpcr与峰图子特征协同")
    # qpcr x 峰图综合质量
    df["L4_qpcr_x_peakquality"] = df["wkqpcr"] * df["L5_peak_quality"]
    # qpcr与peaksize的直接交互
    df["L4_qpcr_x_peaksize"] = df["wkqpcr"] * df["wkpeaksize"]
    # qpcr与xpd的风险交互
    df["L4_qpcr_xpd_risk"] = df["wkqpcr"] * df["wkxpd"]
    # qpcr与adaptorrate的风险交互
    df["L4_qpcr_adaptor_risk"] = df["wkqpcr"] * df["wkadaptorrate"]

    logger.info("  [层级3-4] 库建结果与qpcr/峰图交互")
    # 库建结果风险编码（风险/不合格=1，合格=0）
    df["L3_complex_risk_flag"] = df["wkcomplexresult"].astype(str).str.contains(
        "风险|不合格|异常", case=False, na=False
    ).astype(int)
    # 库建风险下的qpcr惩罚
    df["L3_complex_qpcr"] = df["wkqpcr"] * (1 - 0.3 * df["L3_complex_risk_flag"])
    # 库建风险下的峰图质量惩罚
    df["L3_complex_peakquality"] = df["L5_peak_quality"] * (1 - 0.3 * df["L3_complex_risk_flag"])

    logger.info("  [层级2-3] 构建背景与库建结果交互")
    # 构建背景编码（诺禾自动=0，诺禾手工=1，客户自建=2风险最高）
    df["L2_jkhj_str"] = df["wkjkhj"].astype(str).fillna("未知")
    df["L2_jkhj_risk"] = 0
    df.loc[df["L2_jkhj_str"].str.contains("手工", na=False), "L2_jkhj_risk"] = 1
    df.loc[df["L2_jkhj_str"].str.contains("自建|客户", na=False), "L2_jkhj_risk"] = 2
    # 构建背景风险下的质量惩罚
    df["L2_jkhj_quality"] = df["L4_qpcr_x_peakquality"] * (1 - 0.15 * df["L2_jkhj_risk"])
    # 客户自建+库建风险的高风险标记
    df["L2_jkhj_complex_risk"] = (
        (df["L2_jkhj_risk"] >= 1) & (df["L3_complex_risk_flag"] == 1)
    ).astype(int)

    # ================================================================
    # Phase 2: 计算第一批分位数阈值（训练模式）
    # 这些阈值基于原始特征和 Phase 1 派生特征
    # ================================================================
    if is_training:
        thresholds["wkxpd_q70"] = float(df["wkxpd"].quantile(0.7))
        thresholds["wkadaptorrate_q70"] = float(df["wkadaptorrate"].quantile(0.7))
        thresholds["wkpeaksize_q30"] = float(df["wkpeaksize"].quantile(0.3))
        thresholds["wkqpcr_q60"] = float(df["wkqpcr"].quantile(0.6))
        thresholds["L5_peak_quality_q40"] = float(df["L5_peak_quality"].quantile(0.4))
        thresholds["L4_qpcr_x_peakquality_q40"] = float(
            df["L4_qpcr_x_peakquality"].quantile(0.4)
        )
        logger.info(f"  [阈值] 第一批阈值已从训练数据计算（共 6 个）")

    # ================================================================
    # Phase 3: 应用第一批分位数阈值生成二值标记
    # ================================================================
    # 双重风险（xpd高 且 adaptorrate高）
    df["L5_double_risk"] = (
        (df["wkxpd"] > thresholds["wkxpd_q70"]) &
        (df["wkadaptorrate"] > thresholds["wkadaptorrate_q70"])
    ).astype(int)

    # 峰图异常标记（peaksize低 且 xpd高或adaptorrate高）
    df["L5_peak_abnormal"] = (
        (df["wkpeaksize"] < thresholds["wkpeaksize_q30"]) &
        (
            (df["wkxpd"] > thresholds["wkxpd_q70"]) |
            (df["wkadaptorrate"] > thresholds["wkadaptorrate_q70"])
        )
    ).astype(int)

    # 质量不一致标记（高qpcr但峰图差）
    df["L4_qpcr_peak_mismatch"] = (
        (df["wkqpcr"] > thresholds["wkqpcr_q60"]) &
        (df["L5_peak_quality"] < thresholds["L5_peak_quality_q40"])
    ).astype(int)

    # 库建风险且质量差的高风险标记
    df["L3_complex_quality_risk"] = (
        (df["L3_complex_risk_flag"] == 1) &
        (df["L4_qpcr_x_peakquality"] < thresholds["L4_qpcr_x_peakquality_q40"])
    ).astype(int)

    # ================================================================
    # Phase 4: 依赖 Phase 3 标记的派生特征
    # ================================================================
    logger.info("  [层级1-2] 文库类型与构建背景交互")
    # 综合层级质量（从层级1到层级5的综合）
    df["L1_to_L5_quality"] = df["L2_jkhj_quality"] * (1 - 0.2 * df["L5_double_risk"])

    # ================================================================
    # Phase 5: 第二批分位数阈值（依赖 Phase 4 结果）
    # ================================================================
    if is_training:
        thresholds["wkcontractdata_q70"] = float(df["wkcontractdata"].quantile(0.7))
        thresholds["L1_to_L5_quality_q30"] = float(df["L1_to_L5_quality"].quantile(0.3))
        logger.info(f"  [阈值] 第二批阈值已从训练数据计算（共 2 个）")

    # ================================================================
    # Phase 6: 应用第二批阈值生成二值标记
    # ================================================================
    logger.info("  [独立分支A] 合同量与各层级交互")
    # 合同量 x 最终质量
    df["A_contract_x_quality"] = df["wkcontractdata"] * df["L1_to_L5_quality"]
    # 合同量 x qpcr
    df["A_contract_x_qpcr"] = df["wkcontractdata"] * df["wkqpcr"]
    # 合同量 x 峰图质量
    df["A_contract_x_peakquality"] = df["wkcontractdata"] * df["L5_peak_quality"]
    # 质量密度（单位合同量的质量）
    with np.errstate(divide="ignore", invalid="ignore"):
        df["A_quality_per_contract"] = df["L1_to_L5_quality"] / (df["wkcontractdata"] + 1e-6)
    df["A_quality_per_contract"] = df["A_quality_per_contract"].replace(
        [np.inf, -np.inf], 0
    ).fillna(0)

    # 高需求低质量风险（使用训练集阈值）
    df["A_high_demand_low_quality"] = (
        (df["wkcontractdata"] > thresholds["wkcontractdata_q70"]) &
        (df["L1_to_L5_quality"] < thresholds["L1_to_L5_quality_q30"])
    ).astype(int)
    # 高需求+构建背景风险
    df["A_high_demand_jkhj_risk"] = (
        (df["wkcontractdata"] > thresholds["wkcontractdata_q70"]) &
        (df["L2_jkhj_risk"] >= 1)
    ).astype(int)

    # ================================================================
    # Phase 7: 碱基不均分支（不依赖分位数）
    # ================================================================
    logger.info("  [独立分支B] 碱基不均与各层级交互")
    # 碱基不均数值化
    if df["wk_jjbj"].dtype == 'object':
        df["B_jjbj_flag"] = df["wk_jjbj"].astype(str).str.contains(
            "不均|1|是", na=False
        ).astype(int)
    else:
        df["B_jjbj_flag"] = (
            pd.to_numeric(df["wk_jjbj"], errors="coerce").fillna(0) > 0
        ).astype(int)
    # 碱基不均 x qpcr惩罚
    df["B_jjbj_qpcr"] = df["wkqpcr"] * (1 - 0.2 * df["B_jjbj_flag"])
    # 碱基不均 x 峰图质量惩罚
    df["B_jjbj_peakquality"] = df["L5_peak_quality"] * (1 - 0.2 * df["B_jjbj_flag"])
    # 碱基不均 x 综合质量惩罚
    df["B_jjbj_total_quality"] = df["L1_to_L5_quality"] * (1 - 0.3 * df["B_jjbj_flag"])
    # 碱基不均+库建风险的双重风险
    df["B_jjbj_complex_double_risk"] = (
        (df["B_jjbj_flag"] == 1) & (df["L3_complex_risk_flag"] == 1)
    ).astype(int)
    # 碱基不均+构建背景风险的双重风险
    df["B_jjbj_jkhj_double_risk"] = (
        (df["B_jjbj_flag"] == 1) & (df["L2_jkhj_risk"] >= 1)
    ).astype(int)

    # ================================================================
    # Phase 8: Lane维度特征 + 综合评分
    # ================================================================
    logger.info("  [Lane维度] 直接使用Lane聚合特征，不做交互")
    for col in LANE_NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    logger.info("  [综合] 最终综合评分")
    # 最终综合质量评分
    df["final_quality_score"] = (
        df["L1_to_L5_quality"] *
        (1 - 0.3 * df["B_jjbj_flag"]) *
        np.log1p(df["wkcontractdata"] / 100 + 1)
    )
    # 综合风险评分
    df["final_risk_score"] = (
        df["L2_jkhj_risk"] +
        df["L3_complex_risk_flag"] +
        df["L4_qpcr_peak_mismatch"] +
        df["L5_double_risk"] +
        df["B_jjbj_flag"]
    )

    # 清理临时列
    df.drop(columns=["L2_jkhj_str"], inplace=True, errors="ignore")

    # 统计
    interaction_cols = [
        col for col in df.columns
        if col.startswith(("L1", "L3", "L4", "L5", "A_", "B_", "final_"))
    ]
    logger.info(f"交互特征生成完成，新增 {len(interaction_cols)} 个特征，"
                f"阈值 {len(thresholds)} 个")

    return df, thresholds


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """分层采样"""
    logger.info("开始分层采样...")
    
    try:
        df["rate_bin"] = pd.qcut(df[TARGET_COL], q=3, labels=False, duplicates="drop")
        
        train_df, temp_df = train_test_split(
            df, test_size=(1 - train_ratio),
            stratify=df["rate_bin"],
            random_state=random_state
        )
        
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df, test_size=(1 - val_test_ratio),
            stratify=temp_df["rate_bin"],
            random_state=random_state
        )
        
    except Exception as e:
        logger.warning(f"分层采样失败: {e}，使用简单随机分割")
        train_df, temp_df = train_test_split(
            df, test_size=(1 - train_ratio), random_state=random_state
        )
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df, test_size=(1 - val_test_ratio), random_state=random_state
        )
    
    for split_df in [train_df, val_df, test_df]:
        split_df.drop(columns=["rate_bin"], inplace=True, errors="ignore")
    
    logger.info(f"分层采样完成: 训练={len(train_df):,}, 验证={len(val_df):,}, 测试={len(test_df):,}")
    return train_df, val_df, test_df


def encode_categorical(
    df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder],
    fit: bool
) -> pd.DataFrame:
    """编码分类特征（包括文库维度和Lane维度）"""
    df = df.copy()
    # 所有分类特征 = 文库维度 + Lane维度
    all_categorical = CATEGORICAL_FEATURES + LANE_CATEGORICAL_FEATURES
    
    for col in all_categorical:
        if col not in df.columns:
            logger.warning(f"缺少分类特征: {col}")
            df[col] = "缺失"
        if fit:
            enc = LabelEncoder()
            df[col] = df[col].fillna("缺失").astype(str)
            df[col] = enc.fit_transform(df[col])
            encoders[col] = enc
        else:
            enc = encoders[col]
            df[col] = df[col].fillna("缺失").astype(str)
            df[col] = df[col].apply(lambda x: x if x in enc.classes_ else "缺失")
            if "缺失" not in enc.classes_:
                enc.classes_ = np.append(enc.classes_, "缺失")
            df[col] = enc.transform(df[col])
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """准备特征矩阵"""
    # 文库维度交互特征（精简版）
    lib_interaction_cols = [
        # 层级5：峰图子特征内部（peaksize × xpd × adaptorrate）
        "L5_peak_quality", "L5_peak_risk", "L5_adaptor_risk", "L5_xpd_risk", 
        "L5_double_risk", "L5_peak_abnormal",
        # 层级4：qpcr与峰图协同
        "L4_qpcr_x_peakquality", "L4_qpcr_x_peaksize", "L4_qpcr_xpd_risk", 
        "L4_qpcr_adaptor_risk", "L4_qpcr_peak_mismatch",
        # 层级3-4：库建结果交互
        "L3_complex_risk_flag", "L3_complex_qpcr", "L3_complex_peakquality", "L3_complex_quality_risk",
        # 层级2-3：构建背景交互
        "L2_jkhj_risk", "L2_jkhj_quality", "L2_jkhj_complex_risk",
        # 层级1-5综合
        "L1_to_L5_quality",
        # 独立分支A：合同量（精简：保留核心3个）
        "A_contract_x_quality",       # 合同量×综合质量
        "A_quality_per_contract",     # 质量密度
        "A_high_demand_low_quality",  # 高需求低质量风险
        # 独立分支B：碱基不均（精简：保留核心2个）
        "B_jjbj_flag",                # 碱基不均标记
        "B_jjbj_total_quality",       # 碱基不均×综合质量
    ]
    
    # 综合评分（已去掉非线性变换）
    summary_cols = [
        "final_quality_score",   # 最终质量评分
        "final_risk_score",      # 综合风险评分
    ]
    
    # 动态获取Lane特征（所有以lane_开头的列，除了lane_unique_id）
    lane_features = [col for col in df.columns if str(col).startswith('lane_') and col != 'lane_unique_id']
    
    # 所有特征列（Lane维度直接使用原始特征，不做交互）
    feature_cols = (
        NUMERIC_FEATURES +           # 文库数值特征
        CATEGORICAL_FEATURES +       # 文库分类特征
        lane_features +              # Lane统计特征（动态读取）
        lib_interaction_cols +       # 文库交互特征
        summary_cols                 # 综合评分
    )
    available_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[available_cols].copy()
    y = df[TARGET_COL]  # 下单量
    output = df["loutput"]  # 真实产出（用于评估）
    
    logger.info(f"特征准备完成: {len(available_cols)} 个特征（文库{len(NUMERIC_FEATURES)+len(CATEGORICAL_FEATURES)}个 + Lane{len(lane_features)}个 + 交互{len(lib_interaction_cols)}个 + 综合{len(summary_cols)}个）")
    return X, y, output


def compute_feature_weights(X_train: pd.DataFrame) -> np.ndarray:
    """
    计算每个特征的权重（基于用户定义的业务权重）
    
    原始特征使用用户权重，交互特征根据参与的原始特征计算权重
    Lane特征使用默认权重
    """
    feature_weights = []
    
    for col in X_train.columns:
        # 原始特征：直接使用用户定义的权重
        if col in FEATURE_WEIGHTS:
            weight = FEATURE_WEIGHTS[col]
        
        # Lane统计特征：使用默认权重
        elif str(col).startswith('lane_'):
            # Lane特征的默认权重取决于其重要性
            if 'jjbj' in str(col) or 'contract' in str(col) or 'diversity' in str(col):
                weight = 0.08  # 核心Lane统计特征
            else:
                weight = 0.03  # 文库类型占比等特征
        
        # 交互特征：根据命名规则推断权重
        elif col.startswith('L5_'):  # 峰图子特征交互
            # peaksize(6%) + xpd(6%) + adaptorrate(6%)
            weight = (0.06 + 0.06 + 0.06) / 3
        elif col.startswith('L4_'):  # qpcr与峰图协同
            # qpcr(13%) + 峰图相关(6%)
            weight = (0.13 + 0.06) / 2
        elif col.startswith('L3_'):  # 库建结果交互
            # complexresult(6%) + qpcr(13%)或峰图(6%)
            weight = (0.06 + 0.10) / 2
        elif col.startswith('L2_'):  # 构建背景交互
            # jkhj(10%) + complexresult(6%)
            weight = (0.10 + 0.06) / 2
        elif col.startswith('L1_'):  # 全链路质量
            # 所有特征参与，使用平均权重
            weight = 0.10
        elif col.startswith('A_'):  # 合同量交互
            # contractdata(22%) + 其他
            weight = 0.15
        elif col.startswith('B_'):  # 碱基不均交互
            # wk_jjbj(5%) + 其他
            weight = 0.08
        elif col.startswith('final_'):  # 综合评分
            weight = 0.10
        else:
            # 默认权重
            weight = 0.05
        
        feature_weights.append(weight)
    
    # 归一化到和为特征数（XGBoost要求）
    feature_weights = np.array(feature_weights)
    feature_weights = feature_weights / feature_weights.sum() * len(feature_weights)
    
    logger.info(f"特征权重范围: min={feature_weights.min():.4f}, max={feature_weights.max():.4f}, mean={feature_weights.mean():.4f}")
    
    return feature_weights


def compute_sample_weights(
    df: pd.DataFrame, 
    strategy: str = "tiered"
) -> np.ndarray:
    """
    计算样本权重：对小文库赋予更高权重
    
    Args:
        df: 数据框（需包含loutput列，真实产出值）
        strategy: 权重策略
            - "tiered": 分层权重（针对不同质量的小文库）
            - "simple": 简单权重（阈值+倍数）
    
    Returns:
        样本权重数组
    """
    output_vals = df["loutput"].values
    weights = np.ones(len(df))
    
    if strategy == "tiered":
        # 分层权重策略（基于数据分析的达标率）
        # 0-3G: 达标率56.6% → 权重5.0x
        # 3-5G: 达标率38.7%（最差）→ 权重6.0x
        # 5-7G: 达标率43.3% → 权重4.0x
        # 7-10G: 达标率87.8% → 权重1.5x
        # >10G: 达标率70%+ → 权重1.0x
        
        mask_0_3 = (output_vals >= 0) & (output_vals < 3)
        mask_3_5 = (output_vals >= 3) & (output_vals < 5)
        mask_5_7 = (output_vals >= 5) & (output_vals < 7)
        mask_7_10 = (output_vals >= 7) & (output_vals < 10)
        mask_10_plus = output_vals >= 10
        
        weights[mask_0_3] = 5.0
        weights[mask_3_5] = 6.0   # 最差段，最高权重
        weights[mask_5_7] = 4.0
        weights[mask_7_10] = 1.5
        weights[mask_10_plus] = 1.0
        
        logger.info("样本权重策略: 分层权重（基于历史达标率）")
        logger.info(f"  0-3G:   {mask_0_3.sum():>7,}个样本 ({mask_0_3.sum()/len(df)*100:>5.1f}%), 权重=5.0x (达标率56.6%)")
        logger.info(f"  3-5G:   {mask_3_5.sum():>7,}个样本 ({mask_3_5.sum()/len(df)*100:>5.1f}%), 权重=6.0x (达标率38.7% 最差)")
        logger.info(f"  5-7G:   {mask_5_7.sum():>7,}个样本 ({mask_5_7.sum()/len(df)*100:>5.1f}%), 权重=4.0x (达标率43.3%)")
        logger.info(f"  7-10G:  {mask_7_10.sum():>7,}个样本 ({mask_7_10.sum()/len(df)*100:>5.1f}%), 权重=1.5x (达标率87.8%)")
        logger.info(f"  10G+:   {mask_10_plus.sum():>7,}个样本 ({mask_10_plus.sum()/len(df)*100:>5.1f}%), 权重=1.0x (达标率70%+)")
        logger.info(f"  整体统计: min={weights.min():.2f}, max={weights.max():.2f}, mean={weights.mean():.2f}")
        
    else:
        # 简单权重策略（保留，向后兼容）
        threshold = 7.0
        small_weight = 3.0
        small_mask = output_vals < threshold
        
        if small_mask.sum() > 0:
            weights[small_mask] = small_weight * np.sqrt(threshold / output_vals[small_mask])
            weights[small_mask] = np.clip(weights[small_mask], 1.0, small_weight)
        
        logger.info(f"样本权重策略: 简单权重（阈值={threshold}G, 最大倍数={small_weight}x）")
        logger.info(f"  小文库(<{threshold}G): {small_mask.sum():,}个样本, 平均权重={weights[small_mask].mean():.2f}x")
        logger.info(f"  大文库(>={threshold}G): {(~small_mask).sum():,}个样本, 平均权重={weights[~small_mask].mean():.2f}x")
    
    return weights


def train_regression_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weights: np.ndarray = None
) -> xgb.XGBRegressor:
    """训练普通回归模型（使用XGBoost，支持特征权重和样本权重）"""
    logger.info("训练XGBoost回归模型（支持特征权重+样本权重）...")
    logger.info(f"训练集大小: {len(X_train):,}, 特征数: {X_train.shape[1]}")
    
    # 计算特征权重
    feature_weights = compute_feature_weights(X_train)
    
    # 使用XGBoost（支持feature_weights参数）
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=2000,           # 迭代次数
        max_depth=10,                # 树深度
        learning_rate=0.01,          # 学习率
        min_child_weight=20,         # 最小子节点权重
        subsample=0.9,               # 行采样
        colsample_bytree=0.9,        # 列采样
        reg_alpha=0.05,              # L1正则化
        reg_lambda=0.1,              # L2正则化
        random_state=42,
        n_jobs=8,
        tree_method='hist',          # 使用histogram算法加速
        feature_weights=feature_weights,  # 🔥 关键：特征权重
        eval_metric='mae',
    )
    
    logger.info(f"模型参数: n_estimators=2000, max_depth=10, lr=0.01")
    logger.info(f"🔥 已应用用户定义的特征权重（合同量22%, 文库类型20%, qpcr13%等）")
    if sample_weights is not None:
        logger.info(f"🔥 已应用样本权重（小文库权重更高，减少小文库预测误差）")
    
    # 训练（XGBoost）
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weights,  # 🔥 关键：样本权重
        verbose=100
    )
    
    # XGBoost不使用early_stopping时，best_iteration为None
    best_iter = model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration is not None else model.n_estimators
    logger.info(f"XGBoost模型训练完成，训练轮数: {best_iter}")
    return model


def calc_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_model(
    model,
    X: pd.DataFrame,
    y_order: pd.Series,
    output: pd.Series,
    name: str,
    safety_factor: float = 1.0
) -> Dict:
    """
    评估下单量预测模型
    
    逻辑说明：
    - 训练和推理的 wkcontractdata 均为真实合同量
    - 模型学习：合同量 + 质量特征 -> 下单量
    - 评估：对比预测下单量与真实下单量，并检查产出是否达标
    """
    pred_order = model.predict(X)  # 预测下单量
    true_order = y_order.values     # 真实下单量
    true_output = output.values     # 真实产出
    
    # === 1. 下单量预测准确性 ===
    mae = mean_absolute_error(true_order, pred_order)
    rmse = np.sqrt(mean_squared_error(true_order, pred_order))
    r2 = r2_score(true_order, pred_order)
    mape = calc_mape(true_order, pred_order)
    
    # === 2. 业务模拟：使用预测下单量计算产率 ===
    # 预测产率 = 真实产出 / 预测下单量
    with np.errstate(divide='ignore', invalid='ignore'):
        pred_rate = true_output / pred_order
    pred_rate = np.where(np.isinf(pred_rate) | np.isnan(pred_rate), 0, pred_rate)
    
    # 真实产率 = 真实产出 / 真实下单量
    with np.errstate(divide='ignore', invalid='ignore'):
        true_rate = true_output / true_order
    true_rate = np.where(np.isinf(true_rate) | np.isnan(true_rate), 0, true_rate)
    
    # === 3. 业务达标分析 ===
    # 核心目标：预测下单量能让产出>=合同（这里产出=wkcontractdata字段值）
    # 达标：预测产率 >= 1.0（即预测下单量 <= 产出）
    达标率 = float(np.mean(pred_rate >= 1.0) * 100)
    # 基本达标：预测产率 >= 0.95
    基本达标率 = float(np.mean(pred_rate >= 0.95) * 100)
    # 理想：预测产率 在 [1.0, 1.2]（产出略多于下单，不浪费）
    理想率 = float(np.mean((pred_rate >= 1.0) & (pred_rate <= 1.2)) * 100)
    # 过多浪费：预测产率 > 1.3
    过多率 = float(np.mean(pred_rate > 1.3) * 100)
    # 不足：预测产率 < 0.95
    不足率 = float(np.mean(pred_rate < 0.95) * 100)
    
    # 平均产率
    avg_pred_rate = float(np.mean(pred_rate[pred_rate > 0]))
    median_pred_rate = float(np.median(pred_rate[pred_rate > 0]))
    
    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape),
        "达标率": 达标率,
        "基本达标率": 基本达标率,
        "理想率": 理想率,
        "过多率": 过多率,
        "不足率": 不足率,
        "平均预测产率": avg_pred_rate,
        "中位预测产率": median_pred_rate,
    }
    
    logger.info(f"{name}集评估结果:")
    logger.info(f"  [1] 下单量预测准确性")
    logger.info(f"      MAE={mae:.4f}G, MAPE={mape:.2f}%, R²={r2:.4f}")
    logger.info(f"  [2] 业务达标分析（基于预测下单量的产率）")
    logger.info(f"      达标率={达标率:.1f}% (预测产率>=1.0)")
    logger.info(f"      基本达标率={基本达标率:.1f}% (预测产率>=0.95)")
    logger.info(f"      理想率={理想率:.1f}% (预测产率在1.0-1.2)")
    logger.info(f"  [3] 风险分析")
    logger.info(f"      过多率={过多率:.1f}% (预测产率>1.3，下单太少)")
    logger.info(f"      不足率={不足率:.1f}% (预测产率<0.95，下单太多)")
    logger.info(f"  [4] 预测产率")
    logger.info(f"      均值={avg_pred_rate:.3f}, 中位={median_pred_rate:.3f}")
    
    return metrics


def save_artifacts(
    models: Dict[str, xgb.XGBRegressor],
    encoders: Dict[str, LabelEncoder],
    metrics: Dict,
    feature_cols: List[str],
    quantile_thresholds: Dict[str, float]
) -> None:
    """保存模型、编码器、阈值和配置"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        # pickle 保存完整模型
        with open(OUTPUT_DIR / f"model_{model_name}.pkl", "wb") as f:
            pickle.dump(model, f)
        # XGBoost 原生格式保存
        model.save_model(str(OUTPUT_DIR / f"model_{model_name}.json"))

    with open(OUTPUT_DIR / "label_encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 保存分位数阈值（推理时必须加载使用，确保训练-推理一致性）
    with open(OUTPUT_DIR / "quantile_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(quantile_thresholds, f, ensure_ascii=False, indent=2)
    logger.info(f"分位数阈值已保存: {len(quantile_thresholds)} 个阈值")

    config = {
        "model_type": "下单量预测模型（XGBoost回归 + 特征权重）",
        "target": TARGET_COL,
        "target_meaning": "下单量（lorderdata）",
        "training_logic": "wkcontractdata = 真实合同量，模型学习 合同量+质量->下单量",
        "prediction_logic": "推理时 wkcontractdata = 合同量，与训练一致",
        "quantile_thresholds_file": "quantile_thresholds.json",
        "usage": "pred_order = model.predict(X), 其中X['wkcontractdata']=合同量",
        "feature_weights_enabled": True,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_weights": FEATURE_WEIGHTS,
        "feature_cols": feature_cols,
        "models": list(models.keys()),
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(OUTPUT_DIR / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f"模型与配置已保存到: {OUTPUT_DIR}")


def main() -> None:
    """主函数"""
    logger.info("=" * 80)
    logger.info("训练产率预测模型 V6")
    logger.info("目标: 预测产率，然后计算下单量 = 合同量/产率×安全系数")
    logger.info("业务目标: 产出>=合同，产出/合同尽量接近1")
    logger.info(f"数据目录: {DATA_DIR}")
    logger.info("=" * 80)
    
    # 1. 加载预先划分好的数据集
    TRAIN_SAMPLE_FRAC = 1.0  # 训练集采样比例（1.0=全量数据）
    logger.info("步骤1: 加载预划分数据集...")
    train_df = load_split_data("train", sample_frac=TRAIN_SAMPLE_FRAC)
    val_df = load_split_data("val")
    test_df = load_split_data("test")
    
    logger.info(f"数据集大小: 训练={len(train_df):,}, 验证={len(val_df):,}, 测试={len(test_df):,}")
    
    # 2. 创建交互特征（训练集计算阈值，验证/测试集复用）
    logger.info("\n步骤2: 创建交互特征...")
    train_df, quantile_thresholds = create_interaction_features(train_df, quantile_thresholds=None)
    val_df, _ = create_interaction_features(val_df, quantile_thresholds=quantile_thresholds)
    test_df, _ = create_interaction_features(test_df, quantile_thresholds=quantile_thresholds)
    logger.info(f"训练集阈值: {quantile_thresholds}")
    
    # 3. 编码分类特征
    logger.info("\n步骤3: 编码分类特征...")
    encoders: Dict[str, LabelEncoder] = {}
    train_df = encode_categorical(train_df, encoders, fit=True)
    val_df = encode_categorical(val_df, encoders, fit=False)
    test_df = encode_categorical(test_df, encoders, fit=False)
    
    # 4. 准备特征
    logger.info("\n步骤4: 准备特征矩阵...")
    X_train, y_train, output_train = prepare_features(train_df)
    X_val, y_val, output_val = prepare_features(val_df)
    X_test, y_test, output_test = prepare_features(test_df)
    
    logger.info(f"特征数: {X_train.shape[1]}, 训练样本: {X_train.shape[0]:,}")
    
    # 5. 计算样本权重（小文库分层加权，降低小文库MAPE）
    logger.info("\n步骤5: 计算样本权重（分层策略）...")
    sample_weights_train = compute_sample_weights(
        train_df, 
        strategy="tiered"   # 分层权重：0-3G(5x), 3-5G(6x), 5-7G(4x), 7-10G(1.5x), 10G+(1x)
    )
    
    # 6. 训练回归模型（预测下单量）
    logger.info("\n步骤6: 训练回归模型（预测下单量，应用样本权重）...")
    model = train_regression_model(X_train, y_train, X_val, y_val, sample_weights_train)
    models = {"regression": model}
    
    # 7. 评估模型
    logger.info("\n步骤7: 评估模型...")
    all_metrics = {}
    all_metrics["regression"] = {
        "train": evaluate_model(model, X_train, y_train, output_train, "训练"),
        "val": evaluate_model(model, X_val, y_val, output_val, "验证"),
        "test": evaluate_model(model, X_test, y_test, output_test, "测试"),
    }
    
    # 8. 保存
    logger.info("\n步骤8: 保存模型和配置...")
    save_artifacts(models, encoders, all_metrics, list(X_train.columns), quantile_thresholds)
    
    # 9. 输出最终建议
    logger.info("\n" + "=" * 80)
    logger.info("训练完成！")
    logger.info("=" * 80)
    logger.info("模型类型: XGBoost回归（预测下单量）")
    logger.info("")
    logger.info("训练逻辑：")
    logger.info("  - wkcontractdata 为真实合同量（训练/推理一致）")
    logger.info("  - 模型学习：合同量 + 质量特征 -> 下单量")
    logger.info("  - 分位数阈值从训练集计算并保存到 quantile_thresholds.json")
    logger.info("")
    logger.info("使用方法：")
    logger.info("  1. 加载模型: model = pickle.load('model_regression.pkl')")
    logger.info("  2. 加载阈值: thresholds = json.load('quantile_thresholds.json')")
    logger.info("  3. 生成特征: df, _ = create_interaction_features(df, thresholds)")
    logger.info("  4. 预测下单量: pred_order = model.predict(X_new)")
    logger.info("")
    logger.info(f"模型文件位置: {OUTPUT_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    logger.add(
        "logs/train_output_rate_v6.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO",
    )
    main()

