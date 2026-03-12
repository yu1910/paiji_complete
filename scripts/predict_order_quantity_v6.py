"""
端到端预测脚本：从原始数据到预测结果
创建时间：2026-02-05 15:00:00
更新时间：2026-03-03 16:26:23

功能：整合数据处理和预测，自动对齐特征

修复说明 (2026-02-09):
  适配 create_interaction_features 新接口（返回 Tuple[DataFrame, Dict]），
  分段加载训练集分位数阈值，确保推理时交互特征与训练一致。
"""

import json
from pathlib import Path
import pickle
import argparse
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from loguru import logger

# 导入训练脚本中的函数
sys.path.insert(0, str(Path(__file__).parent))
from train_output_rate_v6 import create_interaction_features, encode_categorical

# ==================== 文库类型产出分组（基于历史数据统计） ====================
# 高产出文库类型（产出率 >= 1.15）
HIGH_OUTPUT_TYPES = [
    'VIP-人重测序文库', 'ATAC-seq文库', 'CUT Tag文库（细胞）', 'miniATAC-seq文库', 
    '客户-ATAC-seq文库', '客户-DNA PCR free（动植物）', 'CUT Tag文库（动物组织）', 
    '客户-CUT-Tag文库', '低起始量RNA甲基化文库-IP(人)', '客户-Ribo-seq 文库', 
    '10X转录组-5\'膜蛋白文库', '客户-DNA PCR free（人）', '10X转录组-3\'膜蛋白文库', 
    '客户-全基因组文库', 'Meta文库-非常规', '单细胞文库', 'Agilent全外文库', 
    '客户-10X ATAC文库', '低起始量RNA甲基化文库-Input(人)', '低起始量RNA甲基化文库-Input(鼠)', 
    '客户-10X VDJ文库', '客户-PCR产物/CRISPR', '动植物全基因组重测序文库-非常规', 
    '客户-small RNA文库', '遗传全外文库', '客户-扩增子文库', 'RIP-seq文库', 
    '10X转录组V(D)J-BCR文库', '客户-Chip-seq文库', 'IDT全外文库'
]

# 低产出文库类型（产出率 < 0.95）
LOW_OUTPUT_TYPES = [
    '10X Visium空间转录组文库', 'PCR-free(老师引物)', '客户-寻因单细胞文库', 
    '新筛Panel文库', '客户-10X 3 单细胞转录组文库', '10X Visium FFPEV2空间转录组文库(V2)', 
    '客户-10X全基因组文库', 'RRBS文库', '客户-墨卓单细胞文库', 'LncRNA文库', 
    '10X转录组文库-3V4文库', '肿瘤遗传易感基因筛查文库', '10x HD Visium空间转录组文库(新)', 
    '客户文库-10X Visium 文库', 'UMI mRNA-seq 文库', 'R LOOP CUT&Tag文库', 
    '墨卓转录组-5端文库',
]


def add_sampletype_output_group(df: pd.DataFrame) -> pd.DataFrame:
    """添加文库类型产出分组特征"""
    df = df.copy()
    
    def classify_sampletype(sampletype: str) -> str:
        sampletype = str(sampletype)
        if sampletype in HIGH_OUTPUT_TYPES:
            return 'high'
        elif sampletype in LOW_OUTPUT_TYPES:
            return 'low'
        else:
            return 'medium'
    
    df['sampletype_output_group'] = df['wksampletype'].apply(classify_sampletype)
    return df


def add_lane_unique_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建 lane_unique_id

    支持两种字段命名：
    - 新格式：runid + lane_id
    - 旧格式：lrunid + llaneorder（向后兼容）
    """
    df = df.copy()

    if "runid" in df.columns and "lane_id" in df.columns:
        df["runid"] = df["runid"].astype(str).fillna("")
        df["lane_id"] = df["lane_id"].astype(str).fillna("")
        df["lane_unique_id"] = df["runid"] + "_" + df["lane_id"]
    elif "lrunid" in df.columns and "llaneorder" in df.columns:
        df["lrunid"] = df["lrunid"].astype(str).fillna("")
        df["llaneorder"] = df["llaneorder"].astype(str).fillna("")
        df["lane_unique_id"] = df["lrunid"] + "_" + df["llaneorder"]
    else:
        logger.warning("未找到Lane标识字段(runid+lane_id 或 lrunid+llaneorder)，"
                       "使用行号作为lane_unique_id")
        df["lane_unique_id"] = [f"row_{i}" for i in range(len(df))]

    return df


def remove_leakage_features(df: pd.DataFrame) -> pd.DataFrame:
    """移除数据泄露特征"""
    leakage_features = [
        'lane_effective_rate', 'lane_ht_cv', 'lane_enough_rate',
        'lane_q30', 'lane_lht_mean', 'lane_lht_std', 
        'lane_wk_number', 'lane_cf_number',
    ]
    removed = [col for col in leakage_features if col in df.columns]
    if removed:
        df = df.drop(columns=removed)
        logger.info(f"已移除泄漏特征: {len(removed)}个")
    return df


def compute_lane_stats(df: pd.DataFrame) -> pd.DataFrame:
    """计算Lane统计特征"""
    df = df.copy()
    df = remove_leakage_features(df)
    
    # 预处理
    df['wk_jjbj'] = df['wk_jjbj'].astype(str).fillna('否')
    df['wkindexseq'] = df['wkindexseq'].astype(str).fillna('')
    df['wk_10bp_data'] = pd.to_numeric(df['wk_10bp_data'], errors='coerce').fillna(0)
    df['wkaddtestsremark'] = df['wkaddtestsremark'].astype(str).fillna('非加测')
    df['wksampletype'] = df['wksampletype'].astype(str).fillna('未知')
    df['wkcontractdata'] = pd.to_numeric(df['wkcontractdata'], errors='coerce').fillna(0)
    
    lane_stats_dict = {}
    
    for lane_id, group in df.groupby('lane_unique_id'):
        stats = {}
        total_libs = len(group)
        
        stats['lane_jjbj_ratio'] = (group['wk_jjbj'] == '是').sum() / total_libs
        stats['lane_single_index_ratio'] = (~group['wkindexseq'].str.contains(';', na=False)).sum() / total_libs
        
        total_contract = group['wkcontractdata'].sum()
        total_10bp = group['wk_10bp_data'].sum()
        stats['lane_10bp_ratio'] = total_10bp / total_contract if total_contract > 0 else 0
        
        stats['lane_addtest_ratio'] = (group['wkaddtestsremark'] != '非加测').sum() / total_libs
        
        sampletype_counts = group['wksampletype'].value_counts()
        for stype, count in sampletype_counts.items():
            stats[f'lane_sampletype_{stype}_ratio'] = count / total_libs
        
        sampletype_probs = sampletype_counts / total_libs
        entropy = -1 * (sampletype_probs * np.log(sampletype_probs + 1e-10)).sum()
        stats['lane_sampletype_diversity'] = entropy
        
        # 注意：v6.1版本移除了lane_contract_mean和lane_contract_std（泛化性差）
        stats['lane_contract_median'] = group['wkcontractdata'].median()
        stats['lane_library_count'] = total_libs
        
        lane_stats_dict[lane_id] = stats
    
    lane_stats_df = pd.DataFrame.from_dict(lane_stats_dict, orient='index')
    lane_stats_df.index.name = 'lane_unique_id'
    lane_stats_df = lane_stats_df.reset_index()
    
    # 删除旧的lane_sampletype特征
    existing_lane_cols = [col for col in df.columns if str(col).startswith('lane_sampletype_')]
    if existing_lane_cols:
        df = df.drop(columns=existing_lane_cols)
    
    df = df.merge(lane_stats_df, on='lane_unique_id', how='left')
    
    # 填充缺失值
    lane_feature_cols = [col for col in df.columns if str(col).startswith('lane_') and col != 'lane_unique_id']
    for col in lane_feature_cols:
        df[col] = df[col].fillna(0)
    
    logger.info(f"✓ Lane统计特征: {len(lane_feature_cols)}个")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """清洗数据"""
    df = df.copy()
    
    # 数值特征
    numeric_cols = ['wkqpcr', 'wkcontractdata', 'wkpeaksize', 'wkxpd', 'wkadaptorrate']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median() if col != 'wkcontractdata' else 0)
    
    # 分类特征
    categorical_defaults = {
        'wksampletype': '未知',
        'wkjkhj': '诺禾自动',
        'wkcomplexresult': '合格',
        'wkpeakmap': '良好',
        'wk_jjbj': '否'
    }
    for col, default in categorical_defaults.items():
        if col in df.columns:
            df[col] = df[col].astype(str).fillna(default).replace(['', 'nan', 'None'], default)
    
    return df


def align_features(df: pd.DataFrame, model_features: list) -> pd.DataFrame:
    """对齐特征：确保数据特征与模型特征一致"""
    df = df.copy()
    
    # 1. 添加模型需要但数据中没有的特征（填充0）
    for feat in model_features:
        if feat not in df.columns:
            df[feat] = 0
            logger.debug(f"添加缺失特征: {feat} (填充0)")
    
    # 2. 只保留模型需要的特征，按模型顺序排列
    df = df[model_features]
    
    return df


def _load_thresholds(path: Path) -> Optional[Dict[str, float]]:
    """
    加载分位数阈值文件。

    找不到时返回 None，create_interaction_features 会从当前数据计算（旧模型兼容）。
    """
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)
        logger.info(f"  已加载分位数阈值: {path.name} ({len(thresholds)} 个)")
        return thresholds
    logger.warning(f"  分位数阈值文件不存在: {path}，将从当前数据计算")
    return None


def _load_order_guard_config(model_dir: Path) -> Dict[str, float]:
    """加载下单风控参数，缺失时使用默认值。"""
    defaults = {
        "target_shortage_prob": 0.26,
        "target_enough_prob": 0.78,
        "risk_base_scale": 0.45,
        "risk_extra_scale": 1.20,
        "max_uplift": 1.80,
        "min_order_ratio": 0.95,
        "max_order_ratio": 2.20,
        "group_cols": ["wksampletype", "wkjkhj", "wkproductline"],
        "group_target_enough_prob": {"small": {}, "large": {}},
        "group_hard_min_ratio": {"small": {}, "large": {}},
        "runtime_order_multiplier_by_sampletype": {},
        "runtime_order_multiplier_rules": [],
        "lane_realloc": {
            "enabled": True,
            "max_iter": 2,
            "move_ratio": 0.03,
            "receiver_quantile": 0.75,
            "donor_quantile": 0.30,
        },
        "risk_uplift_bonus_by_sampletype": {},
    }
    config_path = model_dir / "config.json"
    if not config_path.exists():
        logger.warning(f"  下单配置不存在: {config_path}，使用默认风控参数")
        return defaults

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        risk_cfg = config_data.get("risk_adjustment", {})
        for key, val in defaults.items():
            if isinstance(val, dict):
                risk_cfg[key] = dict(risk_cfg.get(key, val))
            elif isinstance(val, list):
                raw_list = risk_cfg.get(key, val)
                risk_cfg[key] = list(raw_list) if isinstance(raw_list, list) else list(val)
            else:
                risk_cfg[key] = float(risk_cfg.get(key, val))
        logger.info(f"  已加载风控参数: {config_path.name}")
        return risk_cfg
    except Exception as exc:
        logger.warning(f"  加载风控参数失败({exc})，使用默认值")
        return defaults


def _build_group_keys_for_df(df_segment: pd.DataFrame, group_cols: List[str]) -> np.ndarray:
    """构建分组键。"""
    if len(df_segment) == 0:
        return np.array([], dtype=object)
    parts: List[pd.Series] = []
    for col in group_cols:
        if col in df_segment.columns:
            part = df_segment[col].astype(str).fillna("NA").replace(["", "nan", "None"], "NA")
        else:
            part = pd.Series(["NA"] * len(df_segment), index=df_segment.index, dtype=object)
        parts.append(part)
    key_series = parts[0]
    for part in parts[1:]:
        key_series = key_series + "|" + part
    return key_series.astype(str).values


def _lane_reallocate_orders(
    orders: np.ndarray,
    contracts: np.ndarray,
    shortage_prob: np.ndarray,
    lane_ids: Optional[np.ndarray],
    min_ratio: np.ndarray,
    lane_cfg: Dict[str, float],
) -> np.ndarray:
    """Lane内二次分配：总量不变，向高风险样本倾斜。"""
    if lane_ids is None or len(orders) == 0 or not bool(lane_cfg.get("enabled", True)):
        return orders

    adjusted = orders.copy()
    contracts_safe = np.maximum(contracts, 0.0)
    max_ratio = float(lane_cfg.get("max_order_ratio", 2.20))
    max_iter = int(lane_cfg.get("max_iter", 2))
    move_ratio = float(lane_cfg.get("move_ratio", 0.03))
    receiver_q = float(lane_cfg.get("receiver_quantile", 0.75))
    donor_q = float(lane_cfg.get("donor_quantile", 0.30))

    lane_arr = pd.Series(lane_ids).astype(str).fillna("NA").values
    unique_lanes = np.unique(lane_arr)
    for _ in range(max_iter):
        for lane in unique_lanes:
            lane_idx = np.where(lane_arr == lane)[0]
            if len(lane_idx) < 4:
                continue

            lane_prob = shortage_prob[lane_idx]
            q_high = float(np.quantile(lane_prob, receiver_q))
            q_low = float(np.quantile(lane_prob, donor_q))
            receiver_idx = lane_idx[lane_prob >= q_high]
            donor_idx = lane_idx[lane_prob <= q_low]
            if len(receiver_idx) == 0 or len(donor_idx) == 0:
                continue

            donor_reducible = np.maximum(
                adjusted[donor_idx] - contracts_safe[donor_idx] * min_ratio[donor_idx],
                0.0,
            )
            total_reducible = float(donor_reducible.sum())
            if total_reducible <= 1e-8:
                continue

            lane_total = float(np.sum(adjusted[lane_idx]))
            move_budget = min(total_reducible, lane_total * move_ratio)
            if move_budget <= 1e-8:
                continue

            donor_take = move_budget * (donor_reducible / total_reducible)
            adjusted[donor_idx] -= donor_take

            receiver_cap = np.maximum(contracts_safe[receiver_idx] * max_ratio - adjusted[receiver_idx], 0.0)
            cap_sum = float(receiver_cap.sum())
            if cap_sum <= 1e-8:
                adjusted[donor_idx] += donor_take
                continue

            receiver_add = move_budget * (receiver_cap / cap_sum)
            adjusted[receiver_idx] += receiver_add

    return adjusted


def _apply_runtime_multiplier_rules(
    adjusted: np.ndarray,
    contract_safe: np.ndarray,
    shortage_prob: np.ndarray,
    sampletypes: Optional[np.ndarray],
    min_ratio_vec: np.ndarray,
    max_ratio: float,
    rules: object,
) -> np.ndarray:
    """按规则条件应用运行时倍率。"""
    if sampletypes is None or not isinstance(rules, list) or len(rules) == 0:
        return adjusted

    base_ratio = np.where(contract_safe > 0, adjusted / contract_safe, 0.0)
    rule_multiplier = np.ones(len(adjusted), dtype=float)
    sampletype_arr = np.asarray(sampletypes, dtype=object)

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        multiplier = float(rule.get("multiplier", 1.0))
        if multiplier <= 0:
            continue

        mask = np.ones(len(adjusted), dtype=bool)
        rule_sampletypes = rule.get("sampletypes", [])
        if isinstance(rule_sampletypes, list) and len(rule_sampletypes) > 0:
            mask &= np.isin(sampletype_arr, [str(x) for x in rule_sampletypes])

        if "min_shortage_prob" in rule:
            mask &= shortage_prob >= float(rule["min_shortage_prob"])
        if "max_shortage_prob" in rule:
            mask &= shortage_prob <= float(rule["max_shortage_prob"])
        if "min_base_order_ratio" in rule:
            mask &= base_ratio >= float(rule["min_base_order_ratio"])
        if "max_base_order_ratio" in rule:
            mask &= base_ratio <= float(rule["max_base_order_ratio"])

        rule_multiplier[mask] = np.maximum(rule_multiplier[mask], multiplier)

    adjusted = adjusted * rule_multiplier
    adjusted = np.clip(adjusted, contract_safe * min_ratio_vec, contract_safe * max_ratio)
    return adjusted


def _apply_order_guard(
    base_order: np.ndarray,
    contract: np.ndarray,
    shortage_prob: np.ndarray,
    risk_cfg: Dict[str, float],
    sampletypes: Optional[np.ndarray] = None,
    group_keys: Optional[np.ndarray] = None,
    lane_ids: Optional[np.ndarray] = None,
    segment_name: str = "small",
) -> tuple[np.ndarray, np.ndarray]:
    """按欠产风险概率上调下单量，并做上下界约束。"""
    base = np.maximum(base_order, 0.0)
    contract_safe = np.maximum(contract, 0.0)
    prob = np.clip(shortage_prob, 0.0, 1.0)

    target_prob = risk_cfg["target_shortage_prob"]
    base_scale = risk_cfg["risk_base_scale"]
    extra_scale = risk_cfg["risk_extra_scale"]
    max_uplift = risk_cfg["max_uplift"]

    target_prob_vec = np.full(len(prob), target_prob, dtype=float)
    group_target_map = risk_cfg.get("group_target_enough_prob", {}).get(segment_name, {})
    if group_keys is not None and isinstance(group_target_map, dict) and group_target_map:
        group_target_shortage = np.array(
            [
                max(0.02, 1.0 - float(group_target_map.get(str(g), risk_cfg.get("target_enough_prob", 0.78))))
                for g in group_keys
            ],
            dtype=float,
        )
        target_prob_vec = np.minimum(target_prob_vec, group_target_shortage)

    uplift = 1.0 + base_scale * prob + extra_scale * np.clip(prob - target_prob_vec, 0.0, None)
    if sampletypes is not None:
        bonus_map = risk_cfg.get("risk_uplift_bonus_by_sampletype", {})
        if isinstance(bonus_map, dict) and bonus_map:
            sampletype_bonus = np.array(
                [float(bonus_map.get(str(st), 1.0)) for st in sampletypes],
                dtype=float,
            )
            uplift = uplift * sampletype_bonus
    uplift = np.clip(uplift, 1.0, max_uplift)

    adjusted = base * uplift
    if group_keys is not None:
        hard_ratio_map = risk_cfg.get("group_hard_min_ratio", {}).get(segment_name, {})
        if isinstance(hard_ratio_map, dict) and hard_ratio_map:
            min_ratio_vec = np.array(
                [float(hard_ratio_map.get(str(g), risk_cfg["min_order_ratio"])) for g in group_keys],
                dtype=float,
            )
        else:
            min_ratio_vec = np.full(len(adjusted), risk_cfg["min_order_ratio"], dtype=float)
    else:
        min_ratio_vec = np.full(len(adjusted), risk_cfg["min_order_ratio"], dtype=float)
    min_ratio_vec = np.clip(min_ratio_vec, risk_cfg["min_order_ratio"], risk_cfg["max_order_ratio"])

    adjusted = np.clip(
        adjusted,
        contract_safe * min_ratio_vec,
        contract_safe * risk_cfg["max_order_ratio"],
    )

    runtime_map = risk_cfg.get("runtime_order_multiplier_by_sampletype", {})
    if sampletypes is not None and isinstance(runtime_map, dict) and runtime_map:
        runtime_multiplier = np.array(
            [float(runtime_map.get(str(st), 1.0)) for st in sampletypes],
            dtype=float,
        )
        adjusted = adjusted * runtime_multiplier
        adjusted = np.clip(
            adjusted,
            contract_safe * min_ratio_vec,
            contract_safe * risk_cfg["max_order_ratio"],
        )

    adjusted = _apply_runtime_multiplier_rules(
        adjusted=adjusted,
        contract_safe=contract_safe,
        shortage_prob=prob,
        sampletypes=sampletypes,
        min_ratio_vec=min_ratio_vec,
        max_ratio=float(risk_cfg["max_order_ratio"]),
        rules=risk_cfg.get("runtime_order_multiplier_rules", []),
    )

    adjusted = _lane_reallocate_orders(
        orders=adjusted,
        contracts=contract_safe,
        shortage_prob=prob,
        lane_ids=lane_ids,
        min_ratio=min_ratio_vec,
        lane_cfg={
            **dict(risk_cfg.get("lane_realloc", {})),
            "max_order_ratio": risk_cfg["max_order_ratio"],
        },
    )
    adjusted = np.clip(
        adjusted,
        contract_safe * min_ratio_vec,
        contract_safe * risk_cfg["max_order_ratio"],
    )
    return adjusted, uplift


def _predict_segment_shortage_probability(
    df_segment: pd.DataFrame,
    clf_model: Optional[Any],
    fallback_X: pd.DataFrame,
) -> np.ndarray:
    """预测欠产概率；模型不存在时返回0。"""
    if clf_model is None:
        return np.zeros(len(df_segment), dtype=float)
    try:
        if hasattr(clf_model, "get_booster"):
            clf_features = clf_model.get_booster().feature_names
            X_clf = align_features(df_segment, clf_features)
        else:
            X_clf = fallback_X
        if hasattr(clf_model, "predict_proba"):
            proba = clf_model.predict_proba(X_clf)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return np.asarray(proba[:, 1], dtype=float)
            return np.asarray(proba.ravel(), dtype=float)
        return np.zeros(len(df_segment), dtype=float)
    except Exception as exc:
        logger.warning(f"  欠产概率预测失败，降级为0: {exc}")
        return np.zeros(len(df_segment), dtype=float)


def predict(
    input_file: Path,
    output_file: Path,
    model_dir: Path = Path("/data/work/yuyongpeng/liblane_v2_deepseek/models/segmented_v6_prod_r5")
):
    """
    完整的预测流程
    """
    logger.info("="*70)
    logger.info("开始预测流程")
    logger.info("="*70)
    
    # 步骤1：加载数据
    logger.info(f"\n[1/8] 加载数据")
    df = pd.read_csv(input_file, low_memory=False)
    logger.info(f"  原始数据: {len(df):,} 条")
    
    # 步骤2：创建lane_unique_id
    logger.info(f"\n[2/8] 创建 Lane 标识")
    if 'lane_unique_id' not in df.columns:
        df = add_lane_unique_id(df)
    logger.info(f"  唯一 Lanes: {df['lane_unique_id'].nunique()}")
    
    # 步骤3：清洗数据
    logger.info(f"\n[3/9] 清洗数据")
    df = clean_data(df)
    
    # 步骤4：添加文库类型产出分组
    logger.info(f"\n[4/9] 添加文库类型产出分组")
    df = add_sampletype_output_group(df)
    group_counts = df['sampletype_output_group'].value_counts()
    logger.info(f"  高产出: {group_counts.get('high', 0):,}, 中等: {group_counts.get('medium', 0):,}, 低产出: {group_counts.get('low', 0):,}")
    
    # 步骤5：计算Lane特征
    logger.info(f"\n[5/9] 计算 Lane 统计特征")
    df = compute_lane_stats(df)
    
    # 步骤6：加载模型、编码器、分类器和分位数阈值
    logger.info(f"\n[6/9] 加载模型和分位数阈值")
    with open(model_dir / 'small_library_model.pkl', 'rb') as f:
        small_model = pickle.load(f)
    with open(model_dir / 'large_library_model.pkl', 'rb') as f:
        large_model = pickle.load(f)
    with open(model_dir / 'small_encoders.pkl', 'rb') as f:
        small_encoders = pickle.load(f)
    with open(model_dir / 'large_encoders.pkl', 'rb') as f:
        large_encoders = pickle.load(f)

    small_shortage_clf = None
    large_shortage_clf = None
    small_clf_path = model_dir / "small_shortage_classifier.pkl"
    large_clf_path = model_dir / "large_shortage_classifier.pkl"
    if small_clf_path.exists() and large_clf_path.exists():
        with open(small_clf_path, "rb") as f:
            small_shortage_clf = pickle.load(f)
        with open(large_clf_path, "rb") as f:
            large_shortage_clf = pickle.load(f)
        logger.info("  已加载欠产风险分类器")
    else:
        logger.warning("  未找到欠产风险分类器，使用纯回归下单")

    risk_cfg = _load_order_guard_config(model_dir)

    # 加载分位数阈值（分段模型各自有独立阈值）
    small_thresholds = _load_thresholds(model_dir / "small_quantile_thresholds.json")
    large_thresholds = _load_thresholds(model_dir / "large_quantile_thresholds.json")
    logger.info(f"  模型加载完成")
    
    # 步骤7：分段创建交互特征、编码、预测
    logger.info(f"\n[7/9] 分段创建交互特征并预测")
    
    small_mask = df['wkcontractdata'] < 7.0
    large_mask = df['wkcontractdata'] >= 7.0
    
    # 小文库：用小文库阈值生成交互特征
    if small_mask.sum() > 0:
        logger.info(f"  处理小文库 (<7G): {small_mask.sum():,} 条")
        df_small = df[small_mask].copy()
        sampletypes_small = df_small["wksampletype"].astype(str).values if "wksampletype" in df_small.columns else None
        group_cols = risk_cfg.get("group_cols", ["wksampletype", "wkjkhj", "wkproductline"])
        group_keys_small = _build_group_keys_for_df(df_small, group_cols)
        lane_ids_small = (
            df_small["lane_unique_id"].astype(str).values if "lane_unique_id" in df_small.columns else None
        )
        df_small, _ = create_interaction_features(df_small, quantile_thresholds=small_thresholds)
        df_small = encode_categorical(df_small, small_encoders, fit=False)
        
        # 对齐特征
        model_features = small_model.get_booster().feature_names
        X_small = align_features(df_small, model_features)
        
        # 回归预测 + 欠产风险矫正
        pred_small_base = small_model.predict(X_small)
        shortage_prob_small = _predict_segment_shortage_probability(
            df_small, small_shortage_clf, X_small
        )
        contract_small = pd.to_numeric(df_small["wkcontractdata"], errors="coerce").fillna(0.0).values
        pred_small_adj, uplift_small = _apply_order_guard(
            pred_small_base,
            contract_small,
            shortage_prob_small,
            risk_cfg,
            sampletypes=sampletypes_small,
            group_keys=group_keys_small,
            lane_ids=lane_ids_small,
            segment_name="small",
        )

        df.loc[small_mask, "predicted_order_base"] = pred_small_base
        df.loc[small_mask, "shortage_probability"] = shortage_prob_small
        df.loc[small_mask, "risk_group_key"] = group_keys_small
        df.loc[small_mask, "order_guard_uplift"] = uplift_small
        df.loc[small_mask, "predicted_order"] = pred_small_adj
        logger.info(
            f"    平均基础预测: {pred_small_base.mean():.2f}G, "
            f"平均矫正后: {pred_small_adj.mean():.2f}G"
        )
    
    # 大文库：用大文库阈值生成交互特征
    if large_mask.sum() > 0:
        logger.info(f"  处理大文库 (>=7G): {large_mask.sum():,} 条")
        df_large = df[large_mask].copy()
        sampletypes_large = df_large["wksampletype"].astype(str).values if "wksampletype" in df_large.columns else None
        group_cols = risk_cfg.get("group_cols", ["wksampletype", "wkjkhj", "wkproductline"])
        group_keys_large = _build_group_keys_for_df(df_large, group_cols)
        lane_ids_large = (
            df_large["lane_unique_id"].astype(str).values if "lane_unique_id" in df_large.columns else None
        )
        df_large, _ = create_interaction_features(df_large, quantile_thresholds=large_thresholds)
        df_large = encode_categorical(df_large, large_encoders, fit=False)
        
        # 对齐特征
        model_features = large_model.get_booster().feature_names
        X_large = align_features(df_large, model_features)
        
        # 回归预测 + 欠产风险矫正
        pred_large_base = large_model.predict(X_large)
        shortage_prob_large = _predict_segment_shortage_probability(
            df_large, large_shortage_clf, X_large
        )
        contract_large = pd.to_numeric(df_large["wkcontractdata"], errors="coerce").fillna(0.0).values
        pred_large_adj, uplift_large = _apply_order_guard(
            pred_large_base,
            contract_large,
            shortage_prob_large,
            risk_cfg,
            sampletypes=sampletypes_large,
            group_keys=group_keys_large,
            lane_ids=lane_ids_large,
            segment_name="large",
        )

        df.loc[large_mask, "predicted_order_base"] = pred_large_base
        df.loc[large_mask, "shortage_probability"] = shortage_prob_large
        df.loc[large_mask, "risk_group_key"] = group_keys_large
        df.loc[large_mask, "order_guard_uplift"] = uplift_large
        df.loc[large_mask, "predicted_order"] = pred_large_adj
        logger.info(
            f"    平均基础预测: {pred_large_base.mean():.2f}G, "
            f"平均矫正后: {pred_large_adj.mean():.2f}G"
        )
    
    # 步骤8：保存结果
    logger.info(f"\n[8/9] 保存结果")
    df['model_used'] = df['wkcontractdata'].apply(lambda x: 'small' if x < 7 else 'large')
    
    # 选择输出列
    output_cols = [
        'lane_unique_id',
        'wkcontractdata',
        'predicted_order_base',
        'shortage_probability',
        'risk_group_key',
        'order_guard_uplift',
        'predicted_order',
        'model_used',
    ]
    optional_cols = ['lorderdata', 'loutput', 'wksampletype', 'wkqpcr', 'wkpeaksize']
    for col in optional_cols:
        if col in df.columns:
            output_cols.append(col)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df[output_cols].to_csv(output_file, index=False)
    logger.info(f"  ✓ 已保存至: {output_file}")
    
    # 统计信息
    logger.info(f"\n" + "="*70)
    logger.info("预测完成")
    logger.info("="*70)
    logger.info(f"总样本数: {len(df):,}")
    logger.info(f"  小文库: {small_mask.sum():,} 条")
    logger.info(f"  大文库: {large_mask.sum():,} 条")
    logger.info(f"预测下单量: {df['predicted_order'].min():.2f}G ~ {df['predicted_order'].max():.2f}G")
    logger.info(f"           平均 {df['predicted_order'].mean():.2f}G")
    logger.info("="*70)


def main():
    parser = argparse.ArgumentParser(description='端到端预测脚本')
    parser.add_argument('--input', type=str, required=True, help='输入数据文件')
    parser.add_argument('--output', type=str, required=True, help='输出结果文件')
    parser.add_argument(
        '--model-dir',
        type=str,
        default='/data/work/yuyongpeng/liblane_v2_deepseek/models/segmented_v6_prod_r5',
        help='模型目录',
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_file = Path(args.output)
    model_dir = Path(args.model_dir)
    
    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    if not model_dir.exists():
        logger.error(f"模型目录不存在: {model_dir}")
        return
    
    predict(input_file, output_file, model_dir)


if __name__ == "__main__":
    main()

