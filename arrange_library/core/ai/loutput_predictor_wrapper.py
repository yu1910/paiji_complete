"""
loutput产出预测模型封装类 - 用于排机系统集成
创建时间：2025-12-23 10:00:00
更新时间：2025-12-23 14:15:00

功能：
1. 封装训练好的loutput预测模型
2. 将EnhancedLibraryInfo对象转换为模型输入格式
3. 计算Lane级别特征
4. 预测Lane总产出和统计指标

字段映射（EnhancedLibraryInfo -> 训练特征）：
- qpcr_molar -> wkqpcr
- qubit_concentration -> wkqubit
- contract_data_raw -> wkcontractdata / wkcontractdata_1
- peak_size -> wkpeaksize
- sj_number -> wksjnumber
- adaptor_rate -> wkadaptorrate
- complex_result -> wkcomplexresult
- data_unit -> wkdataunit
- risk_build_flag -> wkriskbuildflag
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# from liblane_paths import setup_liblane_paths
# setup_liblane_paths()

from models.library_info import EnhancedLibraryInfo


class LOutputPredictorWrapper:
    """
    loutput预测模型封装类
    
    用于在排机系统中预测Lane的产出，支持：
    1. 单个Lane预测
    2. 批量Lane预测
    3. 产出统计（总量、均值、稳定性）
    """
    
    # EnhancedLibraryInfo属性到训练特征的映射
    FEATURE_MAPPING = {
        # 数值特征映射
        'peak_size': 'wkpeaksize',
        'qpcr_molar': 'wkqpcr',  # 优先使用qpcr_molar
        'qpcr_concentration': 'wkqpcr',  # 备选
        'contract_data_raw': 'wkcontractdata',
        'total_contract_data': 'wktotalcontractdata',
        'qubit_concentration': 'wkqubit',
        'xpd': 'wkxpd',
        # 分类特征映射
        'eq_type': 'wkeqtype',
        'seq_scheme': 'wkseqscheme',
        'sample_type_code': 'wksampletype',
        'peak_map': 'wkpeakmap',
        'product_line': 'wkproductline',
    }
    
    # 数值特征默认值（用于缺失字段）
    NUMERIC_DEFAULTS = {
        'wkpeaksize': 350,
        'wkqpcr': 0.0,
        'wkcontractdata': 0.0,
        'wkcontractdata_1': 0.0,
        'wktotalcontractdata': 0.0,
        'wksjnumber': 0,
        'wkqubit': 0.0,
        'wkxpd': 0.0,
        'wkadaptorrate': 0.0,
    }
    
    # 分类特征默认值
    CATEGORICAL_DEFAULTS = {
        'wkeqtype': 'Unknown',
        'wkseqscheme': 'Unknown',
        'wkcomplexresult': 'Unknown',
        'wksampletype': 'Unknown',
        'wkpeakmap': 'Unknown',
        'wkdataunit': 'Unknown',
        'wkriskbuildflag': 'Unknown',
        'wkproductline': 'Unknown',
    }
    
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        encoders_path: Optional[str] = None,
        model_version: str = "balanced"
    ):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径，None则使用默认路径
            encoders_path: 编码器文件路径，None则使用默认路径
            model_version: 模型版本 (balanced/gentle)
        """
        self.model = None
        self.encoders = {}
        self.model_version = model_version
        self._is_loaded = False
        self.feature_config: Dict[str, Any] = {}
        self.feature_cols: Optional[List[str]] = None
        self.contract_feature_name: str = "wkcontractdata"
        
        # 设置默认路径
        if model_path is None:
            if model_version == "gentle":
                model_path = "models/loutput_gentle_small_value/lightgbm_model_gentle.pkl"
            else:
                model_path = "models/loutput_balanced_mape/lightgbm_model_balanced.pkl"
        
        if encoders_path is None:
            if model_version == "gentle":
                encoders_path = "models/loutput_gentle_small_value/label_encoders.pkl"
            else:
                encoders_path = "models/loutput_balanced_mape/label_encoders.pkl"
        
        self.model_path = Path(model_path)
        self.encoders_path = Path(encoders_path)

        # 读取特征配置（用于适配不同模型版本）
        self.feature_config = self._load_feature_config()
        self.feature_cols = self._resolve_feature_cols(self.feature_config)
        self.contract_feature_name = self._resolve_contract_feature_name(
            self.feature_config, self.feature_cols
        )
        
        # 尝试加载模型
        self._load_model()
    
    def _load_model(self) -> bool:
        """
        加载模型和编码器
        
        Returns:
            bool: 加载是否成功
        """
        try:
            # 加载模型
            if not self.model_path.exists():
                logger.warning(f"模型文件不存在: {self.model_path}")
                return False
            
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"成功加载loutput预测模型: {self.model_path}")
            
            # 加载编码器
            if self.encoders_path.exists():
                with open(self.encoders_path, 'rb') as f:
                    self.encoders = pickle.load(f)
                logger.info(f"成功加载标签编码器: {self.encoders_path}")
            else:
                logger.warning(f"编码器文件不存在: {self.encoders_path}，将使用默认值")
            
            self._is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"加载loutput模型失败: {e}")
            self._is_loaded = False
            return False
    
    @property
    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self._is_loaded and self.model is not None
    
    def predict_lane_output(
        self, 
        libraries: List[EnhancedLibraryInfo],
        lane_id: str = "temp_lane",
        contract_overrides: Optional[Dict[str, float]] = None,
        total_contract_overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        预测Lane的总产出和统计
        
        Args:
            libraries: Lane内的文库列表
            lane_id: Lane标识
            contract_overrides: 以origrec为key的合同量覆盖值（用于两阶段推理）
            total_contract_overrides: 覆盖wktotalcontractdata的值
            
        Returns:
            Dict包含：
            - total_predicted_output: Lane总预测产出(G)
            - avg_predicted_output: 平均每个库产出(G)
            - predicted_rate: 产出率（预测loutput / contractdata）
            - output_cv: 产出变异系数（衡量均衡性）
            - individual_predictions: 每个库的预测值列表
            - prediction_success: 预测是否成功
        """
        # 默认返回值（预测失败时使用）
        default_result = {
            'total_predicted_output': 0.0,
            'avg_predicted_output': 0.0,
            'predicted_rate': 1.0,
            'output_cv': 0.0,
            'individual_predictions': [],
            'prediction_success': False,
            'error_message': None
        }
        
        # 检查模型是否可用
        if not self.is_available:
            default_result['error_message'] = "模型未加载"
            logger.warning("loutput模型未加载，无法预测")
            return default_result
        
        # 检查输入
        if not libraries:
            default_result['error_message'] = "无文库数据"
            return default_result
        
        try:
            # 1. 将文库对象转换为DataFrame
            df = self._libraries_to_dataframe(
                libraries,
                lane_id,
                contract_overrides=contract_overrides,
                total_contract_overrides=total_contract_overrides,
            )
            
            # 2. 创建Lane级别特征
            df = self._create_lane_features(df)
            
            # 3. 编码分类特征
            df = self._encode_categorical_features(df)
            
            # 4. 准备特征矩阵
            X = self._prepare_feature_matrix(df)
            
            # 5. 预测（log空间）
            y_pred_log = self.model.predict(X)
            
            # 6. 转换回原空间
            y_pred = np.expm1(y_pred_log)
            y_pred = np.maximum(y_pred, 0)  # 确保非负
            
            # 7. 计算统计
            total_output = float(y_pred.sum())
            avg_output = float(y_pred.mean())
            
            # 计算总合同数据量（支持覆盖值）
            if contract_overrides:
                total_contract = sum(
                    float(contract_overrides.get(getattr(lib, 'origrec', '') or '', lib.contract_data_raw or 0))
                    for lib in libraries
                )
            else:
                total_contract = sum(
                    float(lib.contract_data_raw or 0) for lib in libraries
                )
            
            # 产出率
            predicted_rate = total_output / total_contract if total_contract > 0 else 1.0
            
            # 产出变异系数
            output_cv = float(np.std(y_pred) / np.mean(y_pred)) if np.mean(y_pred) > 0 else 0.0
            
            return {
                'total_predicted_output': total_output,
                'avg_predicted_output': avg_output,
                'predicted_rate': float(predicted_rate),
                'output_cv': output_cv,
                'individual_predictions': y_pred.tolist(),
                'prediction_success': True,
                'error_message': None
            }
            
        except Exception as e:
            logger.exception(f"loutput预测失败: {e}")
            default_result['error_message'] = str(e)
            return default_result
    
    def _libraries_to_dataframe(
        self, 
        libraries: List[EnhancedLibraryInfo], 
        lane_id: str,
        contract_overrides: Optional[Dict[str, float]] = None,
        total_contract_overrides: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        将EnhancedLibraryInfo列表转换为DataFrame
        
        处理属性名到训练特征名的映射
        """
        data = []
        
        for lib in libraries:
            lib_id = getattr(lib, 'origrec', '') or ''
            row = {
                'llaneid': lane_id,
                'wkorigrec': lib_id,
            }
            
            # 数值特征
            # wkpeaksize
            row['wkpeaksize'] = self._get_numeric_value(
                lib, ['peak_size'], self.NUMERIC_DEFAULTS['wkpeaksize']
            )
            
            # wkqpcr (优先qpcr_molar，其次qpcr_concentration)
            row['wkqpcr'] = self._get_numeric_value(
                lib, ['qpcr_molar', 'qpcr_concentration'], 
                self.NUMERIC_DEFAULTS['wkqpcr']
            )
            
            # wkcontractdata / wkcontractdata_1（支持两阶段推理的覆盖值）
            override_value = None
            if contract_overrides and lib_id in contract_overrides:
                override_value = contract_overrides[lib_id]
            contract_value = (
                float(override_value)
                if override_value is not None
                else self._get_numeric_value(
                    lib, ['contract_data_raw'],
                    self.NUMERIC_DEFAULTS['wkcontractdata']
                )
            )
            row['wkcontractdata'] = contract_value
            row['wkcontractdata_1'] = contract_value
            
            # wktotalcontractdata
            total_override_value = None
            if total_contract_overrides and lib_id in total_contract_overrides:
                total_override_value = total_contract_overrides[lib_id]
            if total_override_value is not None:
                row['wktotalcontractdata'] = float(total_override_value)
            else:
                row['wktotalcontractdata'] = self._get_numeric_value(
                    lib, ['total_contract_data', 'order_contract_data_raw'],
                    row['wkcontractdata']  # 默认使用contractdata
                )
            
            # wkqubit
            row['wkqubit'] = self._get_numeric_value(
                lib, ['qubit_concentration'], 
                self.NUMERIC_DEFAULTS['wkqubit']
            )
            
            # wkxpd
            row['wkxpd'] = self._get_numeric_value(
                lib, ['xpd'], 
                self.NUMERIC_DEFAULTS['wkxpd']
            )
            
            # wksjnumber (上机次数)
            row['wksjnumber'] = self._get_numeric_value(
                lib, ['sj_number'], 
                self.NUMERIC_DEFAULTS['wksjnumber']
            )
            
            # wkadaptorrate (接头比值)
            row['wkadaptorrate'] = self._get_numeric_value(
                lib, ['adaptor_rate', 'jtb'], 
                self.NUMERIC_DEFAULTS['wkadaptorrate']
            )
            
            # 分类特征
            # wkeqtype
            row['wkeqtype'] = self._get_string_value(
                lib, ['eq_type'], self.CATEGORICAL_DEFAULTS['wkeqtype']
            )
            
            # wkseqscheme
            row['wkseqscheme'] = self._get_string_value(
                lib, ['seq_scheme', 'test_no'], 
                self.CATEGORICAL_DEFAULTS['wkseqscheme']
            )
            
            # wkcomplexresult (库检综合结果)
            row['wkcomplexresult'] = self._get_string_value(
                lib, ['complex_result'], 
                self.CATEGORICAL_DEFAULTS['wkcomplexresult']
            )
            
            # wksampletype
            row['wksampletype'] = self._get_string_value(
                lib, ['sample_type_code', 'species'], 
                self.CATEGORICAL_DEFAULTS['wksampletype']
            )
            
            # wkpeakmap
            row['wkpeakmap'] = self._get_string_value(
                lib, ['peak_map'], self.CATEGORICAL_DEFAULTS['wkpeakmap']
            )
            
            # wkdataunit (数据量单位)
            row['wkdataunit'] = self._get_string_value(
                lib, ['data_unit'], 
                self.CATEGORICAL_DEFAULTS['wkdataunit']
            )
            
            # wkriskbuildflag (风险建库标识)
            row['wkriskbuildflag'] = self._get_string_value(
                lib, ['risk_build_flag'], 
                self.CATEGORICAL_DEFAULTS['wkriskbuildflag']
            )
            
            # wkproductline
            row['wkproductline'] = self._get_string_value(
                lib, ['product_line', 'business_line'], 
                self.CATEGORICAL_DEFAULTS['wkproductline']
            )
            
            # wk_10bp_data (10bp数据量)
            row['wk_10bp_data'] = self._get_numeric_value(
                lib, ['_10bp_data', 'ten_bp_data'], 0.0
            )
            
            # wk_single_index_data (单端Index数据量)
            row['wk_single_index_data'] = self._get_numeric_value(
                lib, ['single_index_data'], 0.0
            )
            
            # 额外的分类特征
            # wk_jjbj (碱基不均衡)
            row['wk_jjbj'] = self._get_string_value(
                lib, ['jjbj', 'base_imbalance'], '否'
            )
            
            # wkdatatype
            row['wkdatatype'] = self._get_string_value(
                lib, ['data_type'], '其他'
            )
            
            # wkaddtestsremark
            row['wkaddtestsremark'] = self._get_string_value(
                lib, ['add_tests_remark'], '非加测'
            )
            
            # wkisaddbalance
            row['wkisaddbalance'] = self._get_string_value(
                lib, ['is_add_balance'], '否'
            )
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _get_numeric_value(
        self, 
        obj: Any, 
        attr_names: List[str], 
        default: float
    ) -> float:
        """从对象获取数值属性，支持多个候选属性名"""
        for attr_name in attr_names:
            value = getattr(obj, attr_name, None)
            if value is not None:
                try:
                    return float(value)
                except (ValueError, TypeError):
                    continue
        return default
    
    def _get_string_value(
        self, 
        obj: Any, 
        attr_names: List[str], 
        default: str
    ) -> str:
        """从对象获取字符串属性，支持多个候选属性名"""
        for attr_name in attr_names:
            value = getattr(obj, attr_name, None)
            if value is not None and str(value).strip():
                return str(value)
        return default
    
    def _create_lane_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建Lane级别特征
        
        与训练时的create_lane_features保持一致
        """
        contract_col = self.contract_feature_name if self.contract_feature_name in df.columns else "wkcontractdata"
        if contract_col not in df.columns:
            contract_col = "wkcontractdata_1" if "wkcontractdata_1" in df.columns else "wkcontractdata"

        # 确保数值类型
        for col in ['wkqpcr', 'wkqubit', contract_col]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 计算Lane统计
        lane_lib_count = len(df)
        
        lane_stats = {
            'lane_lib_count': lane_lib_count,
            'lane_avg_qpcr': df['wkqpcr'].mean(),
            'lane_std_qpcr': df['wkqpcr'].std() if lane_lib_count > 1 else 0,
            'lane_min_qpcr': df['wkqpcr'].min(),
            'lane_max_qpcr': df['wkqpcr'].max(),
            'lane_avg_qubit': df['wkqubit'].mean(),
            'lane_std_qubit': df['wkqubit'].std() if lane_lib_count > 1 else 0,
            'lane_total_contractdata': df[contract_col].sum(),
            'lane_avg_contractdata': df[contract_col].mean(),
            'lane_sample_diversity': df['wksampletype'].nunique() if 'wksampletype' in df.columns else 1,
        }
        
        # 添加Lane特征到每行
        for col, val in lane_stats.items():
            df[col] = val if not pd.isna(val) else 0
        
        # 计算相对特征
        # 文库在Lane中的qpcr百分位
        if lane_lib_count > 1:
            df['lib_qpcr_percentile'] = df['wkqpcr'].rank(pct=True)
        else:
            df['lib_qpcr_percentile'] = 0.5
        
        # 文库占Lane总contractdata的比例
        total_contract = lane_stats['lane_total_contractdata']
        df['lib_contractdata_ratio'] = df[contract_col] / (total_contract + 1e-6)
        
        # 文库qpcr与Lane平均值的比值
        avg_qpcr = lane_stats['lane_avg_qpcr']
        df['lib_qpcr_vs_lane_avg'] = df['wkqpcr'] / (avg_qpcr + 1e-6)
        
        # 衍生特征（与训练时一致）
        df['qpcr_log'] = np.log1p(df['wkqpcr'])
        df['contractdata_log'] = np.log1p(df[contract_col])
        
        # is_very_low_qpcr: 使用全局分位数（如果数据量足够）或固定阈值
        qpcr_20th = df['wkqpcr'].quantile(0.2) if len(df) > 5 else 5.0
        df['is_very_low_qpcr'] = (df['wkqpcr'] < qpcr_20th).astype(int)
        
        # 10bp相关特征
        if 'wk_10bp_data' in df.columns:
            df['_10bp_data_log'] = np.log1p(df['wk_10bp_data'])
            df['_10bp_ratio'] = df['wk_10bp_data'] / (df[contract_col] + 1e-6)
            
            # Lane级别的10bp特征
            df['lane_avg_10bp_data'] = df['wk_10bp_data'].mean()
            df['lane_total_10bp_data'] = df['wk_10bp_data'].sum()
        else:
            df['_10bp_data_log'] = 0.0
            df['_10bp_ratio'] = 0.0
            df['lane_avg_10bp_data'] = 0.0
            df['lane_total_10bp_data'] = 0.0
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码分类特征"""
        categorical_cols = [
            'wkeqtype', 'wkseqscheme', 'wkcomplexresult', 'wksampletype',
            'wkpeakmap', 'wkdataunit', 'wkriskbuildflag', 'wkproductline',
            'wk_jjbj', 'wkdatatype', 'wkaddtestsremark', 'wkisaddbalance'
        ]
        
        for col in categorical_cols:
            if col not in df.columns:
                # 设置默认值
                if col == 'wk_jjbj':
                    df[col] = '否'
                elif col == 'wkdatatype':
                    df[col] = '其他'
                elif col == 'wkaddtestsremark':
                    df[col] = '非加测'
                elif col == 'wkisaddbalance':
                    df[col] = '否'
                else:
                    df[col] = self.CATEGORICAL_DEFAULTS.get(col, 'Unknown')
            
            # 确保是字符串
            df[col] = df[col].astype(str)
            
            # 编码
            encoded_col = f'{col}_encoded'
            if col in self.encoders:
                le = self.encoders[col]
                # 处理未知类别
                df[encoded_col] = df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            else:
                # 没有编码器，使用-1
                df[encoded_col] = -1
        
        return df
    
    def _prepare_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """准备特征矩阵（与feature_config一致）"""
        # 基础数值特征（8个）
        numeric_features = [
            'wkqpcr', 'wkcontractdata', 'wktotalcontractdata', 'wksjnumber',
            'wkqubit', 'wkxpd', 'wkadaptorrate', 'wkpeaksize',
        ]
        
        # 扩展数值特征（7个）
        extended_numeric = [
            'wk_10bp_data', 'wk_single_index_data', 'qpcr_log', 'contractdata_log',
            'is_very_low_qpcr', '_10bp_data_log', '_10bp_ratio',
        ]
        
        # Lane级别聚合特征（15个）
        lane_features = [
            'lane_lib_count', 'lane_avg_qpcr', 'lane_std_qpcr', 'lane_min_qpcr', 
            'lane_max_qpcr', 'lane_avg_qubit', 'lane_std_qubit', 
            'lane_total_contractdata', 'lane_avg_contractdata', 'lane_sample_diversity',
            'lib_qpcr_percentile', 'lib_contractdata_ratio', 'lib_qpcr_vs_lane_avg',
            'lane_avg_10bp_data', 'lane_total_10bp_data',
        ]
        
        # 分类特征编码（12个）
        categorical_encoded = [
            'wkeqtype_encoded', 'wkseqscheme_encoded', 'wkcomplexresult_encoded', 
            'wksampletype_encoded', 'wkpeakmap_encoded', 'wkdataunit_encoded', 
            'wkriskbuildflag_encoded', 'wkproductline_encoded',
            'wk_jjbj_encoded', 'wkdatatype_encoded', 'wkaddtestsremark_encoded', 
            'wkisaddbalance_encoded',
        ]
        
        fallback_features = numeric_features + extended_numeric + lane_features + categorical_encoded
        feature_cols = self.feature_cols or fallback_features

        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            logger.warning(
                f"特征缺失: 期望{len(feature_cols)}个，缺失{len(missing)}个: {missing}"
            )

        X = df.reindex(columns=feature_cols, fill_value=0).fillna(0)
        return X.values

    def _load_feature_config(self) -> Dict[str, Any]:
        """加载模型特征配置"""
        config_path = self.model_path.parent / "feature_config.json"
        if not config_path.exists():
            return {}
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as exc:
            logger.warning(f"加载特征配置失败: {exc}")
            return {}

    def _resolve_feature_cols(self, config: Dict[str, Any]) -> Optional[List[str]]:
        """解析特征列顺序，优先使用feature_cols"""
        if not config:
            return None
        feature_cols = config.get("feature_cols")
        if isinstance(feature_cols, list) and feature_cols:
            return feature_cols

        numeric_features = config.get("numeric_features") or []
        lane_features = config.get("lane_agg_features") or []
        categorical_features = config.get("categorical_features") or []
        if not numeric_features and not lane_features and not categorical_features:
            return None

        categorical_encoded = [f"{name}_encoded" for name in categorical_features]
        return list(numeric_features) + list(lane_features) + categorical_encoded

    def _resolve_contract_feature_name(
        self,
        config: Dict[str, Any],
        feature_cols: Optional[List[str]],
    ) -> str:
        """根据特征配置判断合同量字段名"""
        if feature_cols and "wkcontractdata_1" in feature_cols:
            return "wkcontractdata_1"
        if feature_cols and "wkcontractdata" in feature_cols:
            return "wkcontractdata"

        numeric_features = config.get("numeric_features") or []
        if "wkcontractdata_1" in numeric_features:
            return "wkcontractdata_1"
        if "wkcontractdata" in numeric_features:
            return "wkcontractdata"

        explicit = config.get("contract_feature_name") or config.get("contract_base")
        if explicit in ("wkcontractdata_1", "wkcontractdata"):
            return explicit
        return "wkcontractdata"


def create_loutput_predictor(
    model_version: str = "balanced"
) -> Optional[LOutputPredictorWrapper]:
    """
    创建loutput预测器的工厂函数
    
    Args:
        model_version: 模型版本
            - "balanced": 平衡版（推荐，整体性能最佳）
            - "gentle": 温和版（小值段优化）
    
    Returns:
        LOutputPredictorWrapper实例，加载失败返回None
    """
    try:
        predictor = LOutputPredictorWrapper(model_version=model_version)
        if predictor.is_available:
            return predictor
        else:
            logger.warning(f"创建loutput预测器失败: 模型版本={model_version}")
            return None
    except Exception as e:
        logger.error(f"创建loutput预测器异常: {e}")
        return None


# 单例模式（用于全局访问）
_global_predictor: Optional[LOutputPredictorWrapper] = None


def get_global_loutput_predictor(
    model_version: str = "balanced"
) -> Optional[LOutputPredictorWrapper]:
    """
    获取全局loutput预测器（单例）
    
    首次调用时创建，后续调用返回同一实例
    """
    global _global_predictor
    
    if _global_predictor is None:
        _global_predictor = create_loutput_predictor(model_version)
    
    return _global_predictor
