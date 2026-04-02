"""
Pooling系数优化器
创建时间：2025-12-30 15:30:00
更新时间：2026-02-09 14:30:00
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger

# from liblane_paths import setup_liblane_paths

# setup_liblane_paths()

from arrange_library.core.ai.loutput_predictor_wrapper import LOutputPredictorWrapper, create_loutput_predictor
from arrange_library.core.config.scheduling_config import get_pooling_config
from arrange_library.core.scheduling.scheduling_types import LaneAssignment
from arrange_library.models.library_info import EnhancedLibraryInfo


class PoolingConfigManager:
    """
    Pooling配置管理器
    用于管理不同工序和文库类型的下单量缩小倍数配置
    """
    
    def __init__(self, config_path: str = "/data/work/yuyongpeng/liblane_v2_deepseek/data/wktype_pooling_config.xlsx"):
        self.config_path = Path(config_path)
        self.config_df = None
        self._load_config()
    
    def _load_config(self) -> None:
        """加载配置表"""
        if not self.config_path.exists():
            logger.warning(f"Pooling配置表不存在: {self.config_path}，将使用默认策略")
            return
        
        try:
            self.config_df = pd.read_excel(self.config_path)
            logger.info(f"成功加载Pooling配置表: {self.config_path}，共{len(self.config_df)}条配置")
        except Exception as e:
            logger.error(f"加载Pooling配置表失败: {e}")
            self.config_df = None
    
    def get_reduction_factor(
        self, 
        testno: str, 
        libtype: str, 
        contract_data: float
    ) -> Optional[float]:
        """
        获取下单量缩小倍数
        
        Args:
            testno: 工序（如'Novaseq X Plus-PE150'）
            libtype: 文库类型（wksampletype）
            contract_data: 合同数据量(GB)
        
        Returns:
            如果匹配配置，返回缩小倍数；否则返回None
        """
        if self.config_df is None or self.config_df.empty:
            return None
        
        # 筛选匹配的配置：工序和文库类型都匹配
        matched = self.config_df[
            (self.config_df['TESTNO'].astype(str).str.strip() == str(testno).strip()) &
            (self.config_df['LIBTYPE'].astype(str).str.strip() == str(libtype).strip())
        ]
        
        if matched.empty:
            return None
        
        # 根据数据量范围查找匹配的配置
        for _, row in matched.iterrows():
            min_data = float(row['MINDATA'])
            max_data = float(row['MAXDATA'])
            
            # 数据量在范围内（包含边界）
            if min_data <= contract_data <= max_data:
                return float(row['POOLING'])
        
        return None


@dataclass
class PoolingOptimizationResult:
    """Pooling优化结果"""

    lane_id: str
    coefficients: Dict[str, float]
    predicted_total_output: float
    predicted_avg_output: float
    predicted_cv: float
    original_total: float
    improvement_pct: float
    optimization_applied: bool
    reason: str
    warnings: List[str] = field(default_factory=list)
    order_cv_ratio: float = 0.0
    under_delivery_gb: float = 0.0
    over_order_gb: float = 0.0
    lane_cv: Optional[float] = None


class PoolingCoefficientOptimizer:
    """
    基于loutput预测模型的Pooling系数优化器

    职责：
    1. 使用预测模型获取文库预期产出
    2. 推导最低不欠交付的Pooling系数，并在约束内收敛
    3. 输出可解释的优化结果，包含欠交付/浪费评估
    """

    def __init__(
        self,
        config_path: str = "arrange_library/config/pooling_optimizer_config.yaml",
        predictor: Optional[LOutputPredictorWrapper] = None,
        pooling_config_path: str = "/data/work/yuyongpeng/liblane_v2_deepseek/data/wktype_pooling_config.xlsx",
        efficiency_correction_path: str = "/data/work/yuyongpeng/liblane_v2_deepseek/data/production_efficiency_correction.csv",
    ):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.pooling_config = get_pooling_config()
        self.enabled: bool = bool(self.config.get("enabled", True))
        self.max_coefficient: float = float(self.config.get("max_coefficient", 2.5))
        self.min_coefficient: float = float(self.config.get("min_coefficient", 0.5))
        self.min_ordered_data: float = float(
            self.config.get("min_ordered_data_gb", self.pooling_config.min_ordered_gb)
        )
        self.order_cv_ratio_threshold: float = float(
            self.config.get(
                "order_cv_ratio_threshold", self.pooling_config.lane_order_ratio_limit
            )
        )
        self.safety_factor: float = float(self.config.get("safety_factor", 1.05))
        self.fallback_on_error: bool = bool(self.config.get("fallback_on_error", True))
        self.target_output_gb: float = float(self.config.get("target_output_gb", 90.0))
        
        # 初始化Pooling配置管理器
        self.pooling_config_manager = PoolingConfigManager(pooling_config_path)
        
        # 加载产出效率修正表（基于历史数据分析的后校准）
        self._efficiency_corrections = self._load_efficiency_corrections(efficiency_correction_path)

        if predictor is not None:
            self.predictor = predictor
        else:
            model_path = self.config.get("model_path")
            encoders_path = self.config.get("encoders_path")
            if model_path or encoders_path:
                self.predictor = LOutputPredictorWrapper(
                    model_path=model_path, encoders_path=encoders_path
                )
            else:
                self.predictor = create_loutput_predictor(model_version="balanced")
        if self.predictor is None or not self.predictor.is_available:
            logger.warning("loutput预测器不可用，将使用规则降级策略")
            self.enabled = False
    
    def _load_efficiency_corrections(self, csv_path: str) -> List[Dict]:
        """
        加载产出效率修正表
        
        修正表基于历史数据分析，对产够率较高(>=75%)的文库类型x合同区间组合
        进行后校准，降低过度预测。每条记录包含:
          - LIBTYPE: 文库类型
          - MINDATA/MAXDATA: 合同数据量区间(GB)
          - CORRECTION_FACTOR: 修正系数(乘到final_order_amount上)
        
        Args:
            csv_path: 修正表CSV路径
        
        Returns:
            修正记录列表，每条记录是一个dict
        """
        path = Path(csv_path)
        if not path.exists():
            logger.info(f"产出效率修正表不存在: {csv_path}，跳过后校准")
            return []
        
        try:
            df = pd.read_csv(path, encoding='utf-8-sig')
            corrections = df.to_dict('records')
            logger.info(f"加载产出效率修正表: {len(corrections)}条记录 ({csv_path})")
            return corrections
        except Exception as e:
            logger.error(f"加载产出效率修正表失败: {e}")
            return []
    
    def _get_efficiency_correction(self, libtype: str, contract_data: float) -> Optional[float]:
        """
        查找产出效率修正系数
        
        对于产够率高且过度预测的文库类型x合同区间组合，返回一个<1的修正系数
        用于在最终下单量计算后乘上去，降低溢出。
        
        Args:
            libtype: 文库类型
            contract_data: 合同数据量(GB)
        
        Returns:
            修正系数，未命中返回None（不修正）
        """
        for record in self._efficiency_corrections:
            if (str(record['LIBTYPE']).strip() == str(libtype).strip()
                    and float(record['MINDATA']) <= contract_data < float(record['MAXDATA'])):
                return float(record['CORRECTION_FACTOR'])
        return None

    def optimize_pooling(
        self,
        lane: LaneAssignment,
        target_output: Optional[float] = None,
    ) -> PoolingOptimizationResult:
        """
        优化单条Lane的Pooling系数

        Args:
            lane: Lane分配结果
            target_output: 期望产出，默认使用配置值
        """
        target_output = target_output or self.target_output_gb
        return self._optimize_internal(lane.libraries, lane.lane_id, target_output)

    def optimize_for_libraries(
        self,
        libraries: List[EnhancedLibraryInfo],
        lane_id: str = "TEMP_LANE",
        target_output: Optional[float] = None,
    ) -> PoolingOptimizationResult:
        """
        仅基于文库列表优化Pooling系数（包Lane场景）
        """
        target_output = target_output or self.target_output_gb
        return self._optimize_internal(libraries, lane_id, target_output)

    def calculate_simple_coefficients(
        self, libraries: List[EnhancedLibraryInfo]
    ) -> Dict[str, float]:
        """
        简单规则计算Pooling系数（降级路径）
        """
        coefficients: Dict[str, float] = {}
        for lib in libraries:
            contract_data = float(lib.contract_data_raw or 0)
            coefficient = 1.0
            if contract_data < 1:
                coefficient = 1.5
            elif contract_data > 150:
                coefficient = 1.3
            coefficients[lib.origrec] = coefficient
        return coefficients

    def _optimize_internal(
        self, libraries: List[EnhancedLibraryInfo], lane_id: str, target_output: float
    ) -> PoolingOptimizationResult:
        if not libraries:
            logger.warning(f"Lane {lane_id} 无文库，跳过Pooling优化")
            return PoolingOptimizationResult(
                lane_id=lane_id,
                coefficients={},
                predicted_total_output=0.0,
                predicted_avg_output=0.0,
                predicted_cv=0.0,
                original_total=0.0,
                improvement_pct=0.0,
                optimization_applied=False,
                reason="无文库",
                warnings=[],
                order_cv_ratio=0.0,
                under_delivery_gb=0.0,
                over_order_gb=0.0,
                lane_cv=None,
            )

        if not self.enabled or not self.predictor:
            coefficients = self.calculate_simple_coefficients(libraries)
            return self._build_result_without_model(
                lane_id, coefficients, "模型未启用", libraries
            )

        try:
            prediction = self.predictor.predict_lane_output(libraries, lane_id)
            if not prediction.get("prediction_success", False):
                logger.warning(f"Lane {lane_id} 预测失败: {prediction.get('error_message')}")
                coefficients = self.calculate_simple_coefficients(libraries)
                return self._build_result_without_model(
                    lane_id, coefficients, "预测失败，使用规则降级", libraries
                )

            individual_outputs = prediction.get("individual_predictions", [])
            coefficients = self._derive_coefficients(libraries, individual_outputs)
            warnings: List[str] = []
            order_cv_ratio, under_delivery, over_order = self._evaluate_order_metrics(
                libraries, coefficients, individual_outputs, target_output
            )

            if order_cv_ratio > self.order_cv_ratio_threshold:
                warning = (
                    f"Lane {lane_id} order_cv_ratio={order_cv_ratio:.3f} 超过阈值"
                )
                logger.warning(warning)
                if self.fallback_on_error:
                    coefficients = self.calculate_simple_coefficients(libraries)
                    return self._build_result_without_model(
                        lane_id, coefficients, "order_cv_ratio超限，使用规则降级", libraries
                    )
                warnings.append(warning)

            order_amounts = {
                lib.origrec: float(lib.contract_data_raw or 0.0)
                * coefficients.get(lib.origrec, 1.0)
                for lib in libraries
            }
            lane_cv, predicted_total_output2, lane_cv_warnings = self._compute_lane_cv(
                libraries, lane_id, order_amounts
            )
            warnings.extend(lane_cv_warnings)

            predicted_total_scaled = float(
                np.sum(
                    np.array(individual_outputs, dtype=float)
                    * np.array(list(coefficients.values()))
                )
            )
            predicted_total = (
                predicted_total_output2
                if predicted_total_output2 is not None
                else predicted_total_scaled
            )
            predicted_avg = predicted_total / len(libraries)

            total_contract = sum(float(lib.contract_data_raw or 0.0) for lib in libraries)
            predicted_cv = (
                predicted_total_output2 / total_contract
                if predicted_total_output2 is not None and total_contract > 0
                else None
            )
            if predicted_cv is None:
                predicted_cv = order_cv_ratio

            base_total = float(np.sum(individual_outputs))
            improvement = (
                ((predicted_total - base_total) / base_total * 100.0) if base_total > 0 else 0.0
            )

            return PoolingOptimizationResult(
                lane_id=lane_id,
                coefficients=coefficients,
                predicted_total_output=predicted_total,
                predicted_avg_output=predicted_avg,
                predicted_cv=predicted_cv,
                original_total=base_total,
                improvement_pct=improvement,
                optimization_applied=True,
                reason="使用loutput模型推导Pooling系数",
                warnings=warnings,
                order_cv_ratio=order_cv_ratio,
                under_delivery_gb=under_delivery,
                over_order_gb=over_order,
                lane_cv=lane_cv,
            )
        except Exception as exc:
            logger.exception(f"Lane {lane_id} Pooling优化异常，使用规则降级: {exc}")
            coefficients = self.calculate_simple_coefficients(libraries)
            return self._build_result_without_model(
                lane_id, coefficients, "异常降级", libraries
            )

    def _derive_coefficients(
        self, libraries: List[EnhancedLibraryInfo], individual_outputs: List[float]
    ) -> Dict[str, float]:
        """
        计算Pooling系数

        核心逻辑（2026-02-09更新）：
        1. 根据模型预测产出计算基础下单量
        2. 获取类型感知的qPCR质量系数（QPCR方向随文库类型自动调整）
        3. 获取峰图补偿和库检补偿系数（2026-02-09新增，应用于所有路径）
        4. 命中配置表时：基础下单量 x 配置表倍数 x 质量系数 x 峰图补偿 x 库检补偿
        5. 未命中配置表时：基础下单量 x PoolingPredictor完整系数（含全部维度）
           小文库额外补偿
        6. 转换为系数返回（系数 = 下单量 / 合同量）
        """
        from arrange_library.core.ai.pooling_predictor import PoolingPredictor
        pooling_predictor = PoolingPredictor()

        # 大文库调整系数梯度抑制配置
        # 根据模型预测产出率(rate)，采用不同强度的抑制
        # rate越高（模型越认为能产够），抑制越强
        # 2026-02-06 回调: 原damping值(0.3/0.7)与配置表reduction叠加后压制过度
        # 导致大文库下单量远低于历史合理水平，适度放宽damping
        LARGE_LIB_THRESHOLD = 7.0  # >=7G为大文库

        coefficients: Dict[str, float] = {}
        for lib, pred_output in zip(libraries, individual_outputs):
            contract = float(lib.contract_data_raw or 0.0)
            if contract <= 0:
                coefficients[lib.origrec] = 1.0
                continue

            # --- 步骤1：根据模型预测产出计算基础下单量 ---
            rate = pred_output / contract
            if rate <= 0:
                base_order_amount = contract * self.max_coefficient
            else:
                base_order_amount = contract * (self.safety_factor / rate)

            # 应用最小下单量约束
            if base_order_amount < self.min_ordered_data:
                base_order_amount = self.min_ordered_data

            # 大文库标识
            is_large_lib = (contract >= LARGE_LIB_THRESHOLD)

            # 大文库梯度抑制（核心逻辑一优先，保产够率；逻辑二为辅，降溢出）
            # damping=1.0表示不抑制（完整保留补偿），damping越小抑制越强
            # 2026-02-06 回调：原值(0.3/0.7)与配置表reduction叠加导致大文库下单过低
            # rate >= 1.0: 模型预测产够，适度抑制（damping=0.5，原0.3）
            # 0.85 <= rate < 1.0: 模型预测接近产够，轻度抑制（damping=0.85，原0.7）
            # rate < 0.85: 模型预测明显不够，不抑制（damping=1.0）
            if is_large_lib:
                if rate >= 1.0:
                    large_lib_damping = 0.5
                elif rate >= 0.85:
                    large_lib_damping = 0.85
                else:
                    large_lib_damping = 1.0

            # --- 步骤2：获取类型感知的qPCR质量系数 ---
            qpcr = lib.qpcr_concentration
            qubit = lib.qubit_concentration
            sample_type = str(
                getattr(lib, 'sample_type_code', None)
                or getattr(lib, 'sampletype', '')
                or ''
            )
            quality_factor, quality_reason = pooling_predictor._get_quality_factor(
                qpcr, qubit, sample_type
            )

            # --- 步骤2.5：获取峰图补偿和库检补偿（2026-02-09新增） ---
            # 这两个补偿因子独立于配置表路径，对所有文库生效
            # 它们捕捉的是QPCR/Qubit无法反映的质量问题：
            #   - 峰图异常 = 有效片段比例低，同样浓度产出少
            #   - 库检不合格 = 文库质量有隐患，产出效率偏低
            peak_factor, peak_reason = pooling_predictor._get_peak_quality_factor(lib)
            qc_factor, qc_reason = pooling_predictor._get_qc_result_factor(lib)

            # --- 步骤3：检查是否在配置表中 ---
            testno = str(getattr(lib, 'test_no', None) or getattr(lib, 'seq_scheme', '') or '')
            libtype = sample_type

            reduction_factor = self.pooling_config_manager.get_reduction_factor(
                testno, libtype, contract
            )

            # --- 步骤4：计算最终下单量 ---
            # 综合补偿 = 峰图补偿 x 库检补偿（对所有路径生效）
            extra_compensation = peak_factor * qc_factor

            if reduction_factor is not None:
                # 在配置表中：基础下单量 x 配置表倍数 x 质量系数 x 额外补偿
                if is_large_lib:
                    # 大文库：梯度抑制质量系数的额外调整幅度
                    damped_quality = 1.0 + (quality_factor - 1.0) * large_lib_damping
                    final_order_amount = base_order_amount * reduction_factor * damped_quality * extra_compensation
                    extra_desc = f", 峰图x{peak_factor:.2f}({peak_reason}), 库检x{qc_factor:.2f}({qc_reason})" if extra_compensation != 1.0 else ""
                    logger.info(
                        f"文库{lib.origrec}({libtype})命中配置表[大文库抑制d={large_lib_damping:.2f}]: "
                        f"reduction={reduction_factor:.2f}, "
                        f"质量系数{quality_factor:.2f}->{damped_quality:.2f}({quality_reason}){extra_desc}, "
                        f"基础下单={base_order_amount:.2f}GB, rate={rate:.3f}, 最终下单={final_order_amount:.2f}GB"
                    )
                else:
                    final_order_amount = base_order_amount * reduction_factor * quality_factor * extra_compensation
                    extra_desc = f", 峰图x{peak_factor:.2f}({peak_reason}), 库检x{qc_factor:.2f}({qc_reason})" if extra_compensation != 1.0 else ""
                    logger.info(
                        f"文库{lib.origrec}({libtype})命中配置表: "
                        f"reduction={reduction_factor:.2f}, "
                        f"质量系数={quality_factor:.2f}({quality_reason}){extra_desc}, "
                        f"基础下单={base_order_amount:.2f}GB, 最终下单={final_order_amount:.2f}GB"
                    )
            else:
                # 不在配置表中：使用PoolingPredictor的完整系数
                try:
                    pooling_result = pooling_predictor.calculate_coefficient(lib)
                    adjustment_factor = pooling_result.predicted_coefficient
                    # 修复：被质量过滤的文库系数为0，会导致最终下单归零
                    # 这类文库虽然质量指标异常，但仍然需要正常下单
                    # 使用默认系数1.0（即按模型base_order下单，不做额外调整）
                    if pooling_result.is_filtered or adjustment_factor <= 0:
                        logger.warning(
                            f"文库{lib.origrec}({sample_type})被质量过滤或系数<=0, "
                            f"原因: {pooling_result.filter_reason}, "
                            f"使用默认系数1.0代替"
                        )
                        adjustment_factor = 1.0
                except Exception as e:
                    logger.warning(f"文库{lib.origrec}获取Pooling系数失败: {e}")
                    adjustment_factor = 1.0

                if is_large_lib:
                    # 大文库(>=7G)：梯度抑制额外调整系数幅度
                    # damping越小抑制越强：rate高时强抑制，rate低时保留更多补偿
                    damped_factor = 1.0 + (adjustment_factor - 1.0) * large_lib_damping
                    final_order_amount = base_order_amount * damped_factor
                    logger.debug(
                        f"大文库{lib.origrec}梯度抑制(d={large_lib_damping:.2f}): "
                        f"adjustment {adjustment_factor:.3f}->{damped_factor:.3f}, "
                        f"合同={contract:.2f}GB, rate={rate:.3f}, 最终下单={final_order_amount:.2f}GB"
                    )
                else:
                    # 小文库(<7G)：完整应用调整系数 + 类型感知的额外补偿
                    # 核心思想：天生产出效率高的类型不需要大幅补偿，效率低的才需要
                    final_order_amount = base_order_amount * adjustment_factor

                    # 从LIBRARY_TYPE_CONFIG获取类型效率系数来决定补偿力度
                    from arrange_library.core.ai.pooling_predictor import LIBRARY_TYPE_CONFIG
                    type_eff = LIBRARY_TYPE_CONFIG.get(sample_type, {}).get(
                        'efficiency_factor', 1.0
                    )

                    # 核心理念：天生产的好的类型，模型base_order+size_factor已经足够
                    #          不需要再额外叠加，否则系数爆炸
                    # 只有低效率类型才需要额外补偿来保产够率
                    if type_eff < 0.96:
                        # 中高效率类型(ATAC/CUT-Tag/DNA小片段/Meta/VIP真核等)
                        # 这些类型天生产出好，不额外补偿
                        small_lib_factor = 1.0
                    elif type_eff < 1.05:
                        # 偏低效率(外显子/人重测序/Chip-seq等)
                        small_lib_factor = 1.2
                    else:
                        # 低效率类型(10X-3V4/RRBS/Visium等)：完整补偿
                        small_lib_factor = 1.5

                    final_order_amount = final_order_amount * small_lib_factor
                    logger.debug(
                        f"小文库{lib.origrec}({sample_type})类型感知补偿: "
                        f"type_eff={type_eff:.2f}->factor={small_lib_factor:.2f}, "
                        f"合同={contract:.2f}GB, 最终下单={final_order_amount:.2f}GB"
                    )

            # --- 步骤5：产出效率修正（后校准） ---
            # 基于历史数据分析：对产够率高且溢出偏高的类型x合同区间，
            # 乘一个修正系数降低过度预测，同时对CL1影响极小(<=0.8%)
            eff_correction = self._get_efficiency_correction(libtype, contract)
            if eff_correction is not None:
                original_order = final_order_amount
                final_order_amount = final_order_amount * eff_correction
                logger.debug(
                    f"文库{lib.origrec}({libtype}, 合同{contract:.1f}GB)产出效率修正: "
                    f"x{eff_correction:.4f}, {original_order:.2f}->{final_order_amount:.2f}GB"
                )

            # --- 步骤6：转换为系数 ---
            coef = final_order_amount / contract

            # 应用系数上下限
            coef = float(np.clip(coef, self.min_coefficient, self.max_coefficient))

            coefficients[lib.origrec] = coef

        return coefficients

    def _compute_lane_cv(
        self,
        libraries: List[EnhancedLibraryInfo],
        lane_id: str,
        order_amounts: Dict[str, float],
    ) -> Tuple[Optional[float], Optional[float], List[str]]:
        warnings: List[str] = []
        if not self.predictor:
            return None, None, ["lane_cv未计算: 预测器不可用"]

        prediction = self.predictor.predict_lane_output(
            libraries,
            lane_id,
            contract_overrides=order_amounts,
            total_contract_overrides=order_amounts,
        )
        if not prediction.get("prediction_success", False):
            message = prediction.get("error_message") or "未知原因"
            return None, None, [f"lane_cv未计算: 二次预测失败({message})"]

        outputs = prediction.get("individual_predictions", [])
        if len(outputs) != len(libraries):
            warnings.append(
                f"lane_cv预测长度不匹配: {len(outputs)} vs {len(libraries)}"
            )

        # 按“二次预测产出 / 原始合同数据量”计算lane_cv（合同量不使用覆盖值）
        contracts = [float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in libraries]
        lane_cv, predicted_total_output, calc_warnings = self._calculate_lane_cv(outputs, contracts)
        warnings.extend(calc_warnings)
        return lane_cv, predicted_total_output, warnings

    def _calculate_lane_cv(
        self,
        outputs: List[float],
        contracts: Optional[List[float]] = None,
    ) -> Tuple[Optional[float], Optional[float], List[str]]:
        """
        计算Lane CV（变异系数）
        
        修正说明（2026-01-23）：
        - 确保outputs和contracts同步过滤，避免长度不匹配
        - 使用产出率（output/contract）计算CV，而不是绝对值
        """
        warnings: List[str] = []
        if not outputs:
            warnings.append("lane_cv无法计算: 预测产出为空")
            return None, None, warnings

        # 检查contracts参数
        if not contracts or len(contracts) != len(outputs):
            warnings.append(f"lane_cv无法计算: 合同量缺失或长度不匹配 (outputs={len(outputs)}, contracts={len(contracts) if contracts else 0})")
            return None, None, warnings

        # 同步过滤outputs和contracts，确保一一对应
        valid_pairs: List[Tuple[float, float]] = []
        skipped = 0
        for output, contract in zip(outputs, contracts):
            try:
                out_val = float(output)
                ct_val = float(contract)
            except (TypeError, ValueError):
                skipped += 1
                continue
            if not (np.isfinite(out_val) and np.isfinite(ct_val)):
                skipped += 1
                continue
            if ct_val <= 0:
                skipped += 1
                continue
            valid_pairs.append((out_val, ct_val))

        if skipped:
            warnings.append(f"lane_cv忽略{skipped}个无效数据对")

        if not valid_pairs:
            warnings.append("lane_cv无法计算: 有效数据对为空")
            return None, None, warnings

        if len(valid_pairs) < 2:
            warnings.append("lane_cv无法计算: 有效数据对不足")
            return None, None, warnings

        # 计算产出率列表（产出/合同）
        valid_ratios = [out / ct for out, ct in valid_pairs]
        predicted_total_output = float(sum(out for out, _ in valid_pairs))
        
        mean_ratio = float(np.mean(valid_ratios))
        if mean_ratio <= 0:
            warnings.append("lane_cv无法计算: 产出率均值<=0")
            return None, predicted_total_output, warnings
        
        # CV = std(产出率) / mean(产出率)
        lane_cv = float(np.std(valid_ratios) / mean_ratio)
        return lane_cv, predicted_total_output, warnings

    def _evaluate_order_metrics(
        self,
        libraries: List[EnhancedLibraryInfo],
        coefficients: Dict[str, float],
        individual_outputs: List[float],
        target_output: float,
    ) -> Tuple[float, float, float]:
        contract_sum = 0.0
        ordered_sum = 0.0
        predicted_total = 0.0
        for lib, pred in zip(libraries, individual_outputs):
            contract = float(lib.contract_data_raw or 0.0)
            contract_sum += contract
            coef = coefficients.get(lib.origrec, 1.0)
            ordered = contract * coef
            ordered_sum += ordered
            predicted_total += pred * coef

        order_cv_ratio = ordered_sum / contract_sum if contract_sum > 0 else 0.0
        under_delivery = max(0.0, target_output - predicted_total)
        over_order = max(0.0, ordered_sum - contract_sum)
        return order_cv_ratio, under_delivery, over_order

    def _predict_order_contract_ratio_from_outputs(
        self,
        libraries: List[EnhancedLibraryInfo],
        individual_outputs: List[float],
        target_output_rate: float = 1.0,
    ) -> Optional[float]:
        """
        基于预测产出反推出Lane级“下单/合同比”。

        order_i = predicted_output_i / target_output_rate
        ratio = sum(order_i) / sum(contract_i)
        """
        if not libraries or not individual_outputs or target_output_rate <= 0:
            return None

        max_order_ratio = self.pooling_config.max_ordered_ratio
        min_ordered_gb = self.pooling_config.min_ordered_gb
        total_contract = 0.0
        total_order = 0.0

        for lib, pred_output in zip(libraries, individual_outputs):
            contract = float(lib.contract_data_raw or 0.0)
            if contract <= 0:
                continue
            total_contract += contract
            if pred_output is None:
                continue
            order = float(pred_output) / target_output_rate
            order = max(order, min_ordered_gb)
            order = min(order, contract * max_order_ratio)
            total_order += order

        if total_contract <= 0:
            return None
        return total_order / total_contract

    def _build_result_without_model(
        self,
        lane_id: str,
        coefficients: Dict[str, float],
        reason: str,
        libraries: Optional[List[EnhancedLibraryInfo]] = None,
    ) -> PoolingOptimizationResult:
        order_cv_ratio = 0.0
        over_order = 0.0
        if libraries:
            contract_sum = 0.0
            ordered_sum = 0.0
            for lib in libraries:
                contract = float(lib.contract_data_raw or 0.0)
                contract_sum += contract
                coef = coefficients.get(lib.origrec, 1.0)
                ordered_sum += contract * coef
            if contract_sum > 0:
                order_cv_ratio = ordered_sum / contract_sum
                over_order = max(0.0, ordered_sum - contract_sum)
        return PoolingOptimizationResult(
            lane_id=lane_id,
            coefficients=coefficients,
            predicted_total_output=0.0,
            predicted_avg_output=0.0,
            predicted_cv=order_cv_ratio,
            original_total=0.0,
            improvement_pct=0.0,
            optimization_applied=False,
            reason=reason,
            warnings=[reason],
            order_cv_ratio=order_cv_ratio,
            under_delivery_gb=0.0,
            over_order_gb=over_order,
            lane_cv=None,
        )

    def _load_config(self) -> Dict[str, Any]:
        """
        加载优化器配置，优先读取独立配置文件，缺失时使用默认值
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as file:
                    raw = yaml.safe_load(file) or {}
                return raw.get("pooling_optimizer", raw)
            except Exception as exc:
                logger.warning(f"加载Pooling优化配置失败，使用默认值: {exc}")
        else:
            logger.warning(f"未找到Pooling优化配置文件: {self.config_path}，使用默认值")

        return {
            "enabled": True,
            "safety_factor": 1.05,
            "max_coefficient": 2.5,
            "min_coefficient": 0.5,
            "min_ordered_data_gb": 1.0,
            "order_cv_ratio_threshold": 1.15,
            "fallback_on_error": True,
            "target_output_gb": 90.0,
        }
