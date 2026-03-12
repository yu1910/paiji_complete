"""
Pooling系数预测器
创建时间：2025-11-20
更新时间：2026-02-09 14:30:00
功能：基于质量、数据量和文库类型特征动态计算Pooling系数

核心改进（2026-02-06）：
  - 引入文库类型产出效率修正：不同文库类型有显著不同的产出效率特征
  - QPCR质量系数支持按文库类型调整方向：部分类型高QPCR反而效率更低
  - 以上结论均来自37万+条历史数据的多维度交叉分析

优化更新（2026-02-09）：
  - 基于V5第六轮测试结果分析，针对未产够文库进行定向优化
  - 新增峰图描述质量补偿：接头污染/小片段+杂峰的文库需额外补偿
  - 新增库检结果补偿：不合格/风险建库的文库需额外补偿
  - 调整QPCR极端值补偿力度：低QPCR加强补偿、逆向模式高QPCR加强补偿
  - 调整高Qubit逻辑：取消对高Qubit的一刀切压制（数据证实高Qubit也有高欠产率）
  - 微调部分易欠产文库类型的效率系数
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from models.library_info import EnhancedLibraryInfo


# ---------------------------------------------------------------------------
# 文库类型产出行为配置（基于历史数据分析结果）
# ---------------------------------------------------------------------------
# efficiency_factor: 产出效率修正系数，=1/avg_efficiency 再做阻尼收敛
#   >1.0 表示该类型历史上偏低产，需要多下单
#   <1.0 表示该类型历史上偏高产，需要适度压制
# qpcr_mode: QPCR对产出效率的影响方向
#   'normal'  - 标准行为：低QPCR需要补偿，高QPCR可适当减少（大多数常规文库）
#   'inverse' - 逆向行为：高QPCR反而效率更低，需要适当增加下单（人重测序、Lnc等）
#   'neutral' - QPCR影响极小：固定为1.0，不做调整（10X单细胞类）
# ---------------------------------------------------------------------------
LIBRARY_TYPE_CONFIG: Dict[str, Dict] = {
    # --- 高效率型：历史平均效率>1.2，容易溢出 ---
    'ATAC-seq文库':                 {'efficiency_factor': 0.72, 'qpcr_mode': 'inverse'},
    '客户-ATAC-seq文库':            {'efficiency_factor': 0.75, 'qpcr_mode': 'inverse'},
    '客户-CUT-Tag文库':             {'efficiency_factor': 0.78, 'qpcr_mode': 'inverse'},
    # 2026-02-09: CUT Tag(细胞)从0.80上调到0.85, 测试数据显示存在欠产风险
    'CUT Tag文库（细胞）':          {'efficiency_factor': 0.85, 'qpcr_mode': 'inverse'},
    'CUT Tag文库（动物组织）':      {'efficiency_factor': 0.80, 'qpcr_mode': 'inverse'},
    '单细胞文库':                   {'efficiency_factor': 0.82, 'qpcr_mode': 'neutral'},
    'Agilent全外文库':              {'efficiency_factor': 0.82, 'qpcr_mode': 'normal'},
    '微生物常规小片段文库':         {'efficiency_factor': 0.84, 'qpcr_mode': 'normal'},
    '10X转录组V(D)J-BCR文库':      {'efficiency_factor': 0.84, 'qpcr_mode': 'neutral'},
    '10X转录组V(D)J-TCR文库':      {'efficiency_factor': 0.85, 'qpcr_mode': 'neutral'},
    '客户-扩增子文库':              {'efficiency_factor': 0.85, 'qpcr_mode': 'normal'},
    '客户-微生物常规小片段文库':    {'efficiency_factor': 0.85, 'qpcr_mode': 'inverse'},
    'RIP-seq文库':                  {'efficiency_factor': 0.85, 'qpcr_mode': 'normal'},
    'IDT全外文库':                  {'efficiency_factor': 0.85, 'qpcr_mode': 'normal'},
    'Meta文库':                     {'efficiency_factor': 0.86, 'qpcr_mode': 'normal'},
    # 2026-02-09: 客户-Chip-seq从0.86上调到0.92, 测试数据显示客户自建Chip-seq欠产率高
    '客户-Chip-seq文库':            {'efficiency_factor': 0.92, 'qpcr_mode': 'inverse'},
    '动植物全基因组重测序文库':     {'efficiency_factor': 0.87, 'qpcr_mode': 'normal'},
    '客户-其他碱基均衡文库':        {'efficiency_factor': 0.87, 'qpcr_mode': 'inverse'},
    'DNA小片段文库':                {'efficiency_factor': 0.88, 'qpcr_mode': 'normal'},

    # --- 正常效率型：历史平均效率0.95~1.2 ---
    '客户-Hi-C文库':                {'efficiency_factor': 0.90, 'qpcr_mode': 'normal'},
    '客户-人重测序文库':            {'efficiency_factor': 0.90, 'qpcr_mode': 'inverse'},
    '真核链特异性转录组文库':       {'efficiency_factor': 0.90, 'qpcr_mode': 'inverse'},
    '客户-Meta文库':                {'efficiency_factor': 0.90, 'qpcr_mode': 'normal'},
    'Lnc非互作文库':                {'efficiency_factor': 0.92, 'qpcr_mode': 'inverse'},
    '真核普通转录组文库':           {'efficiency_factor': 0.92, 'qpcr_mode': 'normal'},
    'VIP-真核普通转录组文库':       {'efficiency_factor': 0.93, 'qpcr_mode': 'normal'},
    # 2026-02-09: 微量Methylation从0.93上调到0.98, 甲基化文库整体容易欠产
    '微量 (Methylation文库)':       {'efficiency_factor': 0.98, 'qpcr_mode': 'normal'},
    '客户-真核普通转录组文库':      {'efficiency_factor': 0.94, 'qpcr_mode': 'inverse'},
    '宏转录组':                     {'efficiency_factor': 0.94, 'qpcr_mode': 'normal'},
    # 2026-02-09: 客户-PCR产物从0.95上调到1.02, 测试数据显示PCR产物欠产率偏高
    '客户-PCR产物':                 {'efficiency_factor': 1.02, 'qpcr_mode': 'inverse'},
    '原核链特异性转录组文库':       {'efficiency_factor': 0.95, 'qpcr_mode': 'inverse'},
    '人重测序文库':                 {'efficiency_factor': 0.95, 'qpcr_mode': 'inverse'},
    # 2026-02-09: 外显子文库从0.96上调到1.02, 测试数据中外显子欠产率显著偏高
    '外显子文库':                   {'efficiency_factor': 1.02, 'qpcr_mode': 'normal'},
    'Chip-seq文库':                 {'efficiency_factor': 0.98, 'qpcr_mode': 'inverse'},
    '客户-单细胞文库':              {'efficiency_factor': 0.98, 'qpcr_mode': 'neutral'},
    # 2026-02-09: 客户-Methylation从0.98上调到1.02, 客户甲基化文库欠产率偏高
    '客户-Methylation文库':         {'efficiency_factor': 1.02, 'qpcr_mode': 'normal'},
    'EM-Seq文库':                   {'efficiency_factor': 0.98, 'qpcr_mode': 'normal'},
    '墨卓转录组-3端文库':           {'efficiency_factor': 1.00, 'qpcr_mode': 'neutral'},
    '10X转录组-5\'文库':            {'efficiency_factor': 1.00, 'qpcr_mode': 'inverse'},
    'small RNA文库':                {'efficiency_factor': 0.90, 'qpcr_mode': 'normal'},
    'Ribo-seq文库':                 {'efficiency_factor': 0.90, 'qpcr_mode': 'normal'},
    '全基因组文库':                 {'efficiency_factor': 0.98, 'qpcr_mode': 'normal'},
    'RNA真普文库':                  {'efficiency_factor': 0.96, 'qpcr_mode': 'normal'},

    # --- 低效率型：历史平均效率<0.95，容易产不够 ---
    '10X转录组-3\'文库':            {'efficiency_factor': 1.05, 'qpcr_mode': 'inverse'},
    '10X转录组文库-3V4文库':        {'efficiency_factor': 1.20, 'qpcr_mode': 'inverse'},
    '10X转录组文库-5V3文库':        {'efficiency_factor': 1.05, 'qpcr_mode': 'neutral'},
    '10x HD Visium空间转录组文库(新)': {'efficiency_factor': 1.30, 'qpcr_mode': 'neutral'},
    'RRBS文库':                     {'efficiency_factor': 1.12, 'qpcr_mode': 'normal'},
    '客户-10X 3 单细胞转录组文库':  {'efficiency_factor': 1.10, 'qpcr_mode': 'neutral'},
    'mNGS文库':                     {'efficiency_factor': 1.00, 'qpcr_mode': 'normal'},
    '客户-10X转录组文库':           {'efficiency_factor': 1.00, 'qpcr_mode': 'neutral'},
    '客户-其他碱基不均衡文库':      {'efficiency_factor': 0.95, 'qpcr_mode': 'inverse'},
    '客户-动植物全基因组重测序文库': {'efficiency_factor': 0.95, 'qpcr_mode': 'inverse'},
    # 2026-02-09: 新增鼠全外显子文库，测试数据中欠产率偏高
    '鼠全外显子文库':               {'efficiency_factor': 1.05, 'qpcr_mode': 'normal'},
}


@dataclass
class PoolingPrediction:
    library_id: str
    predicted_coefficient: float
    reasoning: str
    confidence: float
    source: str = "Rule"
    quality_factor: float = 1.0
    size_factor: float = 1.0
    type_efficiency_factor: float = 1.0
    is_filtered: bool = False
    filter_reason: Optional[str] = None


class PoolingPredictor:
    """
    Pooling系数预测器

    核心策略：最终下单倍数 = 质量系数 * 数据量系数 * 文库类型效率系数
    - 质量系数：基于QPCR和Qubit，低质量文库提高倍数；部分类型QPCR方向反转
    - 数据量系数：基于合同数据量，小文库提高倍数保证最小产出
    - 文库类型效率系数：修正不同文库类型的产出效率偏差
    - 上下限：0.85-2.0
    """

    def __init__(self):
        # 系数边界
        self.min_coefficient = 0.85
        self.max_coefficient = 2.0

        # 质量阈值配置
        self.qpcr_thresholds = {
            'very_low': 1.0,
            'low': 2.0,
            'medium': 5.0,
            'normal': 10.0,
            'high': 15.0
        }

        self.qubit_thresholds = {
            'very_low': 2.0,
            'low': 5.0,
            'normal': 80.0
        }

        # 质量过滤规则
        self.filter_qpcr_threshold = 1.0
        self.filter_contract_threshold = 10.0
        self.filter_poor_quality_qpcr = 2.0
        self.filter_poor_quality_qubit = 2.0
    
    def _get_quality_factor(
        self,
        qpcr: Optional[float],
        qubit: Optional[float],
        sample_type: Optional[str] = None
    ) -> Tuple[float, str]:
        """
        计算质量系数（基于QPCR和Qubit值，并根据文库类型调整QPCR方向）

        三种QPCR模式（由LIBRARY_TYPE_CONFIG决定）：
        - normal:  低QPCR多补偿、高QPCR少补偿（绝大多数常规文库）
        - inverse: 高QPCR反而需要更多补偿（人重测序、Lnc非互作等）
        - neutral: QPCR对产出影响极小，统一1.0

        Args:
            qpcr:  QPCR摩尔浓度
            qubit: Qubit值
            sample_type: 文库类型名称，用于查找QPCR模式

        Returns:
            (质量系数, 原因说明)
        """
        # 确定当前文库的QPCR模式
        qpcr_mode = 'normal'
        if sample_type and sample_type in LIBRARY_TYPE_CONFIG:
            qpcr_mode = LIBRARY_TYPE_CONFIG[sample_type].get('qpcr_mode', 'normal')

        qpcr_factor = 1.0
        qubit_factor = 1.0
        qpcr_desc = ""
        qubit_desc = ""

        # ---- QPCR系数（分模式处理）----
        if qpcr is not None:
            if qpcr_mode == 'neutral':
                # 中性模式：QPCR影响极小，统一1.0
                qpcr_factor = 1.0
                qpcr_desc = f"QPCR={qpcr:.1f}(中性类型,不调整)"

            elif qpcr_mode == 'inverse':
                # 逆向模式：高QPCR反而需要更多补偿
                # 历史数据显示这类文库低QPCR效率高、高QPCR效率低
                # 2026-02-09: 高QPCR段(>15)补偿从1.15上调到1.25，
                #   测试数据验证QPCR>15区间未产够率达56%，需要更强补偿
                if qpcr < self.qpcr_thresholds['very_low']:
                    qpcr_factor = 1.15
                    qpcr_desc = f"QPCR<{self.qpcr_thresholds['very_low']}(逆向:低Q高产)"
                elif qpcr < self.qpcr_thresholds['low']:
                    qpcr_factor = 1.10
                    qpcr_desc = f"QPCR<{self.qpcr_thresholds['low']}(逆向:低Q高产)"
                elif qpcr < self.qpcr_thresholds['medium']:
                    qpcr_factor = 1.05
                    qpcr_desc = f"QPCR<{self.qpcr_thresholds['medium']}(逆向)"
                elif qpcr <= self.qpcr_thresholds['normal']:
                    qpcr_factor = 1.0
                    qpcr_desc = "QPCR正常(逆向)"
                elif qpcr <= self.qpcr_thresholds['high']:
                    qpcr_factor = 1.15
                    qpcr_desc = f"QPCR>{self.qpcr_thresholds['normal']}(逆向:高Q低产,需补偿)"
                else:
                    qpcr_factor = 1.25
                    qpcr_desc = f"QPCR>{self.qpcr_thresholds['high']}(逆向:高Q低产,强补偿)"

            else:
                # 标准模式：低QPCR多补偿、高QPCR少补偿
                # 2026-02-09: 极低QPCR(<1.0)补偿从1.5上调到1.6，
                #   测试数据验证QPCR 0-3区间未产够率达57.1%，需要更强保障
                # 2026-02-09: 高QPCR(>15)取消压制，不再降到0.85，
                #   测试数据显示高QPCR区间也有56%的未产够率，可能是虚高浓度
                if qpcr < self.qpcr_thresholds['very_low']:
                    qpcr_factor = 1.6
                    qpcr_desc = f"QPCR<{self.qpcr_thresholds['very_low']}(极低,强补偿)"
                elif qpcr < self.qpcr_thresholds['low']:
                    qpcr_factor = 1.4
                    qpcr_desc = f"QPCR<{self.qpcr_thresholds['low']}"
                elif qpcr < self.qpcr_thresholds['medium']:
                    qpcr_factor = 1.2
                    qpcr_desc = f"QPCR<{self.qpcr_thresholds['medium']}"
                elif qpcr <= self.qpcr_thresholds['normal']:
                    qpcr_factor = 1.0
                    qpcr_desc = "QPCR正常"
                elif qpcr <= self.qpcr_thresholds['high']:
                    qpcr_factor = 0.95
                    qpcr_desc = f"QPCR>{self.qpcr_thresholds['normal']}(轻度压制)"
                else:
                    qpcr_factor = 0.92
                    qpcr_desc = f"QPCR>{self.qpcr_thresholds['high']}(高QPCR可能虚高)"

        # ---- Qubit系数 ----
        # 2026-02-09: 取消高Qubit的压制（原Qubit>80时系数0.90）
        # 测试数据分析发现：Qubit>60的文库反而有46.4%未产够率，
        # 高Qubit可能是因为小片段/接头污染导致的虚高浓度，
        # 对这类文库压制下单反而加剧了欠产问题。
        # 现在只保留低Qubit的补偿逻辑，高Qubit统一不调整。
        if qubit is not None:
            if qubit < self.qubit_thresholds['very_low']:
                qubit_factor = 1.4
                qubit_desc = f"Qubit<{self.qubit_thresholds['very_low']}"
            elif qubit < self.qubit_thresholds['low']:
                qubit_factor = 1.3
                qubit_desc = f"Qubit<{self.qubit_thresholds['low']}"
            else:
                # Qubit正常或偏高，均不做调整
                qubit_factor = 1.0
                qubit_desc = "Qubit正常(>=5不调整)"

        # 取较大值（更保守的策略）
        quality_factor = max(qpcr_factor, qubit_factor)

        # 生成说明
        if quality_factor == qpcr_factor and quality_factor == qubit_factor:
            reason = f"{qpcr_desc}, {qubit_desc}"
        elif quality_factor == qpcr_factor:
            reason = qpcr_desc
        else:
            reason = qubit_desc

        return quality_factor, reason

    def _get_type_efficiency_factor(self, sample_type: Optional[str]) -> Tuple[float, str]:
        """
        获取文库类型产出效率修正系数

        不同文库类型的历史产出效率有显著差异，该系数用于修正模型预测偏差：
        - 高效率类型（如ATAC-seq）：系数<1.0，压制下单避免溢出
        - 低效率类型（如10X-3V4）：系数>1.0，增加下单避免欠产
        - 未知类型：返回1.0，不做额外修正

        Args:
            sample_type: 文库类型名称（wksampletype）

        Returns:
            (效率修正系数, 说明)
        """
        if not sample_type:
            return 1.0, "未知文库类型"

        config = LIBRARY_TYPE_CONFIG.get(sample_type)
        if config is None:
            return 1.0, f"{sample_type}(无历史效率数据)"

        factor = config['efficiency_factor']
        if factor < 0.90:
            desc = f"{sample_type}(高效率型,压制{factor:.2f})"
        elif factor > 1.05:
            desc = f"{sample_type}(低效率型,补偿{factor:.2f})"
        else:
            desc = f"{sample_type}(效率正常{factor:.2f})"

        return factor, desc
    
    def _get_size_factor(self, contract_amount: float) -> Tuple[float, str]:
        """
        计算数据量系数（基于合同数据量）
        
        策略说明（基于实际上机数据分析和反馈）：
        - 小数据量文库（<7G）：未产够率高，需要较高倍数补偿来保证最小产出
        - 中大数据量文库（>=7G）：保持1.0不做压制，大文库的压制统一由配置表控制
        
        Args:
            contract_amount: 合同数据量（G）
            
        Returns:
            (数据量系数, 原因说明)
        """
        if contract_amount <= 1.5:
            # 极小文库，未产够率很高，需要大幅补偿
            return 1.6, "0-1.5G极小文库"
        elif contract_amount <= 2.5:
            return 1.45, "1.5-2.5G小文库"
        elif contract_amount <= 4:
            return 1.35, "2.5-4G小文库"
        elif contract_amount <= 5:
            return 1.25, "4-5G小文库"
        elif contract_amount <= 7:
            return 1.2, "5-7G小文库"
        else:
            # >=7G的文库不在此处做压制，大文库的压制由配置表(wktype_pooling_config.xlsx)统一控制
            return 1.0, ">=7G文库（压制由配置表控制）"
    
    def _check_quality_filter(self, lib: EnhancedLibraryInfo) -> Tuple[bool, Optional[str]]:
        """
        检查文库是否应被过滤（质量过低）
        
        Args:
            lib: 文库信息
            
        Returns:
            (是否过滤, 过滤原因)
        """
        qpcr = lib.qpcr_concentration
        qubit = lib.qubit_concentration
        contract = float(lib.contract_data_raw or 0)
        
        # 规则1: QPCR < 1.0 且 合同量 > 10G
        if (qpcr is not None and qpcr < self.filter_qpcr_threshold 
            and contract > self.filter_contract_threshold):
            return True, f"QPCR<{self.filter_qpcr_threshold}且合同量>{self.filter_contract_threshold}G，质量极差"
        
        # 规则2: Qubit < 2.0 且 QPCR < 2.0
        if (qubit is not None and qubit < self.filter_poor_quality_qubit 
            and qpcr is not None and qpcr < self.filter_poor_quality_qpcr):
            return True, f"Qubit<{self.filter_poor_quality_qubit}且QPCR<{self.filter_poor_quality_qpcr}，双指标质量差"
        
        return False, None

    def _get_peak_quality_factor(self, lib: EnhancedLibraryInfo) -> Tuple[float, str]:
        """
        基于峰图描述计算额外补偿系数

        测试数据分析结果（2026-02-09）：
        - 接头污染：100%未产够，需要最强补偿
        - 小片段+杂峰：57.1%未产够，需要中等补偿
        - 仅有杂峰/碎片描述：轻度补偿
        - 正常峰图：不补偿

        补偿逻辑：峰图异常意味着文库中有效测序片段比例降低，
        同样的QPCR浓度对应更少的有效数据产出，因此需要多下单补偿。

        Args:
            lib: 文库信息

        Returns:
            (补偿系数, 原因说明)
        """
        peak_map = getattr(lib, 'peak_map', None) or ''
        xpd = getattr(lib, 'xpd', None)  # 小片段占比
        jtb = getattr(lib, 'jtb', None)  # 接头比值

        # 先看峰图描述文字中有没有关键风险词
        has_adapter = '接头' in peak_map or 'adapter' in peak_map.lower()
        has_small_frag = '小片段' in peak_map or '碎片' in peak_map
        has_noise = '杂峰' in peak_map or '杂带' in peak_map

        # 最严重：接头污染（有效片段被接头挤占，产出严重不足）
        if has_adapter:
            return 1.30, f"峰图接头污染({peak_map[:20]})"

        # 次严重：小片段+杂峰同时存在
        if has_small_frag and has_noise:
            return 1.20, f"峰图小片段+杂峰({peak_map[:20]})"

        # 辅助判断：数值化的小片段占比和接头比值
        # 小片段占比>20%或接头比值>10%时也需要补偿
        if xpd is not None and xpd > 0.20:
            return 1.15, f"小片段占比{xpd:.1%}偏高"

        if jtb is not None and jtb > 0.10:
            return 1.15, f"接头比值{jtb:.2f}偏高"

        # 仅有杂峰/碎片
        if has_small_frag or has_noise:
            return 1.10, f"峰图{peak_map[:20]}"

        return 1.0, "峰图正常"

    def _get_qc_result_factor(self, lib: EnhancedLibraryInfo) -> Tuple[float, str]:
        """
        基于库检综合结果和建库风险计算补偿系数

        测试数据分析结果（2026-02-09）：
        - 库检不合格：45.5%未产够率，需要补偿
        - 风险建库：通常质量偏低，需要适度补偿
        - 合格/正常建库：不额外补偿

        逻辑：库检不合格意味着文库质量有隐患，同样浓度下产出效率偏低。
        风险建库的文库虽然勉强通过，但产出稳定性较差。

        Args:
            lib: 文库信息

        Returns:
            (补偿系数, 原因说明)
        """
        complex_result = getattr(lib, 'complex_result', None) or ''
        risk_flag = getattr(lib, 'risk_build_flag', None) or ''

        # 库检不合格，需要较强补偿
        if '不合格' in complex_result:
            return 1.15, f"库检不合格({complex_result})"

        # 风险建库标识
        if '风险' in risk_flag:
            return 1.10, f"风险建库({risk_flag})"

        return 1.0, "库检正常"

    def calculate_coefficient(self, lib: EnhancedLibraryInfo) -> PoolingPrediction:
        """
        计算单个文库的Pooling系数

        最终系数 = 质量系数 * 数据量系数 * 文库类型效率修正系数 * 峰图补偿 * 库检补偿
        五个维度各司其职，互不覆盖：
        - 质量系数：反映qPCR/Qubit浓度对产出的影响（且方向随类型调整）
        - 数据量系数：小文库额外补偿
        - 类型效率系数：修正不同文库类型的系统性产出偏差
        - 峰图补偿：接头污染/小片段/杂峰等异常峰图的额外补偿
        - 库检补偿：不合格/风险建库的额外补偿

        Args:
            lib: 文库信息

        Returns:
            PoolingPrediction对象
        """
        contract_amount = float(lib.contract_data_raw or 0)
        sample_type = getattr(lib, 'sample_type_code', None) or ''

        # 检查质量过滤
        is_filtered, filter_reason = self._check_quality_filter(lib)
        if is_filtered:
            logger.warning(f"文库 {lib.origrec} 被过滤: {filter_reason}")
            return PoolingPrediction(
                library_id=lib.origrec,
                predicted_coefficient=0.0,
                reasoning=filter_reason,
                confidence=1.0,
                source="Filter",
                is_filtered=True,
                filter_reason=filter_reason
            )

        if contract_amount <= 0:
            logger.warning(f"文库 {lib.origrec} 合同数据量<=0，使用默认系数1.0")
            return PoolingPrediction(
                library_id=lib.origrec,
                predicted_coefficient=1.0,
                reasoning="合同数据量无效，使用默认系数",
                confidence=0.5,
                source="Default"
            )

        # 计算质量系数（根据文库类型自动调整QPCR方向）
        qpcr = lib.qpcr_concentration
        qubit = lib.qubit_concentration
        quality_factor, quality_reason = self._get_quality_factor(qpcr, qubit, sample_type)

        # 计算数据量系数
        size_factor, size_reason = self._get_size_factor(contract_amount)

        # 计算文库类型效率修正系数
        type_factor, type_reason = self._get_type_efficiency_factor(sample_type)

        # 计算峰图补偿系数（2026-02-09新增）
        peak_factor, peak_reason = self._get_peak_quality_factor(lib)

        # 计算库检结果补偿系数（2026-02-09新增）
        qc_factor, qc_reason = self._get_qc_result_factor(lib)

        # 中高效率类型的小文库：size补偿与类型效率存在矛盾
        # 大量数据证实：eff<0.96的类型天生产出效率较高
        # (如DNA小片段1.78x、Meta1.59x、VIP-真核1.26x)
        # 但少数没产够的仍然需要保留一定补偿
        # 折中方案：保留部分补偿（用type_factor做阻尼）
        if type_factor < 0.96 and size_factor > 1.1:
            retain_ratio = (type_factor / 0.96) * 0.2
            old_size = size_factor
            size_factor = 1.0 + (size_factor - 1.0) * retain_ratio
            logger.debug(
                f"高效率类型{sample_type}(eff={type_factor:.2f})削减size补偿: "
                f"{old_size:.2f}->{size_factor:.2f}(保留{retain_ratio:.0%})"
            )
            size_reason += f"(高效率削减至{size_factor:.2f})"

        # 最终系数 = 五个维度相乘
        coefficient = quality_factor * size_factor * type_factor * peak_factor * qc_factor

        # 构造可读的推理说明
        reasoning_parts = [
            f"质量{quality_factor:.2f}({quality_reason})",
            f"x 数据量{size_factor:.2f}({size_reason})",
            f"x 类型效率{type_factor:.2f}({type_reason})",
        ]
        # 只有峰图/库检系数不为1.0时才显示，避免日志冗余
        if peak_factor != 1.0:
            reasoning_parts.append(f"x 峰图{peak_factor:.2f}({peak_reason})")
        if qc_factor != 1.0:
            reasoning_parts.append(f"x 库检{qc_factor:.2f}({qc_reason})")
        reasoning_parts.append(f"= {coefficient:.2f}")
        reasoning = " ".join(reasoning_parts)

        # 应用上下限
        coefficient = max(self.min_coefficient, min(self.max_coefficient, coefficient))

        return PoolingPrediction(
            library_id=lib.origrec,
            predicted_coefficient=coefficient,
            reasoning=reasoning,
            confidence=1.0,
            source="Rule",
            quality_factor=quality_factor,
            size_factor=size_factor,
            type_efficiency_factor=type_factor
        )
        
    async def predict_coefficients(self, libraries: List[EnhancedLibraryInfo]) -> Dict[str, PoolingPrediction]:
        """
        预测一组文库的Pooling系数（基于质量和数据量的动态策略）
        
        Args:
            libraries: 文库列表
            
        Returns:
            Dict[str, PoolingPrediction]: {lib_id: prediction}
        """
        logger.info(f"开始Pooling系数预测：共{len(libraries)}个文库")
        
        predictions = {}
        filtered_count = 0
        
        for lib in libraries:
            prediction = self.calculate_coefficient(lib)
            predictions[lib.origrec] = prediction
            
            if prediction.is_filtered:
                filtered_count += 1
        
        # 统计信息
        valid_predictions = [p for p in predictions.values() if not p.is_filtered]
        if valid_predictions:
            avg_coeff = sum(p.predicted_coefficient for p in valid_predictions) / len(valid_predictions)
            logger.info(
                f"Pooling系数预测完成："
                f"总数{len(libraries)}, "
                f"有效{len(valid_predictions)}, "
                f"过滤{filtered_count}, "
                f"平均系数{avg_coeff:.3f}"
            )
        else:
            logger.warning(f"Pooling系数预测完成：所有{len(libraries)}个文库均被过滤")
        
        return predictions

