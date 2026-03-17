"""
排机规则检测器

基于《排机规则文档.md》实现的24条关键排机规则自动检测。
用于多任务学习中的规则违反检测任务。

规则分类：
- 规则0-14：文库对级别规则（可在文库对层面检测）
- 规则15-24：Lane级别规则（需要Lane整体数据）

创建时间：2025-12-05 14:21:32
更新时间：2025-12-30 13:59:15

变更记录：
- 2025-12-24: 补充分组21-26的文库类型定义和占比系数
"""

from typing import Dict, List, Tuple, Optional, Set

import pandas as pd
import numpy as np
from loguru import logger


# ========== 碱基不均衡文库分组定义 ==========

# 分组1-26：同类型排整条Lane（不可与其他碱基不均衡文库同Lane）
# 参考文档：docs/排机规则文档.md#碱基不均衡文库类型完整清单
IMBALANCE_GROUPS = {
    1: ['ATAC-seq文库'],
    2: ['CUT Tag文库（动物组织）'],
    3: ['CUT Tag文库（细胞）'],
    4: ['EM-Seq文库'],
    5: ['Methylation文库', 'RRBS文库', 'RNA甲基化文库', 'RNA甲基化文库-Input', 
        'RNA甲基化文库-IP', '羟甲基化文库', 'Ribo-seq文库'],
    6: ['small RNA文库', 'UMI smallRNA文库', '外泌体small RNA'],
    7: ['单细胞文库', 'circRNA'],
    8: ['动植物简化基因组文库(GBS亲代)'],
    9: ['动植物简化基因组文库(GBS子代)'],
    10: ['客户-ATAC-seq文库'],
    11: ['客户-CUT-Tag文库'],
    12: ['客户-Methylation文库', '客户-RRBS文库'],
    13: ['客户-NanoString DSP文库', 'DSP空间转录组文库'],
    14: ['客户-PCR产物', 'PCR-free (PCR产物单建)'],
    15: ['客户-small RNA文库'],
    16: ['客户-单细胞文库', '客户-BD单细胞WTA文库', 'BD单细胞WTA文库'],
    17: ['客户-简化基因组', '客户-GBS(康奈尔GBS文库)', 'GBS(康奈尔GBS文库)',
         '客户-RAD文库 (单)', '客户-RAD文库 (混)', 'RAD文库 (单)', 'RAD文库 (混)'],
    18: ['客户-其他碱基不均衡文库', 'RAD-seq文库(单建)', 'RAD-seq文库(混建)'],
    19: ['微量 (Methylation文库)', '微量RNA甲基化文库', '微量甲基化',
         '低起始量RNA甲基化文库-Input(人)', '低起始量RNA甲基化文库-Input(鼠)',
         '低起始量RNA甲基化文库-IP(人)', '低起始量RNA甲基化文库-IP(鼠)', '外泌体lncRNA文库'],
    20: ['客户-扩增子文库'],
    # 分组21-26：2024.11.21补充
    21: ['10X全基因组文库'],
    22: ['客户-10X全基因组文库'],
    23: ['10xATAC-seq文库', '10x ATAC文库', '10XATAC文库'],
    24: ['客户-10X ATAC文库', '客户-10X ATAC (Multiome)文库'],
    25: ['10x HD Visium空间转录组文库(新)', '10x HD Visium空间转录组文库'],
    26: ['10XFixed RNA文库', '10X转录组-FixedRNA文库', '10x膜蛋白文库'],
}

# 分组27（3'端）：可同组混排
GROUP_27_TYPES = [
    '墨卓转录组-3端文库', '10X转录组-3\'文库', '10X Visium空间转录组文库',
    '10X转录组-3\'膜蛋白文库', '客户-10X 3 单细胞转录组文库',
    '10X Visium FFPEV2空间转录组文库(V2)', '客户文库-10X Visium 文库',
    '客户-10X 3 Feature Barcode文库', '客户-10X Visium空间转录组文库',
    '客户-10X Visium FFPEV2空间转录组文库(V2)', '客户-10X Feature Barcode文库',
    'BD单细胞WTA文库'
]

# 分组28（5'端）：可同组混排
GROUP_28_TYPES = [
    '10X转录组-5\'文库', '10X转录组V(D)J-BCR文库', '10X转录组V(D)J-TCR文库',
    '客户-10X VDJ文库', '客户-10X转录组V(D)J-BCR文库', '客户-10X转录组V(D)J-TCR文库',
    '客户-10X转录组文库', '10X转录组-5\'膜蛋白文库', '客户-10X 5 单细胞转录组文库',
    '客户-10X 5 Feature Barcode文库', '客户-10X Flex文库'
]

# 分组29：分组27+28混排
GROUP_29_TYPES = GROUP_27_TYPES + GROUP_28_TYPES

# 碱基不均衡分组数据量占比配置（Nova X-25B）
# 含义：该分组文库最多只能占Lane容量的该比例，剩余比例需用碱基均衡文库填充
# 例如：分组1（ATAC-seq）占比0.8，意味着ATAC-seq最多占Lane的80%，剩余20%需碱基均衡文库
# 参考文档：docs/排机规则文档.md#天津科技服务实验室-nova-x-25b机器
IMBALANCE_GROUP_RATIO = {
    1: 0.8,   # ATAC-seq文库
    2: 0.8,   # CUT Tag文库（动物组织）
    3: 0.8,   # CUT Tag文库（细胞）
    4: 0.99,  # EM-Seq文库
    5: 0.99,  # Methylation文库系列
    6: 1.0,   # small RNA文库系列
    7: 0.85,  # 单细胞文库
    8: 0.8,   # GBS亲代
    9: 0.8,   # GBS子代
    10: 0.8,  # 客户-ATAC-seq文库
    11: 0.8,  # 客户-CUT-Tag文库
    12: 0.99, # 客户-Methylation文库系列
    13: 0.8,  # 客户-NanoString DSP文库
    14: 0.6,  # 客户-PCR产物
    15: 1.0,  # 客户-small RNA文库
    16: 0.85, # 客户-单细胞文库系列
    17: 0.8,  # 客户-简化基因组系列
    18: 0.8,  # 客户-其他碱基不均衡文库
    19: 0.99, # 微量/低起始量Methylation系列
    20: 0.65, # 客户-扩增子文库
    # 分组21-26（2024.11.21补充）
    21: 1.0,  # 10X全基因组文库
    22: 1.0,  # 客户-10X全基因组文库
    23: 0.8,  # 10xATAC-seq文库
    24: 0.8,  # 客户-10X ATAC文库
    25: 1.0,  # 10x HD Visium空间转录组文库(新)
    26: 0.8,  # 10XFixed RNA文库
    # 分组27-29（可混排）
    27: 1.0,  # 墨卓转录组-3端文库等（3'端系列）
    28: 1.0,  # 10X转录组-5'文库等（5'端系列）
    29: 1.0,  # 分组27+28混排
}

# 特殊文库总量限制（碱基不均衡与均衡混排时）
SPECIAL_LIBRARY_TOTAL = {
    ('Nova X-25B', '25B'): 240,
    ('Nova X-10B', '10B'): 150,
    ('NovaSeq X Plus', '25B'): 240,
    ('Novaseq', 'S4 XP'): 400,
    ('Novaseq', 'S4'): 400,
    ('Novaseq', 'S2'): 500,
    ('Novaseq', 'S1'): 400,
    ('Novaseq-T7', 'S4'): 175,
}


class RuleChecker:
    """排机规则检测器
    
    检测两个文库是否违反15条关键排机规则。
    """
    
    def __init__(self, lane_capacity_config: Optional[Dict] = None):
        """初始化规则检测器
        
        Args:
            lane_capacity_config: Lane容量配置字典
                格式：{(machine_type, process_code): capacity}
                例如：{('Nova X-25B', 1595): 975}
        """
        self.rule_definitions = {
            0: ("机器类型一致性", "硬约束"),
            1: ("工序编码一致性", "硬约束"),
            2: ("Index序列冲突", "硬约束"),
            3: ("Peak Size范围", "红线规则"),
            4: ("10碱基Index占比", "固定规则"),
            5: ("单端Index占比", "固定规则"),
            6: ("客户诺禾占比", "红线规则"),
            7: ("碱基不均衡冲突", "固定规则"),
            8: ("优先级冲突", "可变规则"),
            9: ("产线标识", "固定规则"),
            10: ("测序策略一致性", "硬约束"),
            11: ("双端Index", "固定规则"),
            12: ("库检结果", "可变规则"),
            13: ("数据量级别", "可变规则"),
            14: ("特殊类型排斥", "固定规则"),
            15: ("Lane数据量上限", "硬约束"),
        }
        
        # Lane容量配置（根据机器类型和工序编码）
        self.lane_capacity_config = lane_capacity_config or self._default_capacity_config()
    
    def _default_capacity_config(self) -> Dict:
        """默认Lane容量配置
        
        注意：这些值会定期更新（约半月一次），每次增加10-15G
        """
        return {
            # Nova X系列 - 工序1595
            ('Nova X-25B', 1595): 975,
            ('Nova X-10B', 1595): 380,
            ('NovaSeq X Plus', 1595): 975,
            
            # Novaseq - 工序405（临检/YC有不同配置，这里用通用值）
            ('Novaseq', 405): 880,
            
            # Novaseq-T7 - 工序876
            ('Novaseq-T7', 876): 1670,
            ('T7', 876): 1670,
            
            # T7-C4 - 工序1770
            ('T7-C4', 1770): 4500,
            ('T7-Stereo-FFPE', 1770): 5000,
            ('T7-Stereo', 1770): 4000,
            
            # SURFSEQ系列 - 工序1832
            ('SURFSEQ-5000', 1832): 1200,
            ('ZM SURFSeq5000', 1832): 1200,
            ('SURFSEQ-Q', 1832): 750,
            
            # 其他工序
            ('Novaseq', 418): 435,  # SE50
            ('Novaseq', 419): 410,  # PE250
            ('Novaseq', 426): 400,  # PE50
        }
    
    def check_all_rules(self, lib1: Dict, lib2: Dict) -> List[int]:
        """检查所有15条规则
        
        Args:
            lib1: 第一个文库的特征字典
            lib2: 第二个文库的特征字典
        
        Returns:
            violations: 长度15的列表，1表示违反，0表示遵守
        """
        violations = [0] * 15
        
        violations[0] = self.check_machine_type(lib1, lib2)
        violations[1] = self.check_process_code(lib1, lib2)
        violations[2] = self.check_index_conflict(lib1, lib2)
        violations[3] = self.check_peak_size(lib1, lib2)
        violations[4] = self.check_10base_index_ratio(lib1, lib2)
        violations[5] = self.check_single_index_ratio(lib1, lib2)
        violations[6] = self.check_customer_novogene_ratio(lib1, lib2)
        violations[7] = self.check_base_imbalance_conflict(lib1, lib2)
        violations[8] = self.check_priority_conflict(lib1, lib2)
        violations[9] = self.check_production_line(lib1, lib2)
        violations[10] = self.check_sequencing_strategy(lib1, lib2)
        violations[11] = self.check_dual_index_ratio(lib1, lib2)
        violations[12] = self.check_quality_result(lib1, lib2)
        violations[13] = self.check_data_volume_level(lib1, lib2)
        violations[14] = self.check_special_library_conflict(lib1, lib2)
        
        return violations
    
    # ========== 规则0：机器类型一致性 ==========
    def check_machine_type(self, lib1: Dict, lib2: Dict) -> int:
        """规则0：机器类型必须一致
        
        同一Lane必须使用相同机器类型（硬约束）
        """
        machine1 = str(lib1.get('机器类型', '')).strip()
        machine2 = str(lib2.get('机器类型', '')).strip()
        
        if not machine1 or not machine2:
            return 0  # 数据缺失，不判定为违反
        
        return int(machine1 != machine2)
    
    # ========== 规则1：工序编码一致性 ==========
    def check_process_code(self, lib1: Dict, lib2: Dict) -> int:
        """规则1：工序编码必须一致
        
        同一Lane必须使用相同工序（硬约束）
        """
        code1 = lib1.get('工序编码', None)
        code2 = lib2.get('工序编码', None)
        
        if code1 is None or code2 is None:
            return 0
        
        # 转换为字符串比较（处理数值和字符串混合情况）
        return int(str(code1) != str(code2))
    
    # ========== 规则2：Index序列冲突检测 ==========
    def check_index_conflict(self, lib1: Dict, lib2: Dict) -> int:
        """规则2：Index P7端重复位数不能超限
        
        根据碱基长度确定重复阈值：
        - 6碱基：重复不超过4位
        - 7碱基：重复不超过5位
        - 8碱基：重复不超过6位
        - >8碱基：重复不超过7位
        """
        index1 = lib1.get('Index序列', '')
        index2 = lib2.get('Index序列', '')
        
        if not index1 or not index2 or pd.isna(index1) or pd.isna(index2):
            return 0
        
        # 提取P7端（分号前的部分）
        p7_1 = str(index1).split(';')[0].strip()
        p7_2 = str(index2).split(';')[0].strip()
        
        if not p7_1 or not p7_2:
            return 0
        
        # 计算重复碱基数
        min_len = min(len(p7_1), len(p7_2))
        overlap = sum(a == b for a, b in zip(p7_1[:min_len], p7_2[:min_len]))
        
        # 根据碱基长度确定阈值
        len1, len2 = len(p7_1), len(p7_2)
        if len1 <= 6 or len2 <= 6:
            threshold = 4
        elif len1 == 7 or len2 == 7:
            threshold = 5
        elif len1 == 8 or len2 == 8:
            threshold = 6
        else:
            threshold = 7
        
        return int(overlap > threshold)
    
    # ========== 规则3：Peak Size范围限制 ==========
    def check_peak_size(self, lib1: Dict, lib2: Dict) -> int:
        """规则3：Peak Size差异≤150bp
        
        注意：完整规则需要Lane级别统计（数据量≥75%的豁免条件）
        这里只做文库对层面的简化检测
        """
        peak1 = lib1.get('PeakSize', None)
        peak2 = lib2.get('PeakSize', None)
        
        # 处理缺失值和异常值
        if peak1 is None or peak2 is None:
            return 0
        if pd.isna(peak1) or pd.isna(peak2):
            return 0
        if peak1 == -999 or peak2 == -999:  # 常见的缺失值标记
            return 0
        
        try:
            diff = abs(float(peak1) - float(peak2))
            return int(diff > 150)
        except (ValueError, TypeError):
            return 0
    
    # ========== 规则4：10碱基Index文库占比 ==========
    def check_10base_index_ratio(self, lib1: Dict, lib2: Dict) -> int:
        """规则4：10碱基与非10碱基混排时占比>40%
        
        注意：这是占比规则，需要Lane级别统计才能准确判断
        文库对层面返回0，由Lane级别验证
        """
        # 占比规则无法在文库对层面准确判断，返回0
        # 真正的检测在 check_10base_index_ratio_lane() 方法中
        return 0
    
    # ========== 规则5：单端Index文库占比 ==========
    def check_single_index_ratio(self, lib1: Dict, lib2: Dict) -> int:
        """规则5：单端与双端混排时，单端占比<30%
        
        注意：这是占比规则，需要Lane级别统计才能准确判断
        文库对层面返回0，由Lane级别验证
        """
        # 占比规则无法在文库对层面准确判断，返回0
        # 真正的检测在 check_single_index_ratio_lane() 方法中
        return 0
    
    # ========== 规则6：客户/诺禾文库混排占比 ==========
    def check_customer_novogene_ratio(self, lib1: Dict, lib2: Dict) -> int:
        """规则6：存在诺禾文库时，客户文库占比≤50%
        
        注意：这是占比规则，需要Lane级别统计才能准确判断
        文库对层面返回0，由Lane级别验证
        """
        # 占比规则无法在文库对层面准确判断，返回0
        # 真正的检测在 check_customer_ratio_lane() 方法中
        return 0
    
    # ========== 规则7：碱基不均衡文库类型冲突 ==========
    def check_base_imbalance_conflict(self, lib1: Dict, lib2: Dict) -> int:
        """规则7：特定碱基不均衡文库类型不可混排
        
        冲突对：
        - 10x HD Visium ↔ 10X转录组-3'
        - 10x HD Visium ↔ 墨卓转录组-3端
        - 10X ATAC ↔ cellranger拆分方式
        """
        type1 = str(lib1.get('样本类型', '')).lower()
        type2 = str(lib2.get('样本类型', '')).lower()
        
        if not type1 or not type2:
            return 0
        
        # 定义冲突模式
        conflict_patterns = [
            ('10x hd visium', '10x转录组-3'),
            ('10x hd visium', '墨卓转录组-3'),
            ('10x atac', 'cellranger'),
        ]
        
        for pattern1, pattern2 in conflict_patterns:
            # 双向检查
            if (pattern1 in type1 and pattern2 in type2) or \
               (pattern2 in type1 and pattern1 in type2):
                return 1
        
        return 0
    
    # ========== 规则8：数据类型优先级冲突 ==========
    def check_priority_conflict(self, lib1: Dict, lib2: Dict) -> int:
        """规则8：不同优先级文库混排受限
        
        优先级：临检(医学) > YC > 其他
        临检与"其他"类型（非YC）混排受限
        """
        priority_map = {
            '临检': 1,
            '医学': 1,
            'medical': 1,
            'clinical': 1,
            'yc': 2,
            'yc文库': 2,
        }
        
        type1 = str(lib1.get('数据类型', '')).strip().lower()
        type2 = str(lib2.get('数据类型', '')).strip().lower()
        
        p1 = priority_map.get(type1, 3)
        p2 = priority_map.get(type2, 3)
        
        # 临检(优先级1)和其他类型（优先级3）混排需要特殊处理
        violation = (p1 == 1 and p2 > 2) or (p2 == 1 and p1 > 2)
        
        return int(violation)
    
    # ========== 规则9：产线标识匹配 ==========
    def check_production_line(self, lib1: Dict, lib2: Dict) -> int:
        """规则9：S/Z/ZS产线标识匹配规则
        
        S（手工管式）、Z（自动板式）、ZS（混合）
        注意：当前简化处理，实际规则较复杂
        """
        line1 = str(lib1.get('产线标识', '')).strip()
        line2 = str(lib2.get('产线标识', '')).strip()
        
        if not line1 or not line2:
            return 0
        
        # S/Z/ZS可以混排，这里不做严格限制
        # 实际业务中可能有更复杂的规则
        return 0
    
    # ========== 规则10：测序策略一致性 ==========
    def check_sequencing_strategy(self, lib1: Dict, lib2: Dict) -> int:
        """规则10：PE150/SE50等测序策略必须一致
        
        硬约束：同一Lane的测序策略必须相同
        """
        strategy1 = str(lib1.get('测序策略', '')).strip()
        strategy2 = str(lib2.get('测序策略', '')).strip()
        
        if not strategy1 or not strategy2:
            return 0
        
        return int(strategy1 != strategy2)
    
    # ========== 规则11：双端Index一致性 ==========
    def check_dual_index_ratio(self, lib1: Dict, lib2: Dict) -> int:
        """规则11：单双端Index一致性（与规则5类似）"""
        return self.check_single_index_ratio(lib1, lib2)
    
    # ========== 规则12：库检结果兼容性 ==========
    def check_quality_result(self, lib1: Dict, lib2: Dict) -> int:
        """规则12：风险建库与正常建库混排规则
        
        简化处理：存在风险建库时标记
        """
        result1 = str(lib1.get('库检综合结果', '')).strip()
        result2 = str(lib2.get('库检综合结果', '')).strip()
        
        if not result1 or not result2:
            return 0
        
        # 风险建库混排需要特殊处理
        is_risk = ('风险' in result1) or ('风险' in result2)
        
        return int(is_risk)
    
    # ========== 规则13：数据量级别匹配 ==========
    def check_data_volume_level(self, lib1: Dict, lib2: Dict) -> int:
        """规则13：大数据量(>70G)与小数据量混排规则
        
        注意：完整规则需要考虑数据类型和优先级
        这里只做简化检测
        """
        try:
            vol1 = float(lib1.get('合同数据量_文库', 0))
            vol2 = float(lib2.get('合同数据量_文库', 0))
            
            is_large_1 = (vol1 > 70)
            is_large_2 = (vol2 > 70)
            
            # 简化规则：当前不做严格限制
            # 实际需要结合数据类型和优先级判断
            return 0
        except (ValueError, TypeError):
            return 0
    
    # ========== 规则14：特殊文库类型排斥 ==========
    def check_special_library_conflict(self, lib1: Dict, lib2: Dict) -> int:
        """规则14：10X ATAC与cellranger不可同Run
        
        检查特殊文库类型的排斥关系
        """
        type1 = str(lib1.get('样本类型', '')).lower()
        type2 = str(lib2.get('样本类型', '')).lower()
        
        if not type1 or not type2:
            return 0
        
        # 10X ATAC与cellranger冲突
        conflict = ('10x atac' in type1 and 'cellranger' in type2) or \
                  ('10x atac' in type2 and 'cellranger' in type1)
        
        return int(conflict)
    
    # ========== 碱基不均衡分组辅助方法 ==========
    def get_imbalance_group(self, sample_type: str) -> Optional[int]:
        """获取文库的碱基不均衡分组号
        
        Returns:
            分组号（1-26, 27, 28, 29）或 None（非碱基不均衡文库）
            - 分组1-20: 原有分组
            - 分组21-26: 2024.11.21补充（10X全基因组、10xATAC、HD Visium等）
            - 分组27-29: 可混排分组（3'端/5'端转录组）
        """
        if not sample_type:
            return None
        
        sample_type = str(sample_type).strip()
        
        # 检查分组27（3'端）
        if sample_type in GROUP_27_TYPES:
            return 27
        
        # 检查分组28（5'端）
        if sample_type in GROUP_28_TYPES:
            return 28
        
        # 检查分组1-20
        for group_id, types in IMBALANCE_GROUPS.items():
            if sample_type in types:
                return group_id
        
        return None
    
    def is_imbalance_library(self, lib: Dict) -> bool:
        """判断是否为碱基不均衡文库"""
        sample_type = lib.get('样本类型', '')
        return self.get_imbalance_group(sample_type) is not None
    
    # ========== 规则16：碱基不均衡分组冲突（文库对级别简化版） ==========
    def check_imbalance_group_conflict(self, lib1: Dict, lib2: Dict) -> int:
        """检测碱基不均衡分组冲突
        
        返回1表示不同分组的碱基不均衡文库，不可混排
        返回0表示可能兼容（同组或非碱基不均衡）
        """
        sample_type1 = lib1.get('样本类型', '')
        sample_type2 = lib2.get('样本类型', '')
        
        group1 = self.get_imbalance_group(sample_type1)
        group2 = self.get_imbalance_group(sample_type2)
        
        # 都不是碱基不均衡文库
        if group1 is None and group2 is None:
            return 0
        
        # 一个是碱基不均衡，一个不是（可混排，但需检查占比，Lane级别）
        if group1 is None or group2 is None:
            return 0
        
        # 都是碱基不均衡文库
        # 同组可混排
        if group1 == group2:
            return 0
        
        # 检查分组27/28/29的特殊混排规则
        special_groups = {27, 28, 29}
        if group1 in special_groups and group2 in special_groups:
            return 0  # 27/28/29可以互相混排（需Lane级验证占比）
        
        # 不同分组不可混排
        return 1
    
    # ========== 规则17：碱基不均衡占比限制（Lane级别） ==========
    def check_imbalance_ratio(self, lane_libraries: List[Dict], 
                              ratio_limit: float = 0.4) -> int:
        """检查碱基不均衡文库占比是否超限
        
        Args:
            lane_libraries: Lane中所有文库
            ratio_limit: 占比限制，默认40%
        
        Returns:
            1表示违反（占比>=限制），0表示未违反
        """
        if not lane_libraries:
            return 0
        
        total_data = 0.0
        imbalance_data = 0.0
        
        for lib in lane_libraries:
            contract_vol = lib.get('合同数据量_文库', 0)
            if contract_vol and not pd.isna(contract_vol):
                try:
                    vol = float(contract_vol)
                    total_data += vol
                    if self.is_imbalance_library(lib):
                        imbalance_data += vol
                except (ValueError, TypeError):
                    continue
        
        if total_data == 0:
            return 0
        
        ratio = imbalance_data / total_data
        return int(ratio >= ratio_limit)
    
    # ========== 规则18：特殊文库总量限制（Lane级别） ==========
    def check_special_library_total(self, lane_libraries: List[Dict],
                                    machine_type: str,
                                    load_method: str) -> int:
        """检查碱基不均衡文库总量是否超限
        
        Returns:
            1表示违反，0表示未违反
        """
        # 获取限制值
        key = (machine_type, load_method)
        total_limit = SPECIAL_LIBRARY_TOTAL.get(key, None)
        
        if total_limit is None:
            return 0  # 无配置不限制
        
        # 计算碱基不均衡文库总量
        imbalance_total = 0.0
        for lib in lane_libraries:
            if self.is_imbalance_library(lib):
                contract_vol = lib.get('合同数据量_文库', 0)
                if contract_vol and not pd.isna(contract_vol):
                    try:
                        imbalance_total += float(contract_vol)
                    except (ValueError, TypeError):
                        continue
        
        return int(imbalance_total > total_limit)
    
    # ========== 规则19：文库拆分规则（预处理） ==========
    def check_need_split(self, lib: Dict) -> bool:
        """检查文库是否需要拆分
        
        自动线阈值：80G
        手工线阈值：100G
        """
        contract_vol = lib.get('合同数据量_文库', 0)
        production_line = lib.get('产线标识', 'S')
        
        if not contract_vol or pd.isna(contract_vol):
            return False
        
        try:
            vol = float(contract_vol)
            threshold = 80 if production_line == 'Z' else 100
            return vol > threshold
        except (ValueError, TypeError):
            return False
    
    # ========== 规则20：版号一致性（软约束） ==========
    def check_version_match(self, lib1: Dict, lib2: Dict) -> float:
        """检查版号是否一致（软约束）
        
        Returns:
            1.0表示版号相同，0.0表示不同或无版号
        """
        version1 = lib1.get('版号', '')
        version2 = lib2.get('版号', '')
        
        if not version1 or not version2:
            return 0.0
        
        return 1.0 if str(version1) == str(version2) else 0.0
    
    # ========== 规则21：甲基化文库特殊规则（Lane级别） ==========
    def check_methylation_special_rule(self, lane_libraries: List[Dict],
                                       machine_type: str) -> Dict:
        """甲基化文库特殊规则检测
        
        Returns:
            配置调整字典或空字典
        """
        methylation_types = [
            'Methylation文库', '客户-Methylation文库', 
            '微量 (Methylation文库)', 'RRBS文库', '客户-RRBS文库'
        ]
        
        methylation_data = 0.0
        has_methylation = False
        
        for lib in lane_libraries:
            sample_type = lib.get('样本类型', '')
            if sample_type in methylation_types:
                has_methylation = True
                contract_vol = lib.get('合同数据量_文库', 0)
                if contract_vol and not pd.isna(contract_vol):
                    try:
                        methylation_data += float(contract_vol)
                    except (ValueError, TypeError):
                        pass
        
        if has_methylation and 'T7' in machine_type:
            if methylation_data <= 160:
                # 需要调整Lane数据量为1600G
                return {'adjust_lane_capacity': 1600, 'add_phix': 0.0833}
        
        return {}
    
    # ========== 规则22：T7-C4 oligo限制（Lane级别） ==========
    def check_t7c4_oligo_limit(self, lane_libraries: List[Dict],
                               process_code: int) -> int:
        """T7-C4工序oligo数据量限制
        
        cDNA与oligo混排时，oligo上限 < 700G
        """
        if process_code != 1770:  # T7-C4 PE100
            return 0
        
        oligo_types = ['oligo文库']  # 具体类型可能需要扩展
        
        oligo_data = 0.0
        for lib in lane_libraries:
            sample_type = lib.get('样本类型', '')
            if sample_type in oligo_types:
                contract_vol = lib.get('合同数据量_文库', 0)
                if contract_vol and not pd.isna(contract_vol):
                    try:
                        oligo_data += float(contract_vol)
                    except (ValueError, TypeError):
                        pass
        
        return int(oligo_data >= 700)
    
    # ========== 规则23：Lane级别分组文库类型数量检查 ==========
    def check_imbalance_type_count(self, lane_libraries: List[Dict],
                                   max_types: int = 3) -> int:
        """检查Lane中碱基不均衡文库类型数量
        
        混排时文库类型不超过3种
        """
        imbalance_types = set()
        
        for lib in lane_libraries:
            sample_type = lib.get('样本类型', '')
            if self.is_imbalance_library(lib):
                imbalance_types.add(sample_type)
        
        return int(len(imbalance_types) > max_types)
    
    # ========== 规则24：分组29占比规则（Lane级别） ==========
    def check_group29_ratio(self, lane_libraries: List[Dict]) -> int:
        """检查分组27和28混排时，分组27占比是否≤20%
        
        只在分组27和分组28混排时检查
        """
        group27_data = 0.0
        group28_data = 0.0
        total_data = 0.0
        
        has_group27 = False
        has_group28 = False
        
        for lib in lane_libraries:
            sample_type = lib.get('样本类型', '')
            group = self.get_imbalance_group(sample_type)
            contract_vol = lib.get('合同数据量_文库', 0)
            
            if contract_vol and not pd.isna(contract_vol):
                try:
                    vol = float(contract_vol)
                    
                    if group == 27:
                        group27_data += vol
                        has_group27 = True
                    elif group == 28:
                        group28_data += vol
                        has_group28 = True
                    
                    if group in {27, 28}:
                        total_data += vol
                except (ValueError, TypeError):
                    pass
        
        # 只有同时存在27和28时才检查
        if has_group27 and has_group28 and total_data > 0:
            ratio = group27_data / total_data
            return int(ratio > 0.2)  # 分组27占比超过20%违反
        
        return 0
    
    # ========== Lane级别占比规则 ==========
    
    def check_10base_index_ratio_lane(self, lane_libraries: List[Dict]) -> int:
        """规则4：10碱基Index占比检测（Lane级别）
        
        规则：
        - 全是10碱基：不限制
        - 全是非10碱基：不限制
        - 混合时：10碱基占比必须 > 40%（即非10碱基 < 60%）
        
        Returns:
            1表示违反，0表示未违反
        """
        total_data = 0.0
        base10_data = 0.0
        
        has_10base = False
        has_non10base = False
        
        for lib in lane_libraries:
            contract_vol = lib.get('合同数据量_文库', 0)
            if not contract_vol or pd.isna(contract_vol):
                continue
            
            try:
                vol = float(contract_vol)
                total_data += vol
                
                base_num = lib.get('index碱基数目', 0)
                if base_num and not pd.isna(base_num):
                    if int(base_num) == 10:
                        base10_data += vol
                        has_10base = True
                    else:
                        has_non10base = True
            except (ValueError, TypeError):
                continue
        
        # 不是混合情况，不限制
        if not (has_10base and has_non10base):
            return 0
        
        # 混合情况，检查10碱基占比
        if total_data == 0:
            return 0
        
        ratio = base10_data / total_data
        return int(ratio <= 0.4)  # 10碱基占比≤40%违反
    
    def check_single_index_ratio_lane(self, lane_libraries: List[Dict]) -> int:
        """规则5：单端Index占比检测（Lane级别）
        
        规则：
        - 全是单端：不限制
        - 全是双端：不限制
        - 混合时：单端占比必须 < 30%
        
        Returns:
            1表示违反，0表示未违反
        """
        total_data = 0.0
        single_data = 0.0
        
        has_single = False
        has_dual = False
        
        for lib in lane_libraries:
            contract_vol = lib.get('合同数据量_文库', 0)
            if not contract_vol or pd.isna(contract_vol):
                continue
            
            try:
                vol = float(contract_vol)
                total_data += vol
                
                is_dual = str(lib.get('是否双端index测序', '')).strip() == '是'
                if is_dual:
                    has_dual = True
                else:
                    single_data += vol
                    has_single = True
            except (ValueError, TypeError):
                continue
        
        # 不是混合情况，不限制
        if not (has_single and has_dual):
            return 0
        
        # 混合情况，检查单端占比
        if total_data == 0:
            return 0
        
        ratio = single_data / total_data
        return int(ratio >= 0.3)  # 单端占比≥30%违反
    
    def check_customer_ratio_lane(self, lane_libraries: List[Dict]) -> int:
        """规则6：客户/诺禾混排占比检测（Lane级别）
        
        规则：
        - 全是客户文库：不限制
        - 全是诺禾文库：不限制
        - 混合时：客户文库占比必须 ≤ 50%
        
        Returns:
            1表示违反，0表示未违反
        """
        total_data = 0.0
        customer_data = 0.0
        
        has_customer = False
        has_novogene = False
        
        for lib in lane_libraries:
            contract_vol = lib.get('合同数据量_文库', 0)
            if not contract_vol or pd.isna(contract_vol):
                continue
            
            try:
                vol = float(contract_vol)
                total_data += vol
                
                is_customer = str(lib.get('是否是客户文库', '')).strip() == '是'
                if is_customer:
                    customer_data += vol
                    has_customer = True
                else:
                    has_novogene = True
            except (ValueError, TypeError):
                continue
        
        # 不是混合情况，不限制
        if not (has_customer and has_novogene):
            return 0
        
        # 混合情况，检查客户文库占比
        if total_data == 0:
            return 0
        
        ratio = customer_data / total_data
        return int(ratio > 0.5)  # 客户占比>50%违反
    
    def check_all_lane_ratio_rules(self, lane_libraries: List[Dict]) -> Dict:
        """检查所有Lane级别占比规则
        
        Returns:
            包含所有占比规则检测结果的字典
        """
        results = {
            'rule4_10base_ratio': self.check_10base_index_ratio_lane(lane_libraries),
            'rule5_single_index_ratio': self.check_single_index_ratio_lane(lane_libraries),
            'rule6_customer_ratio': self.check_customer_ratio_lane(lane_libraries),
            'rule17_imbalance_ratio': self.check_imbalance_ratio(lane_libraries),
            'rule24_group29_ratio': self.check_group29_ratio(lane_libraries),
        }
        
        # 计算总违反数
        violation_count = sum(results.values())
        results['total_violations'] = violation_count
        results['all_passed'] = (violation_count == 0)
        
        return results
    
    # ========== 综合Lane级别规则检查 ==========
    def check_all_lane_rules(self, lane_libraries: List[Dict],
                             machine_type: str,
                             process_code: int,
                             load_method: str = '25B',
                             priority: str = '其他') -> Dict:
        """检查所有Lane级别规则（包括占比规则）
        
        Returns:
            包含所有Lane级别规则检查结果的字典
        """
        # 占比规则（规则4,5,6,17,24）
        results = {
            'rule4_10base_ratio': self.check_10base_index_ratio_lane(lane_libraries),
            'rule5_single_index_ratio': self.check_single_index_ratio_lane(lane_libraries),
            'rule6_customer_ratio': self.check_customer_ratio_lane(lane_libraries),
            'rule17_imbalance_ratio': self.check_imbalance_ratio(lane_libraries),
            'rule24_group29_ratio': self.check_group29_ratio(lane_libraries),
        }
        
        # 容量和数量规则（规则15,18,23）
        results['rule15_capacity'] = self.check_lane_capacity_limit(
            lane_libraries, machine_type, process_code, priority
        )
        results['rule18_special_total'] = self.check_special_library_total(
            lane_libraries, machine_type, load_method
        )
        results['rule23_type_count'] = self.check_imbalance_type_count(lane_libraries)
        
        # 特殊工序规则（规则21,22）
        results['rule21_methylation'] = self.check_methylation_special_rule(
            lane_libraries, machine_type
        )
        results['rule22_t7c4_oligo'] = self.check_t7c4_oligo_limit(
            lane_libraries, process_code
        )
        
        # 计算容量使用情况
        results['capacity_usage'] = self.get_lane_capacity_usage(
            lane_libraries, machine_type, process_code
        )
        
        # 计算总违反数（不包括rule21_methylation，它返回配置调整而非违反标记）
        violation_count = sum([
            results['rule4_10base_ratio'],
            results['rule5_single_index_ratio'],
            results['rule6_customer_ratio'],
            results['rule15_capacity'],
            results['rule17_imbalance_ratio'],
            results['rule18_special_total'],
            results['rule22_t7c4_oligo'],
            results['rule23_type_count'],
            results['rule24_group29_ratio'],
        ])
        
        results['total_violations'] = violation_count
        results['is_valid'] = (violation_count == 0)
        
        return results
    
    # ========== 规则15：Lane数据量上限检查（Lane级别） ==========
    def check_lane_capacity_limit(self, lane_libraries: List[Dict], 
                                   machine_type: str, 
                                   process_code: int,
                                   priority: Optional[str] = None) -> int:
        """规则15：Lane总合同数据量不能超过上限
        
        注意：这是Lane级别的规则，不能在文库对层面检测
        
        Args:
            lane_libraries: Lane中所有文库的列表
            machine_type: 机器类型
            process_code: 工序编码
            priority: 优先级档位（临检/YC/其他），可选
        
        Returns:
            1表示违反（超过上限），0表示未违反
        """
        # 获取Lane容量配置
        capacity_key = (machine_type, process_code)
        lane_capacity = self.lane_capacity_config.get(capacity_key, None)
        
        if lane_capacity is None:
            # 配置中没有该机器/工序组合，无法判断
            return 0
        
        # 计算Lane总合同数据量
        total_contract = 0.0
        for lib in lane_libraries:
            contract_vol = lib.get('合同数据量_文库', 0)
            if contract_vol and not pd.isna(contract_vol):
                try:
                    total_contract += float(contract_vol)
                except (ValueError, TypeError):
                    continue
        
        # 检查是否超过上限（考虑±10G浮动范围）
        float_range = 10
        return int(total_contract > (lane_capacity + float_range))
    
    def check_can_add_to_lane(self, lane_libraries: List[Dict], 
                              new_library: Dict,
                              machine_type: str,
                              process_code: int,
                              priority: Optional[str] = None) -> bool:
        """检查新文库是否可以加入Lane（容量角度）
        
        Returns:
            True表示可以加入，False表示容量已满
        """
        # 获取Lane容量配置
        capacity_key = (machine_type, process_code)
        lane_capacity = self.lane_capacity_config.get(capacity_key, None)
        
        if lane_capacity is None:
            # 没有配置，默认允许
            return True
        
        # 计算当前总量
        current_total = 0.0
        for lib in lane_libraries:
            contract_vol = lib.get('合同数据量_文库', 0)
            if contract_vol and not pd.isna(contract_vol):
                try:
                    current_total += float(contract_vol)
                except (ValueError, TypeError):
                    continue
        
        # 计算加入新文库后的总量
        new_contract = new_library.get('合同数据量_文库', 0)
        if new_contract and not pd.isna(new_contract):
            try:
                new_total = current_total + float(new_contract)
            except (ValueError, TypeError):
                new_total = current_total
        else:
            new_total = current_total
        
        # 检查是否超限
        float_range = 10
        return new_total <= (lane_capacity + float_range)
    
    def get_lane_capacity_usage(self, lane_libraries: List[Dict],
                                machine_type: str,
                                process_code: int) -> Dict:
        """获取Lane容量使用情况
        
        Returns:
            字典包含：capacity, current_total, usage_rate, remaining
        """
        capacity_key = (machine_type, process_code)
        lane_capacity = self.lane_capacity_config.get(capacity_key, 0)
        
        # 计算当前总量
        current_total = 0.0
        for lib in lane_libraries:
            contract_vol = lib.get('合同数据量_文库', 0)
            if contract_vol and not pd.isna(contract_vol):
                try:
                    current_total += float(contract_vol)
                except (ValueError, TypeError):
                    continue
        
        usage_rate = current_total / lane_capacity if lane_capacity > 0 else 0
        remaining = lane_capacity - current_total if lane_capacity > 0 else 0
        
        return {
            'capacity': lane_capacity,
            'current_total': current_total,
            'usage_rate': usage_rate,
            'remaining': remaining,
            'is_full': (current_total > lane_capacity + 10)
        }
    
    def get_violation_summary(self, lib1: Dict, lib2: Dict) -> Dict:
        """获取规则违反的详细摘要
        
        Returns:
            摘要字典，包含：
            - violations: 15维违反向量
            - violated_count: 违反规则数量
            - violated_rules: 违反的规则列表（名称+类型）
        """
        violations = self.check_all_rules(lib1, lib2)
        violated_count = sum(violations)
        
        violated_rules = []
        for idx, is_violated in enumerate(violations):
            if is_violated:
                rule_name, rule_type = self.rule_definitions[idx]
                violated_rules.append({
                    'rule_id': idx,
                    'rule_name': rule_name,
                    'rule_type': rule_type
                })
        
        return {
            'violations': violations,
            'violated_count': violated_count,
            'violated_rules': violated_rules,
            'is_compatible': (violated_count == 0)
        }


def test_lane_capacity():
    """测试Lane容量检查"""
    checker = RuleChecker()
    
    # 测试用例：检查Lane容量
    lane_libraries = [
        {'合同数据量_文库': 50.0, 'SID_文库': '001'},
        {'合同数据量_文库': 45.0, 'SID_文库': '002'},
        {'合同数据量_文库': 38.0, 'SID_文库': '003'},
        # 当前总计：133G
    ]
    
    new_library = {'合同数据量_文库': 250.0, 'SID_文库': '004'}
    
    logger.info("=== Lane容量检查测试 ===")
    
    # 检查当前容量使用情况
    usage = checker.get_lane_capacity_usage(
        lane_libraries, 'Nova X-25B', 1595
    )
    logger.info("机器类型: Nova X-25B, 工序: 1595")
    logger.info("Lane容量上限: {}G", usage['capacity'])
    logger.info("当前使用: {:.2f}G", usage['current_total'])
    logger.info("使用率: {:.1%}", usage['usage_rate'])
    logger.info("剩余容量: {:.2f}G", usage['remaining'])
    logger.info("是否已满: {}", usage['is_full'])
    
    # 检查是否可以加入新文库
    can_add = checker.check_can_add_to_lane(
        lane_libraries, new_library, 'Nova X-25B', 1595
    )
    logger.info("新文库数据量: {}G", new_library['合同数据量_文库'])
    logger.info("加入后总量: {:.2f}G", usage['current_total'] + new_library['合同数据量_文库'])
    logger.info("是否可以加入: {}", "通过" if can_add else "不通过")
    
    # 测试超限情况
    logger.info("=== 测试超限Lane ===")
    large_lane = [
        {'合同数据量_文库': 300.0, 'SID_文库': '001'},
        {'合同数据量_文库': 300.0, 'SID_文库': '002'},
        {'合同数据量_文库': 300.0, 'SID_文库': '003'},
        {'合同数据量_文库': 100.0, 'SID_文库': '004'},
        # 总计：1000G（超过975G+10G）
    ]
    
    violation = checker.check_lane_capacity_limit(
        large_lane, 'Nova X-25B', 1595
    )
    usage2 = checker.get_lane_capacity_usage(
        large_lane, 'Nova X-25B', 1595
    )
    logger.info("Lane总量: {:.2f}G", usage2['current_total'])
    logger.info("容量上限: {}G (+/-10G)", usage2['capacity'])
    logger.info("是否违反规则15: {}", "[X] 违反" if violation else "[V] 未违反")


def test_imbalance_rules():
    """测试碱基不均衡规则"""
    checker = RuleChecker()
    
    logger.info("=== 碱基不均衡分组检测测试 ===")
    
    # 测试分组识别
    test_types = [
        'ATAC-seq文库',           # 分组1
        'Methylation文库',        # 分组5
        '10X转录组-3\'文库',       # 分组27
        '10X转录组-5\'文库',       # 分组28
        'VIP-真核普通转录组文库',  # 非碱基不均衡
    ]
    
    logger.info("分组识别测试：")
    for sample_type in test_types:
        group = checker.get_imbalance_group(sample_type)
        logger.info("  {}: 分组{}", sample_type, group if group else '无(碱基均衡)')
    logger.info("")
    
    # 测试分组冲突检测
    logger.info("分组冲突检测测试：")
    
    # 测试1：同分组（应该兼容）
    lib1 = {'样本类型': '10X转录组-3\'文库'}  # 分组27
    lib2 = {'样本类型': '10X Visium空间转录组文库'}  # 分组27
    conflict1 = checker.check_imbalance_group_conflict(lib1, lib2)
    logger.info("  分组27 vs 分组27: {}", "[X] 冲突" if conflict1 else "[V] 兼容")
    
    # 测试2：分组27和28（应该兼容，特殊规则）
    lib3 = {'样本类型': '10X转录组-5\'文库'}  # 分组28
    conflict2 = checker.check_imbalance_group_conflict(lib1, lib3)
    logger.info("  分组27 vs 分组28: {}", "[X] 冲突" if conflict2 else "[V] 兼容(可混排)")
    
    # 测试3：不同分组（应该冲突）
    lib4 = {'样本类型': 'ATAC-seq文库'}  # 分组1
    lib5 = {'样本类型': 'Methylation文库'}  # 分组5
    conflict3 = checker.check_imbalance_group_conflict(lib4, lib5)
    logger.info("  分组1 vs 分组5: {}", "[X] 冲突" if conflict3 else "[V] 兼容")
    
    # 测试4：碱基不均衡与均衡（文库对层面不冲突，需Lane级验证）
    lib6 = {'样本类型': 'VIP-真核普通转录组文库'}  # 非碱基不均衡
    conflict4 = checker.check_imbalance_group_conflict(lib4, lib6)
    logger.info("  碱基不均衡 vs 碱基均衡: {}", "[X] 冲突" if conflict4 else "[V] 可能兼容(需验证占比)")
    logger.info("")
    
    # 测试碱基不均衡占比
    logger.info("=== 碱基不均衡占比测试 ===")
    
    lane_mixed = [
        {'样本类型': 'ATAC-seq文库', '合同数据量_文库': 100},  # 碱基不均衡
        {'样本类型': 'VIP-真核普通转录组文库', '合同数据量_文库': 200},  # 碱基均衡
        {'样本类型': 'VIP-真核普通转录组文库', '合同数据量_文库': 200},  # 碱基均衡
        # 碱基不均衡占比: 100/500 = 20% < 40%
    ]
    
    ratio_violation = checker.check_imbalance_ratio(lane_mixed)
    logger.info("碱基不均衡占比 20%: {}", "[X] 违反" if ratio_violation else "[V] 未违反(<40%)")
    
    lane_high_ratio = [
        {'样本类型': 'ATAC-seq文库', '合同数据量_文库': 200},  # 碱基不均衡
        {'样本类型': 'Methylation文库', '合同数据量_文库': 150},  # 碱基不均衡
        {'样本类型': 'VIP-真核普通转录组文库', '合同数据量_文库': 150},  # 碱基均衡
        # 碱基不均衡占比: 350/500 = 70% >= 40%
    ]
    
    ratio_violation2 = checker.check_imbalance_ratio(lane_high_ratio)
    logger.info("碱基不均衡占比 70%: {}", "[X] 违反(>=40%)" if ratio_violation2 else "[V] 未违反")
    logger.info("")
    
    # 测试分组29占比规则
    logger.info("=== 分组29占比规则测试 ===")
    
    lane_g27_g28 = [
        {'样本类型': '10X转录组-3\'文库', '合同数据量_文库': 50},   # 分组27
        {'样本类型': '10X转录组-5\'文库', '合同数据量_文库': 200},  # 分组28
        {'样本类型': '10X转录组V(D)J-BCR文库', '合同数据量_文库': 200},  # 分组28
        # 分组27占比: 50/450 = 11% <= 20%
    ]
    
    g29_violation = checker.check_group29_ratio(lane_g27_g28)
    logger.info("分组27占比 11%: {}", "[X] 违反" if g29_violation else "[V] 未违反(<=20%)")
    
    lane_g27_high = [
        {'样本类型': '10X转录组-3\'文库', '合同数据量_文库': 150},   # 分组27
        {'样本类型': '10X Visium空间转录组文库', '合同数据量_文库': 100},  # 分组27
        {'样本类型': '10X转录组-5\'文库', '合同数据量_文库': 200},  # 分组28
        # 分组27占比: 250/450 = 55% > 20%
    ]
    
    g29_violation2 = checker.check_group29_ratio(lane_g27_high)
    logger.info("分组27占比 55%: {}", "[X] 违反(>20%)" if g29_violation2 else "[V] 未违反")
    logger.info("")


def test_lane_ratio_rules():
    """测试Lane级别占比规则"""
    checker = RuleChecker()
    
    logger.info("=== 10碱基Index占比测试（规则4） ===")
    
    # 测试1：混合且10碱基占比>40%（通过）
    lane_10base_pass = [
        {'合同数据量_文库': 60, 'index碱基数目': 10},  # 10碱基
        {'合同数据量_文库': 40, 'index碱基数目': 8},   # 非10碱基
        # 10碱基占比: 60/100 = 60% > 40% ✅
    ]
    result1 = checker.check_10base_index_ratio_lane(lane_10base_pass)
    logger.info("10碱基占比 60%: {}", "[X] 违反" if result1 else "[V] 通过(>40%)")
    
    # 测试2：混合且10碱基占比<=40%（违反）
    lane_10base_fail = [
        {'合同数据量_文库': 30, 'index碱基数目': 10},  # 10碱基
        {'合同数据量_文库': 70, 'index碱基数目': 8},   # 非10碱基
        # 10碱基占比: 30/100 = 30% <= 40%
    ]
    result2 = checker.check_10base_index_ratio_lane(lane_10base_fail)
    logger.info("10碱基占比 30%: {}", "[X] 违反(<=40%)" if result2 else "[V] 通过")
    
    # 测试3：全是10碱基（不限制）
    lane_all_10base = [
        {'合同数据量_文库': 50, 'index碱基数目': 10},
        {'合同数据量_文库': 50, 'index碱基数目': 10},
    ]
    result3 = checker.check_10base_index_ratio_lane(lane_all_10base)
    logger.info("全是10碱基: {}", "[X] 违反" if result3 else "[V] 通过(不限制)")
    logger.info("")
    
    logger.info("=== 单端Index占比测试（规则5） ===")
    
    # 测试1：混合且单端占比<30%（通过）
    lane_single_pass = [
        {'合同数据量_文库': 20, '是否双端index测序': '否'},  # 单端
        {'合同数据量_文库': 80, '是否双端index测序': '是'},  # 双端
        # 单端占比: 20/100 = 20% < 30% ✅
    ]
    result4 = checker.check_single_index_ratio_lane(lane_single_pass)
    logger.info("单端占比 20%: {}", "[X] 违反" if result4 else "[V] 通过(<30%)")
    
    # 测试2：混合且单端占比>=30%（违反）
    lane_single_fail = [
        {'合同数据量_文库': 40, '是否双端index测序': '否'},  # 单端
        {'合同数据量_文库': 60, '是否双端index测序': '是'},  # 双端
        # 单端占比: 40/100 = 40% >= 30%
    ]
    result5 = checker.check_single_index_ratio_lane(lane_single_fail)
    logger.info("单端占比 40%: {}", "[X] 违反(>=30%)" if result5 else "[V] 通过")
    logger.info("")
    
    logger.info("=== 客户/诺禾占比测试（规则6） ===")
    
    # 测试1：混合且客户占比≤50%（通过）
    lane_customer_pass = [
        {'合同数据量_文库': 40, '是否是客户文库': '是'},  # 客户
        {'合同数据量_文库': 60, '是否是客户文库': '否'},  # 诺禾
        # 客户占比: 40/100 = 40% <= 50% ✅
    ]
    result6 = checker.check_customer_ratio_lane(lane_customer_pass)
    logger.info("客户占比 40%: {}", "[X] 违反" if result6 else "[V] 通过(<=50%)")
    
    # 测试2：混合且客户占比>50%（违反）
    lane_customer_fail = [
        {'合同数据量_文库': 60, '是否是客户文库': '是'},  # 客户
        {'合同数据量_文库': 40, '是否是客户文库': '否'},  # 诺禾
        # 客户占比: 60/100 = 60% > 50%
    ]
    result7 = checker.check_customer_ratio_lane(lane_customer_fail)
    logger.info("客户占比 60%: {}", "[X] 违反(>50%)" if result7 else "[V] 通过")
    
    # 测试3：全是客户文库（不限制）
    lane_all_customer = [
        {'合同数据量_文库': 50, '是否是客户文库': '是'},
        {'合同数据量_文库': 50, '是否是客户文库': '是'},
    ]
    result8 = checker.check_customer_ratio_lane(lane_all_customer)
    logger.info("全是客户文库: {}", "[X] 违反" if result8 else "[V] 通过(不限制)")
    logger.info("")


def test_all_lane_rules():
    """测试所有Lane级别规则"""
    checker = RuleChecker()
    
    logger.info("=== 综合Lane级别规则测试 ===")
    
    # 构造一个复杂的Lane
    lane = [
        {'样本类型': 'VIP-真核普通转录组文库', '合同数据量_文库': 300, 'SID_文库': '001',
         'index碱基数目': 10, '是否双端index测序': '是', '是否是客户文库': '否'},
        {'样本类型': 'VIP-真核普通转录组文库', '合同数据量_文库': 250, 'SID_文库': '002',
         'index碱基数目': 10, '是否双端index测序': '是', '是否是客户文库': '否'},
        {'样本类型': 'ATAC-seq文库', '合同数据量_文库': 100, 'SID_文库': '003',
         'index碱基数目': 10, '是否双端index测序': '是', '是否是客户文库': '否'},
        {'样本类型': 'VIP-真核普通转录组文库', '合同数据量_文库': 200, 'SID_文库': '004',
         'index碱基数目': 10, '是否双端index测序': '是', '是否是客户文库': '否'},
        # 总计: 850G, 碱基不均衡占比: 100/850 = 11.8%
    ]
    
    results = checker.check_all_lane_rules(
        lane, 
        machine_type='Nova X-25B',
        process_code=1595,
        load_method='25B',
        priority='其他'
    )
    
    logger.info("Lane配置: Nova X-25B, 工序1595, 上机方式25B")
    logger.info("Lane文库数: {}", len(lane))
    logger.info("Lane总数据量: {:.2f}G", results['capacity_usage']['current_total'])
    logger.info("")
    logger.info("占比规则检测结果：")
    logger.info("  规则4 (10碱基占比): {}", "[X] 违反" if results['rule4_10base_ratio'] else "[V] 通过")
    logger.info("  规则5 (单端占比): {}", "[X] 违反" if results['rule5_single_index_ratio'] else "[V] 通过")
    logger.info("  规则6 (客户占比): {}", "[X] 违反" if results['rule6_customer_ratio'] else "[V] 通过")
    logger.info("  规则17 (碱基不均衡占比): {}", "[X] 违反" if results['rule17_imbalance_ratio'] else "[V] 通过")
    logger.info("  规则24 (分组29占比): {}", "[X] 违反" if results['rule24_group29_ratio'] else "[V] 通过")
    logger.info("")
    logger.info("容量和数量规则检测结果：")
    logger.info("  规则15 (Lane容量): {}", "[X] 违反" if results['rule15_capacity'] else "[V] 通过")
    logger.info("  规则18 (特殊文库总量): {}", "[X] 违反" if results['rule18_special_total'] else "[V] 通过")
    logger.info("  规则23 (文库类型数): {}", "[X] 违反" if results['rule23_type_count'] else "[V] 通过")
    logger.info("")
    logger.info("特殊工序规则检测结果：")
    logger.info("  规则22 (T7-C4 oligo): {}", "[X] 违反" if results['rule22_t7c4_oligo'] else "[V] 通过(N/A)")
    logger.info("")
    logger.info("总违反数: {}", results['total_violations'])
    logger.info("Lane是否有效: {}", "[V] 有效" if results['is_valid'] else "[X] 无效")
    logger.info("")


def test_rule_checker():
    """测试规则检测器"""
    checker = RuleChecker()
    
    # 测试用例1：高度兼容的文库对
    lib1 = {
        '机器类型': 'Nova X-25B',
        '工序编码': 1595,
        '样本类型': 'VIP-真核普通转录组文库',
        '数据类型': 'YC',
        'PeakSize': 440,
        'Index序列': 'ACAGAGCGAT;GACTACTCAC',
        '是否双端index测序': '是',
        'index碱基数目': 10,
        '测序策略': 'PE150',
        '库检综合结果': '合格',
        '是否是客户文库': '否',
        '合同数据量_文库': 5.2,
        '产线标识': 'S'
    }
    
    lib2 = {
        '机器类型': 'Nova X-25B',
        '工序编码': 1595,
        '样本类型': 'VIP-真核普通转录组文库',
        '数据类型': 'YC',
        'PeakSize': 445,
        'Index序列': 'GCCGTCTTAA;ACCTCGAACT',
        '是否双端index测序': '是',
        'index碱基数目': 10,
        '测序策略': 'PE150',
        '库检综合结果': '合格',
        '是否是客户文库': '否',
        '合同数据量_文库': 4.8,
        '产线标识': 'S'
    }
    
    summary1 = checker.get_violation_summary(lib1, lib2)
    logger.info("测试用例1：兼容的文库对")
    logger.info("  违反规则数: {}", summary1['violated_count'])
    logger.info("  是否兼容: {}", summary1['is_compatible'])
    logger.info("")
    
    # 测试用例2：明确不兼容的文库对
    lib3 = {
        '机器类型': 'ZM SURFSeq5000',  # 不同机器
        '工序编码': 1722,  # 不同工序
        '样本类型': '肿瘤靶向基因检测文库',
        '数据类型': '医学',
        'PeakSize': 350,
        'Index序列': 'ACAGAGCGAT;TGAGTGCACA',  # 可能冲突
        '是否双端index测序': '是',
        'index碱基数目': 10,
        '测序策略': 'PE150',
        '库检综合结果': '风险',
        '是否是客户文库': '否',
        '合同数据量_文库': 2.4,
        '产线标识': 'S'
    }
    
    summary2 = checker.get_violation_summary(lib1, lib3)
    logger.info("测试用例2：不兼容的文库对")
    logger.info("  违反规则数: {}", summary2['violated_count'])
    logger.info("  是否兼容: {}", summary2['is_compatible'])
    logger.info("  违反的规则:")
    for rule in summary2['violated_rules']:
        logger.info("    - [{}] {}", rule['rule_type'], rule['rule_name'])


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("文库对兼容性检测测试（规则0-14）")
    logger.info("=" * 60)
    test_rule_checker()
    
    logger.info("\n" + "=" * 60)
    logger.info("Lane容量检测测试（规则15）")
    logger.info("=" * 60)
    test_lane_capacity()
    
    logger.info("\n" + "=" * 60)
    logger.info("碱基不均衡规则测试（规则16-24）")
    logger.info("=" * 60)
    test_imbalance_rules()
    
    logger.info("\n" + "=" * 60)
    logger.info("Lane级别占比规则测试（规则4,5,6）")
    logger.info("=" * 60)
    test_lane_ratio_rules()
    
    logger.info("\n" + "=" * 60)
    logger.info("综合Lane级别规则测试")
    logger.info("=" * 60)
    test_all_lane_rules()

