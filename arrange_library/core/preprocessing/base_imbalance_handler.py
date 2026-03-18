"""
碱基不均衡处理器
创建时间：2025-11-20 00:00:00
更新时间：2026-03-04 19:41:04
功能：处理碱基不均衡文库的分组映射、混排限制和包Lane规则
变更记录：
- 2025-12-24: 补充分组21-26（10X全基因组、10xATAC、HD Visium、FixedRNA等）
- 2025-12-25: [已废弃] 根据人工排机数据分析，放宽分组间混排限制
- 2026-01-28: 严格遵守规则文档，执行严格混排限制
  - 严格规则：仅允许G27(3')、G28(5')、G29(G27+G28)混排
  - 其他分组（G1-G26）必须"同类型排整条Lane"，不可与其他碱基不均衡文库混排
  - 恢复碱基不均衡总量限制为240G（统一口径）
  - G29规则：文库类型≤3种，G27占比≤20%
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from collections import defaultdict

from arrange_library.models.library_info import EnhancedLibraryInfo

# 配置常量
DEFAULT_MAX_DATA_GB: float = 240.0  # 默认最大数据量(GB) 2024.11.21更新

@dataclass
class GroupDefinition:
    group_id: str
    description: str
    library_types: Set[str]
    allow_internal_mixing: bool = False  # 是否允许同组内不同类型混排
    exclusive_lane: bool = True # 是否必须整条Lane
    max_data_gb: float = DEFAULT_MAX_DATA_GB  # 最大数据量(GB)
    max_data_ratio: float = 1.0  # 分组数据量占比系数（相对240G）

class BaseImbalanceHandler:
    """碱基不均衡规则处理器"""
    
    def __init__(self):
        self.groups = self._initialize_groups()
        self.type_to_group_map = self._build_type_map()
        
    def _initialize_groups(self) -> Dict[str, GroupDefinition]:
        """初始化分组定义"""
        groups = {}
        
        # 分组1: ATAC-seq
        groups['G1'] = GroupDefinition('G1', 'ATAC-seq', {'ATAC-seq文库'}, exclusive_lane=False) # 0.8占比，意味着非exclusive? 规则说"不与...混排"
        # 规则：分组1 ATAC-seq文库: 0.8占比，不可与其他碱基不均衡文库同Lane -> 意味着可以和碱基均衡混排？
        # 规则表述："同类型排整条Lane" vs "不可与其他碱基不均衡文库同Lane"
        # 这里exclusive_lane=True 表示不与其他碱基不均衡混排，但可能和均衡混排（如果允许）
        
        # 分组2-3: CUT Tag
        groups['G2'] = GroupDefinition('G2', 'CUT Tag(动物)', {'CUT Tag文库（动物组织）'})
        groups['G3'] = GroupDefinition('G3', 'CUT Tag(细胞)', {'CUT Tag文库（细胞）'})
        
        # 分组4: EM-Seq
        groups['G4'] = GroupDefinition('G4', 'EM-Seq', {'EM-Seq文库'})
        
        # 分组5: Methylation
        groups['G5'] = GroupDefinition('G5', 'Methylation', {
            'Methylation文库', 'RRBS文库', 'RNA甲基化文库', 'RNA甲基化文库-Input', 
            'RNA甲基化文库-IP', '羟甲基化文库', 'Ribo-seq文库'
        })
        
        # 分组6: small RNA
        groups['G6'] = GroupDefinition('G6', 'small RNA', {'small RNA文库', 'UMI smallRNA文库', '外泌体small RNA'})
        
        # 分组7: 单细胞
        groups['G7'] = GroupDefinition('G7', '单细胞', {'单细胞文库', 'circRNA'})
        
        # 分组8-9: GBS
        groups['G8'] = GroupDefinition('G8', 'GBS亲代', {'动植物简化基因组文库(GBS亲代)'})
        groups['G9'] = GroupDefinition('G9', 'GBS子代', {'动植物简化基因组文库(GBS子代)'})
        
        # 分组10-20: 客户系列
        groups['G10'] = GroupDefinition('G10', '客户-ATAC', {'客户-ATAC-seq文库'})
        groups['G11'] = GroupDefinition('G11', '客户-CUT-Tag', {'客户-CUT-Tag文库'})
        groups['G12'] = GroupDefinition('G12', '客户-Methylation', {'客户-Methylation文库', '客户-RRBS文库'})
        groups['G13'] = GroupDefinition('G13', '客户-NanoString', {'客户-NanoString DSP文库', 'DSP空间转录组文库'})
        groups['G14'] = GroupDefinition('G14', '客户-PCR', {'客户-PCR产物', 'PCR-free (PCR产物单建)'})
        groups['G15'] = GroupDefinition('G15', '客户-small RNA', {'客户-small RNA文库'})
        groups['G16'] = GroupDefinition('G16', '客户-单细胞', {'客户-单细胞文库', '客户-BD单细胞WTA文库', 'BD单细胞WTA文库'})
        groups['G17'] = GroupDefinition('G17', '客户-简化基因组', {
            '客户-简化基因组', '客户-GBS(康奈尔GBS文库)', 'GBS(康奈尔GBS文库)', 
            '客户-RAD文库 (单)', '客户-RAD文库 (混)', 'RAD文库 (单)', 'RAD文库 (混)'
        })
        groups['G18'] = GroupDefinition('G18', '客户-其他', {'客户-其他碱基不均衡文库', 'RAD-seq文库(单建)', 'RAD-seq文库(混建)'})
        groups['G19'] = GroupDefinition('G19', '微量Methylation', {
            '微量 (Methylation文库)', '微量RNA甲基化文库', '微量甲基化', 
            '低起始量RNA甲基化文库-Input(人)', '低起始量RNA甲基化文库-Input(鼠)',
            '低起始量RNA甲基化文库-IP(人)', '低起始量RNA甲基化文库-IP(鼠)', '外泌体lncRNA文库'
        })
        groups['G20'] = GroupDefinition('G20', '客户-扩增子', {'客户-扩增子文库'})
        
        # 分组21-26: 2024.11.21补充
        # 分组21: 10X全基因组文库 (1.0)
        groups['G21'] = GroupDefinition('G21', '10X全基因组', {'10X全基因组文库'})
        
        # 分组22: 客户-10X全基因组文库 (1.0)
        groups['G22'] = GroupDefinition('G22', '客户-10X全基因组', {'客户-10X全基因组文库'})
        
        # 分组23: 10xATAC-seq文库 (0.8)
        groups['G23'] = GroupDefinition('G23', '10xATAC-seq', {
            '10xATAC-seq文库', '10x ATAC文库', '10XATAC文库'
        })
        
        # 分组24: 客户-10X ATAC文库 (0.8)
        groups['G24'] = GroupDefinition('G24', '客户-10X ATAC', {
            '客户-10X ATAC文库', '客户-10X ATAC (Multiome)文库'
        })
        
        # 分组25: 10x HD Visium空间转录组文库(新) (1.0)
        groups['G25'] = GroupDefinition('G25', '10x HD Visium(新)', {
            '10x HD Visium空间转录组文库(新)', '10x HD Visium空间转录组文库'
        })
        
        # 分组26: 10XFixed RNA文库 (0.8)
        groups['G26'] = GroupDefinition('G26', '10XFixed RNA', {
            '10XFixed RNA文库', '10X转录组-FixedRNA文库', '10x膜蛋白文库'
        })
        
        # 分组27 (3')
        groups['G27'] = GroupDefinition('G27', '3端转录组', {
            '墨卓转录组-3端文库', '10X转录组-3\'文库', '10X Visium空间转录组文库', 
            '10X转录组-3\'膜蛋白文库', '客户-10X 3 单细胞转录组文库', 
            '10X Visium FFPEV2空间转录组文库(V2)', '客户文库-10X Visium 文库', 
            '客户-10X 3 Feature Barcode文库', '客户-10X Visium空间转录组文库', 
            '客户-10X Visium FFPEV2空间转录组文库(V2)', '客户-10X Feature Barcode文库' # 修正：文档列出11种，补齐
        }, allow_internal_mixing=True)
        
        # 分组28 (5')
        groups['G28'] = GroupDefinition('G28', '5端转录组', {
            '10X转录组-5\'文库', '10X转录组V(D)J-BCR文库', '10X转录组V(D)J-TCR文库', 
            '客户-10X VDJ文库', '客户-10X转录组V(D)J-BCR文库', '客户-10X转录组V(D)J-TCR文库', 
            '客户-10X转录组文库', '10X转录组-5\'膜蛋白文库', '客户-10X 5 单细胞转录组文库',
            '客户-10X 5 Feature Barcode文库', '客户-10X Flex文库'
        }, allow_internal_mixing=True)
        
        # 分组29是 G27 + G28 混排，不单独定义GroupDefinition，在逻辑中处理
        ratio_map = self._get_group_data_ratio_map()
        for group_id, ratio in ratio_map.items():
            if group_id in groups:
                groups[group_id].max_data_ratio = ratio
        return groups

    @staticmethod
    def _get_group_data_ratio_map() -> Dict[str, float]:
        """返回文档定义的分组数据量占比（相对240G）。"""
        return {
            "G1": 0.8,
            "G2": 0.8,
            "G3": 0.8,
            "G4": 0.99,
            "G5": 0.99,
            "G6": 1.0,
            "G7": 0.85,
            "G8": 0.8,
            "G9": 0.8,
            "G10": 0.8,
            "G11": 0.8,
            "G12": 0.99,
            "G13": 0.8,
            "G14": 0.6,
            "G15": 1.0,
            "G16": 0.85,
            "G17": 0.8,
            "G18": 0.8,
            "G19": 0.99,
            "G20": 0.65,
            "G21": 1.0,
            "G22": 1.0,
            "G23": 0.8,
            "G24": 0.8,
            "G25": 1.0,
            "G26": 0.8,
            "G27": 1.0,
            "G28": 1.0,
        }

    def _build_type_map(self) -> Dict[str, str]:
        """构建 类型 -> 分组ID 的映射"""
        mapping = {}
        for gid, group in self.groups.items():
            for lib_type in group.library_types:
                mapping[lib_type] = gid
        return mapping
        
    def identify_imbalance_type(self, lib: EnhancedLibraryInfo) -> Optional[str]:
        """识别文库是否为碱基不均衡，返回分组ID
        
        优先使用文库对象的jjbj字段（数据预处理时已标记）
        然后根据文库类型映射到具体分组
        
        Args:
            lib: 文库信息对象
            
        Returns:
            分组ID（如'G1', 'G27'等），非碱基不均衡返回None
        """
        # 1. 优先检查jjbj字段（数据预处理时已标记）
        jjbj = getattr(lib, 'jjbj', None)
        if jjbj is not None and str(jjbj).strip() == '否':
            return None  # 明确标记为非碱基不均衡
        
        # 2. 根据文库类型查找分组
        # 尝试多个可能的类型字段
        lib_type = (
            getattr(lib, 'sample_type_code', None) or 
            getattr(lib, 'data_type', None) or 
            getattr(lib, 'lab_type', None) or
            getattr(lib, 'library_type', None)
        )
        
        if lib_type:
            # 精确匹配
            group_id = self.type_to_group_map.get(lib_type)
            if group_id:
                return group_id
            
            # 模糊匹配（处理类型名称变体）
            lib_type_lower = lib_type.lower()
            for known_type, gid in self.type_to_group_map.items():
                if known_type.lower() in lib_type_lower or lib_type_lower in known_type.lower():
                    return gid
        
        # 3. 如果jjbj字段标记为"是"，但没有匹配到具体分组，返回通用标记
        if jjbj is not None and str(jjbj).strip() == '是':
            # 返回一个通用的碱基不均衡标记（非特定分组）
            return 'G_UNKNOWN'
        
            return None
    
    def is_imbalance_library(self, lib: EnhancedLibraryInfo) -> bool:
        """判断文库是否为碱基不均衡文库
        
        优先使用jjbj字段，其次使用分组识别
        
        Args:
            lib: 文库信息对象
            
        Returns:
            True表示是碱基不均衡文库
        """
        # 优先使用jjbj字段
        jjbj = getattr(lib, 'jjbj', None)
        if jjbj is not None and str(jjbj).strip() != '':
            return str(jjbj).strip() == '是'
        
        # 回退到分组识别
        return self.identify_imbalance_type(lib) is not None

    def get_group_info(self, group_id: str) -> Optional[GroupDefinition]:
        """获取分组信息"""
        return self.groups.get(group_id)

    def get_group_data_ratio(self, group_id: str) -> float:
        """获取分组数据量占比，未知分组默认1.0。"""
        group_info = self.get_group_info(group_id)
        if group_info is None:
            return 1.0
        return float(group_info.max_data_ratio or 1.0)

    def check_group_data_ratio(self, libs: List[EnhancedLibraryInfo]) -> Tuple[bool, str]:
        """严格检查各分组数据量占比约束（以240G为基准）。"""
        if not libs:
            return True, ""

        group_data: Dict[str, float] = defaultdict(float)
        for lib in libs:
            group_id = self.identify_imbalance_type(lib)
            if not group_id:
                continue
            group_data[group_id] += float(getattr(lib, "contract_data_raw", 0.0) or 0.0)

        if not group_data:
            return True, ""

        for group_id, data_gb in group_data.items():
            ratio = self.get_group_data_ratio(group_id)
            allowed = DEFAULT_MAX_DATA_GB * ratio
            if data_gb > allowed + 1e-6:
                return (
                    False,
                    f"分组{group_id}数据量{data_gb:.1f}G超过占比上限{allowed:.1f}G"
                    f"(240G×{ratio:.2f})",
                )
        return True, ""

    def check_mix_compatibility(
        self,
        libs: List[EnhancedLibraryInfo],
        enforce_total_limit: bool = True,
    ) -> Tuple[bool, str]:
        """
        检查一组文库是否符合碱基不均衡混排规则
        
        [2026-01-28] 严格遵守规则文档：
        1. 分组1-26（除27/28）："同类型排整条Lane"，不可与其他碱基不均衡文库混排
        2. 分组27(3')：同组可混排，同Lane不超过3种
        3. 分组28(5')：同组可混排，同Lane不超过3种
        4. 分组29（G27+G28混排）：文库类型≤3种，G27占比≤20%
        
        Args:
            libs: 文库列表
            
        Returns:
            (is_compatible, reason)
        """
        if not libs:
            return True, ""
            
        # 1. 识别所有文库的分组
        group_ids = set()
        types = set()
        imbalance_libs = []
        
        for lib in libs:
            gid = self.identify_imbalance_type(lib)
            if gid:
                group_ids.add(gid)
                lib_type = (
                    getattr(lib, "sample_type_code", None)
                    or getattr(lib, "data_type", None)
                    or getattr(lib, "lab_type", None)
                    or ""
                )
                types.add(lib_type)
                imbalance_libs.append(lib)
                
        if not imbalance_libs:
            return True, "No imbalance libraries"
            
        # 2. 检查互斥规则（硬性约束，必须保留）
        # 10x HD Visium (新) 不与 3' 类混排
        has_hd_new = any('10x HD Visium空间转录组文库(新)' in getattr(l, 'data_type', '') for l in imbalance_libs)
        has_3_prime = any(gid == 'G27' for gid in group_ids) # G27 is 3'
        
        if has_hd_new and has_3_prime:
            return False, "10x HD Visium(新)与3'类文库互斥"
            
        # 10X ATAC 不与 10x_cellranger拆分方式 混排
        has_atac = 'G1' in group_ids or 'G10' in group_ids  # G1: ATAC-seq, G10: 客户-ATAC
        has_cellranger = any(
            self._is_cellranger_library(lib) for lib in imbalance_libs
        )
        
        if has_atac and has_cellranger:
            return False, "10X ATAC与cellranger拆分方式文库不能同Run排机"
        
        # ===== [2026-01-28 恢复严格规则] 分组间混排限制 =====
        # 3. 检查同组/跨组混排规则
        if len(group_ids) > 2:
            # 大多数情况下不能超过2个组（除非是 G27+G28 这种特殊组合）
            # 实际上规则说：大多数碱基不均衡文库不可与其他碱基不均衡文库同Lane
            # 只有 G27 和 G28 可以混排 (形成 G29)
            return False, f"涉及超过允许的分组数量(当前{len(group_ids)}个分组: {', '.join(sorted(group_ids))})"
            
        if len(group_ids) == 1:
            gid = list(group_ids)[0]
            group_def = self.groups.get(gid)
            if group_def and len(types) > 1 and not group_def.allow_internal_mixing:
                return False, f"分组{gid}不允许同组内不同类型混排(当前{len(types)}种)"
            
            # G27/G28 同组混排限制：同Lane不超过3种
            if gid in ['G27', 'G28'] and len(types) > 3:
                return False, f"分组{gid}内混排类型超过3种(当前{len(types)}种)"
                
        elif len(group_ids) == 2:
            if not ({'G27', 'G28'} == group_ids):
                return False, f"不允许的分组间混排(当前: {', '.join(sorted(group_ids))}, 规则仅允许 G27+G28)"
                
            # G29 规则校验
            # 1. 类型不超过3种
            if len(types) > 3:
                return False, f"G27+G28混排类型超过3种(当前{len(types)}种)"
            
            # 2. G27占比 <= 20%
            g27_data = sum(float(l.contract_data_raw or 0) for l in imbalance_libs if self.identify_imbalance_type(l) == 'G27')
            total_data = sum(float(l.contract_data_raw or 0) for l in imbalance_libs)
            if total_data > 0 and (g27_data / total_data > 0.20):
                return False, f"G27占比({g27_data/total_data:.1%})超过20%"
                
        # [2025-12-25 已废弃] 人工排机实际允许不同分组混排（如Lane 328704: G1+G14+G16混排）
        # [2026-01-28] 恢复严格规则，不再使用宽松的5种类型限制
        # if len(types) > 5:
        #     return False, f"碱基不均衡文库类型过多({len(types)}种 > 5种)"
                
        if enforce_total_limit:
            # ===== [2026-03-04 规则统一] 总量限制 =====
            # 按照规则文档统一口径：碱基不均衡总量上限240G
            total_imbalance_data = sum(float(l.contract_data_raw or 0) for l in imbalance_libs)
            limit = DEFAULT_MAX_DATA_GB
            if total_imbalance_data > limit:
                 return False, f"碱基不均衡总量({total_imbalance_data:.1f}G)超过限制({limit:.0f}G)"

        return True, "Compatible"

    def _is_cellranger_library(self, lib: EnhancedLibraryInfo) -> bool:
        """检查文库是否使用cellranger拆分方式
        
        Args:
            lib: 文库信息
            
        Returns:
            是否为cellranger拆分方式文库
        """
        # 从文库类型检查
        lib_type = getattr(lib, 'data_type', '') or ''
        if '10x_cellranger' in lib_type.lower() or 'cellranger' in lib_type.lower():
            return True
        
        # 从备注检查
        remarks = getattr(lib, 'remark', '') or ''
        machine_notes = getattr(lib, 'machine_notes', '') or ''
        combined_remarks = f"{remarks} {machine_notes}".lower()
        
        if 'cellranger' in combined_remarks:
            return True
        
        # 从拆分方式字段检查
        split_method = getattr(lib, 'split_method', '') or ''
        if 'cellranger' in split_method.lower():
            return True
        
        return False

    def get_max_data_limit(self, library_type: str) -> float:
        """获取特定类型的最大数据量限制"""
        _ = library_type
        return DEFAULT_MAX_DATA_GB

