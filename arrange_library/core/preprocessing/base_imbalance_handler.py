"""
碱基不均衡处理器
创建时间：2025-11-20 00:00:00
更新时间：2026-04-07 15:21:20
功能：处理碱基不均衡文库的分组映射、混排限制和包Lane规则
变更记录：
- 2025-12-24: 补充分组21-26（10X全基因组、10xATAC、HD Visium、FixedRNA等）
- 2025-12-25: [已废弃] 根据人工排机数据分析，放宽分组间混排限制
- 2026-01-28: 严格遵守规则文档，执行严格混排限制
  - 严格规则：仅允许G27(3')、G28(5')、G29(G27+G28)混排
  - 其他分组（G1-G26）必须"同类型排整条Lane"，不可与其他碱基不均衡文库混排
  - 恢复碱基不均衡总量限制为240G（统一口径）
  - G29规则：文库类型≤3种，G27占比≤20%
- 2026-04-07: 按“1-57分组表”升级规则口径
  - 分组1-52：默认不可与其他碱基不均衡分组混排
  - 分组53/54：分别支持组内混排
  - 分组55：高PhiX（>=20%）碱基不均衡类型可混排
  - 分组56/57：碱基均衡+不均衡组合按30%+15%模板校验
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
    phix_ratio: float = 0.0
    max_types_in_lane: int = 1

class BaseImbalanceHandler:
    """碱基不均衡规则处理器"""
    
    def __init__(self):
        self.groups = self._initialize_groups()
        self.type_to_group_map = self._build_type_map()
        self.group53_types = {
            "10X转录组-5'文库",
            "10X转录组文库-5V3文库",
            "10X转录组V(D)J-BCR文库",
            "10X转录组V(D)J-TCR文库",
            "客户-10X VDJ文库",
            "10X转录组-5‘膜蛋白文库",
            "客户-10X 5 Feature Barcode文库",
            "客户-10X 5 单细胞转录组文库",
            "客户-10X转录组V(D)J-BCR文库",
            "客户-10X转录组V(D)J-TCR文库",
        }
        self.group54_types = {
            "10X Visium FFPEV2空间转录组文库(V2)",
            "10X Visium空间转录组文库",
            "10X转录组-3‘膜蛋白文库",
            "客户-10X 3 Feature Barcode文库",
            "客户-10X 3 单细胞转录组文库",
            "客户文库-10X Visium 文库",
            "客户-10X Visium FFPEV2空间转录组文库(V2)",
            "客户-10X Visium空间转录组文库",
            "墨卓转录组-3端文库",
            "10X转录组-3'文库",
            "10X转录组文库-3V4文库",
        }
        self.mozhuo5_types = {
            "墨卓转录组-5端文库",
            "墨卓5'文库",
            "墨卓5端文库",
            "墨卓5",
        }
        self.methylation_types = {
            "Methylation文库",
            "客户-Methylation文库",
            "微量 (Methylation文库)",
            "RRBS文库",
        }
        self.group56_general_excluded_types = (
            set(self.group53_types) | set(self.group54_types) | set(self.mozhuo5_types)
        )
        self.group56_extra_types = {
            "10X转录组-5'文库",
            "10X转录组文库-5V3文库",
        }
        self.group57_general_excluded_types = (
            set(self.group53_types) | set(self.mozhuo5_types) | set(self.methylation_types)
        )
        self.group57_extra_types = set(self.methylation_types)
        self.group55_candidate_types = {
            rule_type
            for rule_type, gid in self.type_to_group_map.items()
            if gid in {
                "G16", "G17", "G18", "G22", "G23", "G24", "G25", "G26",
                "G28", "G29", "G31", "G32", "G33", "G36", "G40", "G41",
                "G42", "G43", "G44", "G45", "G46", "G47", "G48", "G50",
            }
        }

    @staticmethod
    def _normalize_type_name(value: str) -> str:
        return (
            str(value or "")
            .strip()
            .replace("’", "'")
            .replace("‘", "'")
            .replace("＇", "'")
        )

    def _get_library_type(self, lib: EnhancedLibraryInfo) -> str:
        return self._normalize_type_name(
            getattr(lib, "sample_type_code", None)
            or getattr(lib, "sampletype", None)
            or getattr(lib, "data_type", None)
            or getattr(lib, "lab_type", None)
            or ""
        )

    @staticmethod
    def _is_customer_library(lib: EnhancedLibraryInfo) -> bool:
        checker = getattr(lib, "is_customer_library", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                pass
        customer_flag = str(getattr(lib, "customer_library", "") or "").strip()
        if customer_flag in {"是", "客户", "Y", "YES", "TRUE", "1"}:
            return True
        if customer_flag in {"否", "N", "NO", "FALSE", "0"}:
            return False
        sample_id = str(getattr(lib, "sample_id", "") or "").strip()
        sample_type = str(getattr(lib, "sample_type_code", "") or "").strip()
        return sample_id.startswith("FKDL") or sample_type.startswith("客户")

    @staticmethod
    def _customer_ratio(libs: List[EnhancedLibraryInfo]) -> float:
        total = sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in libs)
        if total <= 0:
            return 0.0
        customer = sum(
            float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
            for lib in libs
            if BaseImbalanceHandler._is_customer_library(lib)
        )
        return customer / total
        
    def _initialize_groups(self) -> Dict[str, GroupDefinition]:
        """初始化分组定义"""
        groups = {}
        
        # 1-52：逐组定义（按用户提供的1-57规则表）
        base_group_specs = [
            ("G1", "10X Visium空间转录组文库", {"10X Visium空间转录组文库"}, 1.0, 0.0),
            ("G2", "10xATAC-seq文库", {"10xATAC-seq文库"}, 0.95, 0.05),
            ("G3", "10X全基因组文库", {"10X全基因组文库"}, 1.0, 0.0),
            ("G4", "10X转录组-3'文库", {"10X转录组-3'文库"}, 1.0, 0.0),
            ("G5", "10X转录组-5'文库", {"10X转录组-5'文库"}, 1.0, 0.0),
            ("G6", "10X转录组V(D)J-BCR文库", {"10X转录组V(D)J-BCR文库"}, 1.0, 0.0),
            ("G7", "10X转录组V(D)J-TCR文库", {"10X转录组V(D)J-TCR文库"}, 1.0, 0.0),
            ("G8", "客户-10X ATAC文库", {"客户-10X ATAC文库"}, 0.95, 0.05),
            ("G9", "客户-10X VDJ文库", {"客户-10X VDJ文库"}, 1.0, 0.0),
            ("G10", "客户-10X全基因组文库", {"客户-10X全基因组文库"}, 1.0, 0.0),
            ("G11", "客户-10X转录组V(D)J-BCR文库", {"客户-10X转录组V(D)J-BCR文库"}, 1.0, 0.0),
            ("G12", "客户-10X转录组V(D)J-TCR文库", {"客户-10X转录组V(D)J-TCR文库"}, 1.0, 0.0),
            ("G13", "客户-10X转录组文库", {"客户-10X转录组文库"}, 1.0, 0.0),
            ("G14", "10XFixed RNA文库", {"10XFixed RNA文库"}, 0.85, 0.15),
            ("G15", "客户文库-10X Visium 文库", {"客户文库-10X Visium 文库"}, 1.0, 0.0),
            ("G16", "ATAC-seq文库", {"ATAC-seq文库"}, 0.8, 0.2),
            ("G17", "CUT Tag文库（动物组织）", {"CUT Tag文库（动物组织）"}, 0.8, 0.2),
            ("G18", "CUT Tag文库（细胞）", {"CUT Tag文库（细胞）"}, 0.8, 0.2),
            ("G19", "EM-Seq文库", {"EM-Seq文库"}, 0.99, 0.01),
            ("G20", "Methylation文库", {"Methylation文库"}, 0.95, 0.05),
            ("G21", "small RNA文库", {"small RNA文库"}, 0.9, 0.1),
            ("G22", "单细胞文库", {"单细胞文库"}, 0.8, 0.2),
            ("G23", "动植物简化基因组文库(GBS亲代)", {"动植物简化基因组文库(GBS亲代)"}, 0.8, 0.2),
            ("G24", "动植物简化基因组文库(GBS子代)", {"动植物简化基因组文库(GBS子代)"}, 0.8, 0.2),
            ("G25", "客户-ATAC-seq文库", {"客户-ATAC-seq文库"}, 0.8, 0.2),
            ("G26", "客户-CUT-Tag文库", {"客户-CUT-Tag文库"}, 0.8, 0.2),
            ("G27", "客户-Methylation文库", {"客户-Methylation文库"}, 0.99, 0.01),
            ("G28", "客户-NanoString DSP文库", {"客户-NanoString DSP文库"}, 0.8, 0.2),
            ("G29", "客户-PCR产物", {"客户-PCR产物"}, 0.6, 0.4),
            ("G30", "客户-small RNA文库", {"客户-small RNA文库"}, 0.95, 0.05),
            ("G31", "客户-单细胞文库", {"客户-单细胞文库"}, 0.8, 0.2),
            ("G32", "客户-简化基因组", {"客户-简化基因组"}, 0.8, 0.2),
            ("G33", "客户-其他碱基不均衡文库", {"客户-其他碱基不均衡文库"}, 0.8, 0.2),
            ("G34", "墨卓转录组-3端文库", {"墨卓转录组-3端文库"}, 1.0, 0.0),
            ("G35", "微量 (Methylation文库)", {"微量 (Methylation文库)"}, 0.95, 0.05),
            ("G36", "客户-扩增子文库", {"客户-扩增子文库"}, 0.65, 0.35),
            ("G37", "客户-10X ATAC (Multiome)文库", {"客户-10X ATAC (Multiome)文库"}, 0.95, 0.05),
            ("G38", "客户-10X 5 Feature Barcode文库", {"客户-10X 5 Feature Barcode文库"}, 1.0, 0.0),
            ("G39", "客户-10X 3 Feature Barcode文库", {"客户-10X 3 Feature Barcode文库"}, 1.0, 0.0),
            ("G40", "客户-10X Flex文库", {"客户-10X Flex文库"}, 0.8, 0.2),
            ("G41", "客户-新生RNA文库", {"客户-新生RNA文库"}, 0.8, 0.2),
            ("G42", "客户-BD单细胞多样本标记/Sample Tag文库", {"客户-BD单细胞多样本标记/Sample Tag文库"}, 0.8, 0.2),
            ("G43", "客户-BD单细胞WTA文库", {"客户-BD单细胞WTA文库"}, 0.8, 0.2),
            ("G44", "客户-BD Abseq蛋白检测文库", {"客户-BD Abseq蛋白检测文库"}, 0.8, 0.2),
            ("G45", "客户-Parse单细胞WT文库", {"客户-Parse单细胞WT文库"}, 0.8, 0.2),
            ("G46", "客户-Parse单细胞BCR文库", {"客户-Parse单细胞BCR文库"}, 0.8, 0.2),
            ("G47", "客户-Parse单细胞TCR文库", {"客户-Parse单细胞TCR文库"}, 0.8, 0.2),
            ("G48", "客户-寻因单细胞文库", {"客户-寻因单细胞文库"}, 0.8, 0.2),
            ("G49", "客户-墨卓单细胞文库", {"客户-墨卓单细胞文库"}, 0.95, 0.05),
            ("G50", "客户-PCR产物/CRISPR", {"客户-PCR产物/CRISPR"}, 0.65, 0.35),
            ("G51", "客户-CUT-RUN文库", {"客户-CUT-RUN文库"}, 0.8, 0.2),
            ("G52", "RRBS文库", {"RRBS文库"}, 0.85, 0.15),
        ]

        for gid, desc, types, ratio, phix in base_group_specs:
            groups[gid] = GroupDefinition(
                group_id=gid,
                description=desc,
                library_types={self._normalize_type_name(item) for item in types},
                allow_internal_mixing=False,
                exclusive_lane=True,
                max_data_ratio=ratio,
                phix_ratio=phix,
                max_types_in_lane=1,
            )

        groups["G53"] = GroupDefinition(
            group_id="G53",
            description="10X 5'组合混排",
            library_types={
                self._normalize_type_name(item)
                for item in {
                    "10X转录组-5'文库",
                    "10X转录组文库-5V3文库",
                    "10X转录组V(D)J-BCR文库",
                    "10X转录组V(D)J-TCR文库",
                    "客户-10X VDJ文库",
                    "10X转录组-5‘膜蛋白文库",
                    "客户-10X 5 Feature Barcode文库",
                    "客户-10X 5 单细胞转录组文库",
                    "客户-10X转录组V(D)J-BCR文库",
                    "客户-10X转录组V(D)J-TCR文库",
                }
            },
            allow_internal_mixing=True,
            exclusive_lane=False,
            max_data_ratio=1.0,
            phix_ratio=0.0,
            max_types_in_lane=5,
        )

        groups["G54"] = GroupDefinition(
            group_id="G54",
            description="10X 3'/Visium组合混排",
            library_types={
                self._normalize_type_name(item)
                for item in {
                    "10X Visium FFPEV2空间转录组文库(V2)",
                    "10X Visium空间转录组文库",
                    "10X转录组-3‘膜蛋白文库",
                    "客户-10X 3 Feature Barcode文库",
                    "客户-10X 3 单细胞转录组文库",
                    "客户文库-10X Visium 文库",
                    "客户-10X Visium FFPEV2空间转录组文库(V2)",
                    "客户-10X Visium空间转录组文库",
                    "墨卓转录组-3端文库",
                    "10X转录组-3'文库",
                    "10X转录组文库-3V4文库",
                }
            },
            allow_internal_mixing=True,
            exclusive_lane=False,
            max_data_ratio=1.0,
            phix_ratio=0.0,
            max_types_in_lane=5,
        )

        ratio_map = self._get_group_data_ratio_map()
        for group_id, ratio in ratio_map.items():
            if group_id in groups:
                groups[group_id].max_data_ratio = ratio
        return groups

    @staticmethod
    def _get_group_data_ratio_map() -> Dict[str, float]:
        """返回文档定义的分组数据量占比（相对240G）。"""
        return {
            "G1": 1.0,
            "G2": 0.95,
            "G3": 1.0,
            "G4": 1.0,
            "G5": 1.0,
            "G6": 1.0,
            "G7": 1.0,
            "G8": 0.95,
            "G9": 1.0,
            "G10": 1.0,
            "G11": 1.0,
            "G12": 1.0,
            "G13": 1.0,
            "G14": 0.85,
            "G15": 1.0,
            "G16": 0.8,
            "G17": 0.8,
            "G18": 0.8,
            "G19": 0.99,
            "G20": 0.95,
            "G21": 0.9,
            "G22": 0.8,
            "G23": 0.8,
            "G24": 0.8,
            "G25": 0.8,
            "G26": 0.8,
            "G27": 0.99,
            "G28": 0.8,
            "G29": 0.6,
            "G30": 0.95,
            "G31": 0.8,
            "G32": 0.8,
            "G33": 0.8,
            "G34": 1.0,
            "G35": 0.95,
            "G36": 0.65,
            "G37": 0.95,
            "G38": 1.0,
            "G39": 1.0,
            "G40": 0.8,
            "G41": 0.8,
            "G42": 0.8,
            "G43": 0.8,
            "G44": 0.8,
            "G45": 0.8,
            "G46": 0.8,
            "G47": 0.8,
            "G48": 0.8,
            "G49": 0.95,
            "G50": 0.65,
            "G51": 0.8,
            "G52": 0.85,
            "G53": 1.0,
            "G54": 1.0,
        }

    def _build_type_map(self) -> Dict[str, str]:
        """构建 类型 -> 分组ID 的映射"""
        mapping = {}
        for gid, group in self.groups.items():
            for lib_type in group.library_types:
                mapping[self._normalize_type_name(lib_type)] = gid
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
        
        # 2. 根据文库类型查找分组（优先53/54组合映射）
        lib_type = self._get_library_type(lib)
        if lib_type:
            if lib_type in {self._normalize_type_name(item) for item in self.group53_types}:
                return "G53"
            if lib_type in {self._normalize_type_name(item) for item in self.group54_types}:
                return "G54"
            group_id = self.type_to_group_map.get(lib_type)
            if group_id:
                return group_id
        
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

    def _check_balanced_mix_template(
        self,
        imbalance_libs: List[EnhancedLibraryInfo],
        total_lane_data: float,
        general_excluded_types: Set[str],
        extra_allowed_types: Set[str],
        template_name: str,
    ) -> Tuple[bool, str]:
        """校验分组56/57的“30%主体+15%额外”模板。"""
        if total_lane_data <= 0:
            return True, ""

        normalized_excluded = {
            self._normalize_type_name(item) for item in general_excluded_types
        }
        normalized_extra = {
            self._normalize_type_name(item) for item in extra_allowed_types
        }

        general_data = 0.0
        extra_data = 0.0
        unsupported_types: Set[str] = set()

        for lib in imbalance_libs:
            lib_type = self._get_library_type(lib)
            data_gb = float(getattr(lib, "contract_data_raw", 0.0) or 0.0)
            if lib_type in normalized_extra:
                extra_data += data_gb
            elif lib_type in normalized_excluded:
                unsupported_types.add(lib_type)
            else:
                general_data += data_gb

        if unsupported_types:
            return False, f"{template_name}不允许包含: {', '.join(sorted(unsupported_types))}"

        general_ratio = general_data / total_lane_data
        if general_ratio > 0.30 + 1e-6:
            return False, f"{template_name}主组合占比{general_ratio:.1%}超过30%"

        extra_ratio = extra_data / total_lane_data
        if extra_ratio > 0.15 + 1e-6:
            return False, f"{template_name}额外组合占比{extra_ratio:.1%}超过15%"

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
            
        # 1. 识别所有文库的分组与类型
        group_ids: Set[str] = set()
        types: Set[str] = set()
        imbalance_libs = []
        
        for lib in libs:
            gid = self.identify_imbalance_type(lib)
            if gid and gid != "G_UNKNOWN":
                group_ids.add(gid)
                lib_type = self._get_library_type(lib)
                types.add(lib_type)
                imbalance_libs.append(lib)
                
        if not imbalance_libs:
            return True, "No imbalance libraries"
        balanced_libs = [lib for lib in libs if self.identify_imbalance_type(lib) in {None, "G_UNKNOWN"}]
        has_balanced = bool(balanced_libs)
        if "G53" in group_ids and "G54" in group_ids:
            return False, "分组53与分组54不可同Lane混排"

        if not has_balanced:
            if group_ids == {"G53"}:
                ratio = self._customer_ratio(imbalance_libs)
                if ratio > 0.5:
                    return False, f"分组53客户占比{ratio:.1%}超过50%"
            elif group_ids == {"G54"}:
                ratio = self._customer_ratio(imbalance_libs)
                if ratio > 0.5:
                    return False, f"分组54客户占比{ratio:.1%}超过50%"
            elif group_ids.issubset({f"G{i}" for i in range(1, 53)}):
                if len(group_ids) == 1:
                    gid = next(iter(group_ids))
                    group_def = self.groups.get(gid)
                    if group_def and len(types) > group_def.max_types_in_lane:
                        return False, f"{gid}同Lane类型超过{group_def.max_types_in_lane}种(当前{len(types)}种)"
                else:
                    # 分组55：仅允许高PhiX（>=20%）类型参与混排
                    if not all(lib_type in self.group55_candidate_types for lib_type in types):
                        return False, "仅高PhiX类型可按分组55混排"
                    ratio = self._customer_ratio(imbalance_libs)
                    if ratio > 0.5:
                        return False, f"分组55客户占比{ratio:.1%}超过50%"
            else:
                return False, f"存在未支持的分组组合: {', '.join(sorted(group_ids))}"
        else:
            total_lane_data = sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in libs)
            if total_lane_data <= 0:
                return True, "Compatible"

            imbalance_total_data = sum(float(getattr(lib, "contract_data_raw", 0.0) or 0.0) for lib in imbalance_libs)
            rule56_ok, rule56_reason = self._check_balanced_mix_template(
                imbalance_libs=imbalance_libs,
                total_lane_data=total_lane_data,
                general_excluded_types=self.group56_general_excluded_types,
                extra_allowed_types=self.group56_extra_types,
                template_name="分组56",
            )
            rule57_ok, rule57_reason = self._check_balanced_mix_template(
                imbalance_libs=imbalance_libs,
                total_lane_data=total_lane_data,
                general_excluded_types=self.group57_general_excluded_types,
                extra_allowed_types=self.group57_extra_types,
                template_name="分组57",
            )
            if not (rule56_ok or rule57_ok):
                reasons = "；".join(reason for reason in (rule56_reason, rule57_reason) if reason)
                if reasons:
                    return False, f"混排未满足分组56/57约束（30%+15%）: {reasons}"
                return False, "混排未满足分组56/57约束（30%+15%）"
            if imbalance_total_data / total_lane_data > 0.95:
                return False, f"混排场景碱基不均衡占比{imbalance_total_data / total_lane_data:.1%}超过95%"

        if enforce_total_limit:
            # 统一保留25B/NovaSeq X Plus碱基不均衡总量240G约束
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
