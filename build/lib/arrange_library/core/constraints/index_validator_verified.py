"""
Index验证器 - 基于真实数据验证的Index冲突检测
经过生产数据验证的Index重复检测算法
创建时间：2025-12-17 18:00:00
更新时间：2026-04-12 10:00:00

算法来源：output/check_lane_index_repeat.new.py（真实数据验证通过）
修改记录：2026-01-27 - 单端与双端比对时只核对P7端，不再核对P5端
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger

# from liblane_paths import setup_liblane_paths
# setup_liblane_paths()
from arrange_library.models.library_info import EnhancedLibraryInfo


# 单端默认右端序列基准（从真实数据验证程序复制）
DEFAULT_SINGLE_RIGHT_BASE = "ACCGAGATCT"


class ConflictType(Enum):
    """Index冲突类型"""
    SINGLE_SINGLE = "single_single"  # 单端 vs 单端
    DUAL_DUAL = "dual_dual"          # 双端 vs 双端
    SINGLE_DUAL = "single_dual"      # 单端 vs 双端


@dataclass
class IndexConflict:
    """Index冲突记录"""
    library1_id: str
    library2_id: str
    conflict_type: ConflictType
    left1: str
    right1: Optional[str]
    left2: str
    right2: Optional[str]
    same_count_left: int
    same_count_right: Optional[int] = None


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    conflicts: List[IndexConflict]
    total_comparisons: int
    conflict_count: int


class IndexValidatorVerified:
    """
    基于真实数据验证的Index校验器
    
    核心算法：
    1. 重复判定规则：至少2个碱基不同才算不重复（same > L-2 判定为重复）
    2. 左端（P7）：左对齐比较前L个碱基
    3. 右端（P5）：右对齐截取后L个碱基再比较
    4. 单端vs双端：只核对P7端，P7重复即判定为冲突（2026-01-27修改）
    
    已在真实生产数据上验证通过
    """
    
    def __init__(self):
        """初始化校验器"""
        self.default_right_base = DEFAULT_SINGLE_RIGHT_BASE
        logger.info("Index校验器初始化完成（真实数据验证版本）")

    @staticmethod
    def _same_library_identity(
        lib_a: EnhancedLibraryInfo,
        lib_b: EnhancedLibraryInfo,
    ) -> bool:
        """相同 origrec 的重复明细视为同一文库，不参与自冲突判定。"""
        identity_a = str(getattr(lib_a, "origrec", "") or "").strip()
        identity_b = str(getattr(lib_b, "origrec", "") or "").strip()
        return bool(identity_a) and identity_a == identity_b
    
    def validate_lane(self, libraries: List[EnhancedLibraryInfo], silent: bool = False) -> ValidationResult:
        """
        校验Lane内所有文库的Index是否重复
        
        Args:
            libraries: Lane内的文库列表
            silent: 静默模式，不打印警告日志（用于启发式改进等高频调用场景）
            
        Returns:
            ValidationResult: 验证结果
        """
        conflicts = []
        total_comparisons = 0
        
        if len(libraries) < 2:
            return ValidationResult(
                is_valid=True,
                conflicts=[],
                total_comparisons=0,
                conflict_count=0
            )
        
        # 解析所有文库的Index序列
        lib_indices = []
        for lib in libraries:
            indices = self._parse_library_indices(lib)
            if indices:
                lib_indices.append({
                    'library': lib,
                    'library_id': getattr(lib, 'origrec', '') or str(id(lib)),
                    'indices': indices
                })
        
        # 检查不同文库之间的Index重复
        for i in range(len(lib_indices)):
            for j in range(i + 1, len(lib_indices)):
                lib_a = lib_indices[i]
                lib_b = lib_indices[j]
                if self._same_library_identity(lib_a["library"], lib_b["library"]):
                    continue
                
                # 比较两个文库的所有Index组合
                for left1, right1 in lib_a['indices']:
                    for left2, right2 in lib_b['indices']:
                        total_comparisons += 1
                        
                        is_repeat, conflict_type, same_left, same_right = self._check_index_pair_repeat(
                            left1, right1, left2, right2
                        )
                        
                        if is_repeat:
                            conflicts.append(IndexConflict(
                                library1_id=lib_a['library_id'],
                                library2_id=lib_b['library_id'],
                                conflict_type=conflict_type,
                                left1=left1,
                                right1=right1,
                                left2=left2,
                                right2=right2,
                                same_count_left=same_left,
                                same_count_right=same_right
                            ))
        
        is_valid = len(conflicts) == 0
        
        # 静默模式下不打印警告，避免启发式改进时产生大量日志
        if not is_valid and not silent:
            logger.warning(f"Lane内发现 {len(conflicts)} 个Index冲突")
        
        return ValidationResult(
            is_valid=is_valid,
            conflicts=conflicts,
            total_comparisons=total_comparisons,
            conflict_count=len(conflicts)
        )
    
    def validate_lane_quick(self, libraries: List[EnhancedLibraryInfo]) -> bool:
        """
        快速验证Lane内Index是否安全（静默模式，无日志）

        适用于校验一个完整列表（非增量场景）。若是在循环内逐个尝试添加
        文库，应优先使用 validate_new_lib_quick，避免重复检查已有文库对。

        Args:
            libraries: Lane内的文库列表

        Returns:
            bool: True表示安全，False表示有冲突
        """
        if len(libraries) < 2:
            return True

        # 解析所有文库的Index序列
        lib_indices = []
        for lib in libraries:
            indices = self._parse_library_indices(lib)
            if indices:
                lib_indices.append((lib, indices))

        # 快速检查：发现第一个冲突就返回False
        for i in range(len(lib_indices)):
            for j in range(i + 1, len(lib_indices)):
                lib_a, indices_a = lib_indices[i]
                lib_b, indices_b = lib_indices[j]
                if self._same_library_identity(lib_a, lib_b):
                    continue
                for left1, right1 in indices_a:
                    for left2, right2 in indices_b:
                        is_repeat, _, _, _ = self._check_index_pair_repeat(
                            left1, right1, left2, right2
                        )
                        if is_repeat:
                            return False

        return True

    def validate_new_lib_quick(
        self,
        existing_libs: List[EnhancedLibraryInfo],
        new_lib: EnhancedLibraryInfo,
    ) -> bool:
        """
        增量检查：只验证新文库与已有文库之间的 Index 冲突（O(n)）。

        在逐步向 Lane 添加文库的循环中，已有文库对之间不会再产生新冲突，
        不需要重复检查。用此方法代替 validate_lane_quick(existing + [new])
        可将复杂度从 O(n²) 降到 O(n)，对大 Lane（>50个文库）有显著提速。

        Args:
            existing_libs: Lane 中已确认无冲突的现有文库列表
            new_lib: 待加入的新文库

        Returns:
            bool: True 表示新文库与现有文库没有 Index 冲突，可以加入
        """
        if not existing_libs:
            return True

        new_indices = self._parse_library_indices(new_lib)
        # 新文库无 Index（如 NO INDEX），不需要检查
        if not new_indices:
            return True

        for existing_lib in existing_libs:
            if self._same_library_identity(existing_lib, new_lib):
                continue
            existing_indices = self._parse_library_indices(existing_lib)
            if not existing_indices:
                continue
            for left1, right1 in existing_indices:
                for left2, right2 in new_indices:
                    is_repeat, _, _, _ = self._check_index_pair_repeat(
                        left1, right1, left2, right2
                    )
                    if is_repeat:
                        return False

        return True

    def parse_lib_indices_cached(
        self, lib: EnhancedLibraryInfo
    ) -> List[Tuple[str, Optional[str]]]:
        """
        解析并返回文库的 Index 序列，外部可缓存结果以避免重复解析。

        Args:
            lib: 文库信息

        Returns:
            [(left, right), ...] 列表，空列表表示无 Index 或 NO INDEX
        """
        return self._parse_library_indices(lib)

    def validate_new_lib_quick_with_cache(
        self,
        existing_indices_cache: List[List[Tuple[str, Optional[str]]]],
        new_lib: EnhancedLibraryInfo,
    ) -> Tuple[bool, List[Tuple[str, Optional[str]]]]:
        """
        带缓存的增量检查：接受预解析的 existing 索引列表，只解析 new_lib 一次。

        在 _attempt_build_lane_from_pool 的候选遍历循环中，selected 里每个文库的
        索引都应预先解析并缓存起来（和 selected 列表平行维护），避免对同一文库
        反复调用 _parse_library_indices。这可以把单次检查的开销从 O(n * parse)
        降低到 O(n * compare)，n 为已选文库数。

        Args:
            existing_indices_cache: 与 selected 平行的预解析索引列表
            new_lib: 待加入的新文库

        Returns:
            (is_valid, new_lib_indices)
                - is_valid: True 表示无冲突，可以加入
                - new_lib_indices: new_lib 的解析结果，可追加到缓存供下次使用
        """
        new_indices = self._parse_library_indices(new_lib)
        if not new_indices:
            return True, new_indices

        for existing_indices in existing_indices_cache:
            if not existing_indices:
                continue
            for left1, right1 in existing_indices:
                for left2, right2 in new_indices:
                    is_repeat, _, _, _ = self._check_index_pair_repeat(
                        left1, right1, left2, right2
                    )
                    if is_repeat:
                        return False, new_indices

        return True, new_indices
    
    def _parse_library_indices(self, lib: EnhancedLibraryInfo) -> List[Tuple[str, Optional[str]]]:
        """
        解析文库的Index序列
        
        Args:
            lib: 文库信息
            
        Returns:
            List[Tuple[str, Optional[str]]]: [(left, right), ...] 列表
        """
        index_seq = getattr(lib, 'index_seq', None) or getattr(lib, 'indexseq', '')
        
        if not index_seq or str(index_seq).strip().upper() == "NO INDEX":
            return []
        
        index_seq = str(index_seq).strip()
        
        # 按逗号分隔多条Index
        items = [x.strip() for x in index_seq.split(",") if x.strip()]
        parsed = []
        
        for item in items:
            if ";" in item:
                # 双端Index：左;右
                parts = [p.strip() for p in item.split(";") if p.strip()]
                if len(parts) == 2:
                    parsed.append((parts[0], parts[1]))
                elif len(parts) == 1:
                    parsed.append((parts[0], None))
            else:
                # 单端Index
                parsed.append((item, None))
        
        return parsed
    
    def _check_index_pair_repeat(
        self, 
        left1: str, 
        right1: Optional[str], 
        left2: str, 
        right2: Optional[str]
    ) -> Tuple[bool, Optional[ConflictType], int, Optional[int]]:
        """
        检查两个Index是否重复
        
        核心算法（来自真实数据验证）：
        - 至少2个碱基不同才算不重复
        - same > (L - 2) 判定为重复
        
        Args:
            left1, right1: 第一个Index的左右序列
            left2, right2: 第二个Index的左右序列
            
        Returns:
            (is_repeat, conflict_type, same_count_left, same_count_right)
        """
        # 1. 检查左端（P7）
        is_left_repeat, same_left = self._side_is_repeated_left(left1, left2)
        
        if not is_left_repeat:
            # 左端不重复，整体不重复
            return False, None, same_left, None
        
        # 2. 左端重复，根据单双端组合检查右端
        if right1 is None and right2 is None:
            # 单端 vs 单端：只看左端，已重复
            return True, ConflictType.SINGLE_SINGLE, same_left, None
        
        elif right1 is not None and right2 is not None:
            # 双端 vs 双端：右端也需要重复
            is_right_repeat, same_right = self._side_is_repeated_right(right1, right2)
            if is_right_repeat:
                return True, ConflictType.DUAL_DUAL, same_left, same_right
            else:
                return False, None, same_left, same_right
        
        else:
            # 单端 vs 双端：只核对P7端（左端），不核对P5端
            # 到达这里说明左端（P7）已经重复，直接判定为冲突
            return True, ConflictType.SINGLE_DUAL, same_left, None
    
    def _side_is_repeated_aligned(self, seq1: str, seq2: str) -> Tuple[bool, int]:
        """
        判断对齐后的序列是否重复
        
        规则：至少2个碱基不同才算不重复
        即：same > (L - 2) 判定为重复
        
        Args:
            seq1, seq2: 已对齐的序列
            
        Returns:
            (is_repeated, same_count)
        """
        seq1 = seq1.strip().upper()
        seq2 = seq2.strip().upper()
        
        if not seq1 or not seq2:
            return False, 0
        
        L = min(len(seq1), len(seq2))
        s1 = seq1[:L]
        s2 = seq2[:L]
        
        same = sum(1 for a, b in zip(s1, s2) if a == b)
        
        # 至少2个碱基不同才算不重复
        is_repeated = same > (L - 2)
        
        return is_repeated, same
    
    def _side_is_repeated_left(self, seq1: str, seq2: str) -> Tuple[bool, int]:
        """
        左端（P7）重复判定：左对齐规则
        
        Args:
            seq1, seq2: P7序列
            
        Returns:
            (is_repeated, same_count)
        """
        return self._side_is_repeated_aligned(seq1, seq2)
    
    def _side_is_repeated_right(self, seq1: str, seq2: str) -> Tuple[bool, int]:
        """
        右端（P5）重复判定：右对齐规则
        
        从右侧截取L个碱基后再比较
        
        Args:
            seq1, seq2: P5序列
            
        Returns:
            (is_repeated, same_count)
        """
        seq1 = seq1.strip().upper()
        seq2 = seq2.strip().upper()
        
        if not seq1 or not seq2:
            return False, 0
        
        L = min(len(seq1), len(seq2))
        s1 = seq1[-L:]  # 右对齐：从右侧截取
        s2 = seq2[-L:]
        
        return self._side_is_repeated_aligned(s1, s2)
    
    def _make_default_single_right(self, left_seq: str) -> str:
        """
        根据单端左序列长度，从默认接头右端截取相同长度
        
        Args:
            left_seq: 左端序列
            
        Returns:
            默认右端序列
        """
        left_seq = (left_seq or "").strip()
        if not left_seq:
            return self.default_right_base
        
        take = min(len(left_seq), len(self.default_right_base))
        return self.default_right_base[-take:]


@dataclass
class ConflictResult:
    """冲突检测结果（兼容性）"""
    has_conflict: bool
    conflict_type: Optional[ConflictType] = None
    library1_id: str = ""
    library2_id: str = ""
    sequence1: str = ""
    sequence2: str = ""
    overlap_count: int = 0
    threshold: int = 0


@dataclass
class IndexSequence:
    """Index序列解析结果（兼容性）"""
    p7_sequence: str
    p5_sequence: Optional[str] = None
    is_pe_adapter: bool = False
    is_universal_adapter: bool = False
    is_random_primers: bool = False
    
    @property
    def is_single_end(self) -> bool:
        """是否为单端Index"""
        return self.p5_sequence is None
    
    @property
    def is_double_end(self) -> bool:
        """是否为双端Index"""
        return self.p5_sequence is not None


def validate_lane_index_safety(libraries: List[EnhancedLibraryInfo]) -> bool:
    """
    快速验证Lane内Index是否安全（兼容性函数）
    
    Args:
        libraries: Lane内的文库列表
        
    Returns:
        bool: True表示安全，False表示有冲突
    """
    validator = IndexValidatorVerified()
    result = validator.validate_lane(libraries)
    return result.is_valid


def get_index_conflicts_detail(libraries: List[EnhancedLibraryInfo]) -> dict:
    """
    获取详细的Index冲突信息（兼容性函数）
    
    Args:
        libraries: Lane内的文库列表
        
    Returns:
        dict: 包含冲突详情的字典
    """
    validator = IndexValidatorVerified()
    result = validator.validate_lane(libraries)
    
    return {
        'is_valid': result.is_valid,
        'total_comparisons': result.total_comparisons,
        'conflict_count': result.conflict_count,
        'conflicts': [
            {
                'library1_id': c.library1_id,
                'library2_id': c.library2_id,
                'conflict_type': c.conflict_type.value,
                'left1': c.left1,
                'right1': c.right1 or '',
                'left2': c.left2,
                'right2': c.right2 or '',
                'same_count_left': c.same_count_left,
                'same_count_right': c.same_count_right,
            }
            for c in result.conflicts
        ]
    }


# 为了保持兼容性，创建一个别名
IndexConflictValidator = IndexValidatorVerified
