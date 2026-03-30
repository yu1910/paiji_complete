"""
文库拆分器
创建时间：2025-11-20 10:00:00
更新时间：2026-03-05 10:48:32
功能：严格按照《排机规则文档》执行文库拆分，支持多级拆分
"""

import copy
import math
import uuid
from typing import Any, Dict, List, Tuple

from loguru import logger

from arrange_library.models.library_info import EnhancedLibraryInfo
from arrange_library.core.config.scheduling_config import get_library_split_config


class LibrarySplitter:
    """文库拆分器"""

    MODE_ONE_POINT_ZERO = "1.0"
    MODE_OTHER = "other"

    def __init__(self):
        split_config = get_library_split_config()
        self.single_index_non_1_0_threshold = split_config.single_index_non_1_0_threshold_gb
        self.single_index_mode_1_0_threshold = split_config.single_index_mode_1_0_threshold_gb
        self.multi_index_threshold = split_config.multi_index_threshold_gb
        self.min_split_size = split_config.min_split_size_gb
        
    def split_libraries(self, libraries: List[EnhancedLibraryInfo]) -> Tuple[List[EnhancedLibraryInfo], List[dict]]:
        """
        执行文库拆分
        
        Args:
            libraries: 原始文库列表
            
        Returns:
            Tuple[List[EnhancedLibraryInfo], List[dict]]: (处理后的文库列表, 拆分记录)
        """
        logger.info("=" * 60)
        logger.info("[拆分] 开始文库拆分预处理")
        logger.info(
            "  拆分规则: 非1.0单index >{}G；1.0单index >{}G；多index >{}G".format(
                self.single_index_non_1_0_threshold,
                self.single_index_mode_1_0_threshold,
                self.multi_index_threshold,
            )
        )
        logger.info(f"  最小保留数据量: >{self.min_split_size}G")
        
        processed_libraries = []
        split_records = []
        
        split_count = 0
        original_count = len(libraries)
        
        for lib in libraries:
            # 检查是否需要拆分
            if self._should_split(lib):
                split_libs = self._perform_split(lib)
                
                # 验证拆分结果有效性（单个文库 > 2G）
                if all(sl.contract_data_raw > self.min_split_size for sl in split_libs):
                    processed_libraries.extend(split_libs)
                    split_count += 1
                    
                    split_records.append({
                        'original_id': lib.origrec,
                        'original_size': lib.contract_data_raw,
                        'split_count': len(split_libs),
                        'new_ids': [sl.origrec for sl in split_libs]
                    })
                    logger.debug(f"  拆分文库 {lib.origrec} ({lib.contract_data_raw}G) -> {len(split_libs)}个子文库")
                else:
                    # 拆分后太小，不拆分
                    logger.warning(f"  文库 {lib.origrec} 拆分后数据量小于{self.min_split_size}G，跳过拆分")
                    processed_libraries.append(lib)
            else:
                processed_libraries.append(lib)
                
        logger.info("-" * 60)
        logger.info(f" 拆分完成")
        logger.info(f"  原始文库数: {original_count}")
        logger.info(f"  拆分文库数: {split_count}")
        logger.info(f"  最终文库数: {len(processed_libraries)}")
        logger.info("=" * 60)
        
        return processed_libraries, split_records
    
    def _should_split(self, lib: EnhancedLibraryInfo) -> bool:
        """判断是否需要拆分

        新规则：
        1. 非1.0模式单index合同数据量 > 100G 时拆分
        2. 1.0模式单index合同数据量 > 200G 时拆分
        3. 多index合同数据量 > 300G 时拆分
        """
        # 1. 包Lane/包FC/指定Lane不拆分
        if self._has_fixed_lane_binding(lib):
            return False

        # 2. 读取合同量
        try:
            data_amount = float(lib.contract_data_raw or 0)
        except (ValueError, TypeError):
            return False

        if data_amount <= 0:
            return False

        rule_label, max_data_per_fragment = self._resolve_split_rule(lib)

        should_split = data_amount > max_data_per_fragment
        if should_split:
            logger.debug(
                "  文库 {} 触发拆分: 合同量={}G, 规则={}, 阈值={}G".format(
                    lib.origrec,
                    round(data_amount, 3),
                    rule_label,
                    round(max_data_per_fragment, 3),
                )
            )
        return should_split

    def _resolve_split_rule(self, lib: EnhancedLibraryInfo) -> Tuple[str, float]:
        """根据业务规则解析拆分阈值。"""
        index_count = self._count_index_pairs(lib)
        if index_count > 1:
            return "multi_index", self.multi_index_threshold

        mode = self._detect_sequence_mode(lib)
        if mode == self.MODE_ONE_POINT_ZERO:
            return "single_index_mode_1.0", self.single_index_mode_1_0_threshold
        return "single_index_mode_other", self.single_index_non_1_0_threshold

    def _count_index_pairs(self, lib: EnhancedLibraryInfo) -> int:
        """计算index对数

        规则说明：
        - index序列中被逗号（,）隔开的叫多对index
        - 被分号（;）隔开的叫一对index（P7;P5）
        - 没有分号隔开的就是单端index，也叫单个index
        
        例如：ATCG;GCTA,TTAA;GGCC 表示2对index
        """
        index_seq = getattr(lib, 'index_seq', '') or ''
        if not index_seq:
            return 1
        
        # 被逗号分隔的是多对index
        pairs = [seg.strip() for seg in index_seq.split(',') if seg.strip()]
        return max(len(pairs), 1)

    def _has_fixed_lane_binding(self, lib: EnhancedLibraryInfo) -> bool:
        """包Lane/包FC/指定Lane/指定FC文库不进行拆分。"""
        fields = [
            getattr(lib, "package_lane_number", None),
            getattr(lib, "package_fc_number", None),
            getattr(lib, "lane_id", None),
            getattr(lib, "fc_id", None),
            getattr(lib, "runid", None),
        ]
        for value in fields:
            if value is None:
                continue
            if str(value).strip():
                return True
        return False

    def _detect_sequence_mode(self, lib: EnhancedLibraryInfo) -> str:
        """识别测序模式，匹配到1.0即返回1.0模式，否则为非1.0模式。"""
        mode_candidates = [
            getattr(lib, "_lane_sj_mode_raw", None),
            getattr(lib, "lane_sj_mode", None),
            getattr(lib, "_current_seq_mode_raw", None),
            getattr(lib, "current_seq_mode", None),
            getattr(lib, "_last_cxms_raw", None),
            getattr(lib, "last_cxms", None),
            getattr(lib, "seq_scheme", None),
            getattr(lib, "test_no", None),
        ]
        for value in mode_candidates:
            if value is None:
                continue
            text = str(value).strip().lower()
            if not text:
                continue
            if "1.0" in text:
                return self.MODE_ONE_POINT_ZERO
        return self.MODE_OTHER

    def _perform_split(self, lib: EnhancedLibraryInfo) -> List[EnhancedLibraryInfo]:
        """执行拆分操作 - 支持多级拆分

        规则：
        - 优先对半拆分（拆成2份）
        - 拆分后每个子文库应尽量满足当前拆分规则阈值
        - 确保每个子文库数据量在合理范围内
        """
        data_amount = float(lib.contract_data_raw)

        rule_label, max_data_per_fragment = self._resolve_split_rule(lib)

        # 计算需要拆分成多少份（优先对半，份数为2的幂次）
        split_count = self._calculate_split_count(
            data_amount=data_amount,
            max_data_per_fragment=max_data_per_fragment,
        )

        logger.debug(
            f"  文库 {lib.origrec} ({data_amount}G) 按规则 {rule_label} 需拆分为 {split_count} 份"
        )

        split_libs = []
        split_data_amount = data_amount / split_count

        original_aidbid = str(
            getattr(lib, "wkaidbid", None) or getattr(lib, "aidbid", None) or ""
        ).strip()
        original_total_contract = float(lib.contract_data_raw or 0.0)

        for i in range(split_count):
            new_lib = copy.deepcopy(lib)
            new_lib.contract_data_raw = split_data_amount
            new_lib.is_split = True
            new_lib.wkissplit = "yes"
            new_lib.split_status = "completed"
            new_lib.wktotalcontractdata = original_total_contract
            new_lib.total_contract_data = original_total_contract
            new_lib.original_library_id = str(getattr(lib, "origrec", "") or "")
            new_lib.fragment_index = i + 1
            new_lib.total_fragments = split_count
            new_lib.fragment_id = f"{new_lib.original_library_id}_F{new_lib.fragment_index:03d}"

            # 拆分后保留wkorigrec/wksid/wkpid原始值，使用wkaidbid区分拆分文库。
            if i == 0 and original_aidbid:
                new_aidbid = original_aidbid
            else:
                new_aidbid = str(uuid.uuid4())
            new_lib.wkaidbid = new_aidbid
            new_lib.aidbid = new_aidbid
            new_lib._split_source_library = lib
            split_libs.append(new_lib)
        
        return split_libs
    
    def _calculate_split_count(
        self,
        data_amount: float,
        max_data_per_fragment: float,
    ) -> int:
        """计算拆分份数（优先对半拆分，份数为2的幂次）

        Args:
            data_amount: 原始数据量（G）
            max_data_per_fragment: 单个拆分片段允许的最大合同数据量

        Returns:
            int: 拆分份数
        """
        if data_amount <= max_data_per_fragment:
            return 1

        # 计算最小需要的份数，使得每份满足单index阈值约束
        min_parts = math.ceil(data_amount / max_data_per_fragment)

        # 向上取整到2的幂次（优先对半拆分原则）
        # 例如：需要3份 -> 取4份（2²）；需要5份 -> 取8份（2³）
        power = math.ceil(math.log2(min_parts))
        split_count = 2 ** power

        # 但如果拆分后每份太小（< min_split_size），则减少拆分份数
        while split_count > 2 and (data_amount / split_count) < self.min_split_size:
            split_count = split_count // 2

        return max(2, split_count)  # 至少拆成2份
