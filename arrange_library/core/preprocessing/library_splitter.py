"""
文库拆分器
创建时间：2025-11-20 10:00:00
更新时间：2026-05-11 13:15:14
功能：严格按照《排机规则文档》执行文库拆分，支持多级拆分
"""

import copy
import math
import re
import uuid
from typing import Any, Dict, List, Tuple

from loguru import logger

from arrange_library.models.library_info import EnhancedLibraryInfo
from arrange_library.core.config.scheduling_config import get_library_split_config


class LibrarySplitter:
    """文库拆分器"""

    MODE_ONE_POINT_ONE = "1.1"
    MODE_ONE_POINT_ONE_ALIASES = ("1.1", "1.0")
    MODE_3_6T_NEW = "3.6t-new"
    MODE_LANE_SEQ = "lane seq"
    MODE_OTHER = "other"

    def __init__(self):
        split_config = get_library_split_config()
        self.single_index_non_1_0_threshold = split_config.single_index_non_1_0_threshold_gb
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
            "  拆分规则: 1.1模式文库（兼容旧名1.0）不拆分；3.6T-NEW单对index >{}G 拆分，多对index >{}G 拆分".format(
                self.single_index_non_1_0_threshold,
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
        1. 1.1模式文库（兼容旧名1.0）不拆分
        2. 3.6T-NEW模式按逗号识别index对数
        3. 单对index合同量 >130G 才拆，多对index合同量 >300G 才拆
        4. index序列保持原样，index对数只参与阈值分档
        """
        # 1. 包FC/指定Lane不拆分；带包Lane编号的文库按包Lane规则允许拆分。
        if not self._has_package_lane_binding(lib) and self._has_fixed_lane_binding(lib):
            return False

        # 2. 读取合同量
        try:
            data_amount = float(lib.contract_data_raw or 0)
        except (ValueError, TypeError):
            return False

        if data_amount <= 0:
            return False

        rule_label, split_threshold = self._resolve_split_rule(lib)
        if math.isinf(split_threshold):
            logger.debug(
                "  文库 {} 命中 {}，跳过拆分".format(
                    getattr(lib, "origrec", ""),
                    rule_label,
                )
            )
            return False

        index_pair_count = self._count_index_pairs(lib)
        should_split = data_amount > split_threshold
        if should_split:
            logger.debug(
                "  文库 {} 触发拆分: 合同量={}G, index对数={}, 规则={}, 阈值={}G".format(
                    lib.origrec,
                    round(data_amount, 3),
                    index_pair_count,
                    rule_label,
                    round(split_threshold, 3),
                )
            )
        return should_split

    def _resolve_split_rule(self, lib: EnhancedLibraryInfo) -> Tuple[str, float]:
        """根据业务规则解析拆分阈值。"""
        mode = self._detect_sequence_mode(lib)
        if mode == self.MODE_ONE_POINT_ONE:
            return "mode_1.1_disabled_compat_1.0", math.inf
        if mode != self.MODE_3_6T_NEW:
            return "non_3.6t_new_mode_disabled", math.inf

        index_count = self._count_index_pairs(lib)
        if index_count > 1:
            return "3.6t_new_multi_index_pair", self.multi_index_threshold

        return "3.6t_new_single_index_pair", self.single_index_non_1_0_threshold

    def _is_forced_36t_split_after_mode_1_1_exhausted(
        self,
        lib: EnhancedLibraryInfo,
        data_amount: float,
    ) -> bool:
        """兼容旧调用点；拆分必须统一走单对130G/多对300G阈值，不再强制绕过。"""
        return False

    def _is_single_end_index(self, lib: EnhancedLibraryInfo) -> bool:
        """历史兼容接口：没有逗号分隔的都按单对index处理。"""
        index_seq = str(getattr(lib, "index_seq", "") or "").strip()
        return "," not in index_seq

    def _count_index_pairs(self, lib: EnhancedLibraryInfo) -> int:
        """计算index对数

        规则说明：
        - index序列中被逗号（,）隔开的叫多对index
        - 是否有分号不参与对数判断；没有逗号也叫单对index
        
        例如：ATCG;GCTA,TTAA;GGCC 表示2对index
        """
        index_seq = getattr(lib, 'index_seq', '') or ''
        if not index_seq:
            return 1
        
        pairs = [seg.strip() for seg in index_seq.split(',') if seg.strip()]
        return max(len(pairs), 1)

    def _has_package_lane_binding(self, lib: EnhancedLibraryInfo) -> bool:
        """是否存在包Lane编号。"""
        fields = [
            getattr(lib, "package_lane_number", None),
            getattr(lib, "baleno", None),
        ]
        for value in fields:
            if value is None:
                continue
            if str(value).strip():
                return True
        return False

    def _has_fixed_lane_binding(self, lib: EnhancedLibraryInfo) -> bool:
        """包FC/指定Lane/指定FC文库不进行拆分。"""
        fields = [
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
        """识别测序模式，区分1.1模式族与3.6T-NEW模式。

        拆分规则只关心当前排机上下文，不应被历史字段 ``llastcxms`` 干扰。
        ``llastcxms`` 仅用于 1.1 模式分流和第二轮候选识别，不参与拆分模式判断。
        """
        mode_candidates = [
            getattr(lib, "_current_seq_mode_raw", None),
            getattr(lib, "selected_seq_mode", None),
            getattr(lib, "current_seq_mode", None),
            getattr(lib, "lcxms", None),
            getattr(lib, "_lane_sj_mode_raw", None),
            getattr(lib, "lane_sj_mode", None),
            getattr(lib, "seq_scheme", None),
            getattr(lib, "test_no", None),
        ]
        for value in mode_candidates:
            if value is None:
                continue
            text = str(value).strip().lower()
            if not text:
                continue
            if any(
                self._contains_mode_token(text, mode_token)
                for mode_token in self.MODE_ONE_POINT_ONE_ALIASES
            ):
                return self.MODE_ONE_POINT_ONE
            if self._contains_mode_token(text, self.MODE_3_6T_NEW):
                return self.MODE_3_6T_NEW
            if self._contains_mode_token(text, self.MODE_LANE_SEQ):
                return self.MODE_OTHER

        if self._is_lane_seq_library(lib):
            return self.MODE_OTHER
        if self._is_default_3_6t_new_library(lib):
            return self.MODE_3_6T_NEW
        return self.MODE_OTHER

    @staticmethod
    def _contains_mode_token(text: str, mode_token: str) -> bool:
        """判断文本中是否包含独立的模式标记，如1.0或1.1。"""
        return re.search(rf"(?<!\d){re.escape(mode_token.lower())}(?!\d)", text) is not None

    def _is_lane_seq_library(self, lib: EnhancedLibraryInfo) -> bool:
        """判断是否为 lane seq 策略，避免误按 3.6T-NEW 处理。"""
        strategy_candidates = [
            getattr(lib, "seq_scheme", None),
            getattr(lib, "_seq_scheme_raw", None),
            getattr(lib, "test_no", None),
        ]
        for value in strategy_candidates:
            if value is None:
                continue
            text = str(value).strip().lower()
            if not text:
                continue
            if self._contains_mode_token(text, "10+24"):
                return True
            if self._contains_mode_token(text, self.MODE_LANE_SEQ):
                return True
        return False

    def _is_default_3_6t_new_library(self, lib: EnhancedLibraryInfo) -> bool:
        """按当前配置口径为缺省模式的 X Plus 文库兜底到 3.6T-NEW。"""
        test_code = getattr(lib, "test_code", None)
        try:
            normalized_test_code = int(float(test_code))
        except (TypeError, ValueError):
            normalized_test_code = None

        if normalized_test_code == 1595 and not self._is_lane_seq_library(lib):
            return True

        test_no = str(getattr(lib, "test_no", "") or "").strip().lower()
        eq_type = str(getattr(lib, "eq_type", "") or "").strip().lower()
        if self._is_lane_seq_library(lib):
            return False

        xplus_keywords = (
            "novaseq x plus",
            "nova x plus",
        )
        machine_keywords = (
            "nova x-25b",
            "nova x-10b",
        )

        return any(keyword in test_no for keyword in xplus_keywords) or any(
            keyword in eq_type for keyword in machine_keywords
        )

    def _perform_split(self, lib: EnhancedLibraryInfo) -> List[EnhancedLibraryInfo]:
        """执行拆分操作 - 支持多级拆分

        规则：
        - index序列不拆改
        - 单对index按130G阈值拆，多对index按300G阈值拆
        - 确保每个子文库数据量在合理范围内
        """
        data_amount = float(lib.contract_data_raw)

        rule_label, split_threshold = self._resolve_split_rule(lib)

        split_count = self._calculate_split_count(
            data_amount=data_amount,
            index_pair_count=self._count_index_pairs(lib),
            split_threshold=split_threshold,
        )
        if split_count <= 1:
            return [lib]

        logger.debug(
            f"  文库 {lib.origrec} ({data_amount}G) 按规则 {rule_label} 需拆分为 {split_count} 份"
        )

        split_libs = []
        split_data_amount = data_amount / split_count
        split_single_index_data = self._split_optional_float_value(
            getattr(lib, "single_index_data", None),
            split_count,
        )
        split_ten_bp_data = self._split_optional_float_value(
            getattr(lib, "ten_bp_data", None),
            split_count,
        )

        original_aidbid = str(
            getattr(lib, "wkaidbid", None) or getattr(lib, "aidbid", None) or ""
        ).strip()
        raw_total_contract = (
            getattr(lib, "wktotalcontractdata", None)
            if getattr(lib, "wktotalcontractdata", None) not in (None, "")
            else getattr(lib, "total_contract_data", None)
        )
        try:
            original_total_contract = float(raw_total_contract)
        except (TypeError, ValueError):
            original_total_contract = float(lib.contract_data_raw or 0.0)

        for i in range(split_count):
            new_lib = copy.deepcopy(lib)
            new_lib.contract_data_raw = split_data_amount
            new_lib.single_index_data = split_single_index_data
            new_lib.ten_bp_data = split_ten_bp_data
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
            source_origrec_key = str(
                getattr(lib, "_source_origrec_key", None)
                or getattr(lib, "_origrec_key", None)
                or getattr(lib, "origrec", "")
                or ""
            ).strip()
            new_lib._source_origrec_key = source_origrec_key
            new_lib._detail_output_key = str(new_lib.fragment_id or new_aidbid or source_origrec_key).strip()
            split_libs.append(new_lib)
        
        return split_libs

    @staticmethod
    def _split_optional_float_value(value: Any, split_count: int) -> Any:
        """按拆分份数均分可选数值字段，空值保持不变。"""
        if value in (None, ""):
            return value
        try:
            return float(value) / split_count
        except (TypeError, ValueError):
            return value
    
    def _calculate_split_count(
        self,
        data_amount: float,
        index_pair_count: int,
        split_threshold: float,
    ) -> int:
        """计算拆分份数。

        index序列不拆改；单对按130G阈值，多对按300G阈值，把每片拆到不再触发。
        """
        if math.isinf(split_threshold):
            return 1
        if data_amount <= 0:
            return 1

        if data_amount <= split_threshold:
            return 1

        split_count = math.ceil(data_amount / split_threshold)

        # 若极小阈值或异常输入导致拆分后每份过小，则回退到能满足最小保留量的最大份数。
        while split_count > 2 and (data_amount / split_count) < self.min_split_size:
            split_count -= 1

        return max(2, split_count)
