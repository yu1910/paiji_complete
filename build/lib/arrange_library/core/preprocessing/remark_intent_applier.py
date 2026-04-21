"""
备注意图应用器 - 将识别结果应用到文库对象（v2）
创建时间：2025-11-17
更新时间：2026-01-16 14:49:06
"""
from typing import Dict, List, Optional, Tuple

from loguru import logger

# from liblane_paths import setup_liblane_paths

# setup_liblane_paths()

from arrange_library.models.library_info import EnhancedLibraryInfo
from arrange_library.models.remark_recognition_v2 import CommandItem, RemarkRecognitionResultV2


class RemarkIntentApplier:
    """备注意图应用器（v2）"""

    def __init__(self) -> None:
        logger.info("备注意图应用器(v2)初始化完成")

    def apply_recognition_results(
        self,
        libraries: List[EnhancedLibraryInfo],
        recognition_results: Dict[str, RemarkRecognitionResultV2],
    ) -> Tuple[List[EnhancedLibraryInfo], List[EnhancedLibraryInfo]]:
        """
        应用识别结果到文库对象

        Args:
            libraries: 文库列表
            recognition_results: 识别结果字典 {library_id: RemarkRecognitionResultV2}

        Returns:
            tuple: (有效文库列表, 未识别退回的文库列表)
        """
        valid_libraries: List[EnhancedLibraryInfo] = []
        unrecognized_libraries: List[EnhancedLibraryInfo] = []

        for library in libraries:
            library_id = library.origrec
            result = recognition_results.get(library_id)

            if not result:
                valid_libraries.append(library)
                continue

            if not hasattr(library, "remark_recognition_status"):
                library.remark_recognition_status = None
                library.remark_recognition_reason = None

            if not result.is_recognized:
                library.remark_recognition_status = "unrecognized"
                library.remark_recognition_reason = result.error_message or "未识别退回"
                unrecognized_libraries.append(library)
                logger.debug(
                    f"文库 {library_id} 未识别，标记为退回: {library.remark_recognition_reason}"
                )
                continue

            is_need = result.is_need
            explain = result.explain or ""

            if is_need == "忽略":
                library.remark_recognition_status = "ignored"
                library.remark_recognition_reason = explain or "忽略"
                valid_libraries.append(library)
                continue

            if is_need == "人工识别":
                library.remark_recognition_status = "manual"
                library.remark_recognition_reason = explain or "人工识别"
                valid_libraries.append(library)
                continue

            library.remark_recognition_status = "recognized"
            library.remark_recognition_reason = None

            if explain:
                self._append_remark(library, f"备注意图:{explain}")

            self._apply_commands(library, result.commands)
            valid_libraries.append(library)

        logger.info(
            f"意图应用完成 - 有效: {len(valid_libraries)}, 未识别退回: {len(unrecognized_libraries)}"
        )
        return valid_libraries, unrecognized_libraries

    def _apply_commands(self, library: EnhancedLibraryInfo, commands: List[CommandItem]) -> None:
        for cmd in commands:
            try:
                self._apply_command(library, cmd)
            except Exception as exc:
                logger.warning(f"应用指令失败: {cmd.type} - {exc}")

    def _apply_command(self, library: EnhancedLibraryInfo, cmd: CommandItem) -> None:
        cmd_type = (cmd.type or "").strip()
        params = cmd.params or {}

        if not cmd_type:
            return

        if cmd_type == "PACK_LANE":
            library.is_package_lane = "是"
            lane_number = (
                params.get("with_batch_id")
                or params.get("package_lane_number")
                or params.get("lane_number")
            )
            if lane_number:
                library.package_lane_number = str(lane_number)
            self._append_remark(library, "AI:包lane")
            return

        if cmd_type == "EXCLUDE_1_0_MODE":
            setattr(library, "not_for_new_mode", True)
            setattr(library, "exclude_new_mode", True)
            return

        if cmd_type == "SET_SPECIAL_SPLITS":
            special_splits = params.get("special_splits")
            if special_splits:
                library.special_splits = str(special_splits)
            return

        if cmd_type == "SET_RUN_CYCLE":
            run_cycle = params.get("normalized") or params.get("raw")
            if run_cycle:
                library.run_cycle = str(run_cycle)
            return

        if cmd_type == "ADD_BALANCE_LIBRARY":
            setattr(library, "is_add_balance", "是")
            balance_percent = params.get("balance_percent")
            if balance_percent is not None:
                self._append_remark(library, f"AI:平衡文库{balance_percent}%")
            return

        if cmd_type == "GROUP_WITH_BATCH":
            batch_id = params.get("batch_id")
            relation = str(params.get("relation", ""))
            if batch_id:
                if any(key in relation for key in ["同lane", "凑lane", "包lane"]):
                    self._append_remark(library, f"同批次:{batch_id}")
                else:
                    self._append_remark(library, f"关联批次:{batch_id};关系:{relation}")
            return

        if cmd_type == "MANUAL_CHECK":
            items = params.get("items") if isinstance(params.get("items"), list) else []
            if items:
                self._append_remark(library, "人工核对:" + ";".join([str(i) for i in items]))
            return

        if cmd_type in {
            "PACK_FC",
            "REQUIRE_SAME_RUN",
            "REQUIRE_SAME_LANE",
            "REQUIRE_SEPARATE_LANE",
            "REQUIRE_SEPARATE_RUN",
            "REQUIRE_SAME_FC",
            "SET_FC_LANE_MAPPING",
            "SET_PHIX",
            "SET_CONCENTRATION",
            "SET_DATA_AMOUNT_REQUIREMENT",
            "SET_POOLING_PLAN",
            "SET_INSTRUMENT_PLUGIN",
            "DELIVERY_REQUIREMENT",
        }:
            self._append_remark(library, f"AI指令:{cmd_type} {params}")
            return

        self._append_remark(library, f"AI未落地指令:{cmd_type} {params}")

    @staticmethod
    def _append_remark(library: EnhancedLibraryInfo, text: str) -> None:
        if not text:
            return
        if library.remarks:
            library.remarks = f"{library.remarks}; {text}"
        else:
            library.remarks = text
