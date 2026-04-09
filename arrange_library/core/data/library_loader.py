"""
从真实CSV数据加载文库信息
创建时间：2025-12-08 17:02:11
更新时间：2025-12-08 17:02:11
"""

from __future__ import annotations

import math
import asyncio
import concurrent.futures
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from loguru import logger

from arrange_library.models.library_info import EnhancedLibraryInfo

# 配置常量
REMARK_RECOGNITION_TIMEOUT: int = 300  # 备注识别超时时间（秒）

# ==========================================================
# 辅助函数
# ==========================================================


def _normalize_machine_type(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    lower = text.lower()

    if not text:
        return text

    if "zm" in lower and "surfseq" in lower:
        return "ZM SURFSeq5000"
    if "surfseq" in lower and "q" in lower:
        return "SURFSEQ-Q"
    if "surfseq" in lower:
        return "SURFSEQ-5000"
    if "novaseq x plus" in lower or "nova seq x plus" in lower:
        return "NovaSeq X Plus"
    return text

# CSV列到数据模型字段的映射
CSV_TO_MODEL_MAPPING: Dict[str, str] = {
    # 英文/规则文档字段
    "origrec": "ORIGREC",
    "sid": "SID",
    "wenku_unique": "WENKU_UNIQUE",
    "sampleid": "SAMPLEID",
    "sampletype": "SAMPLETYPECODE",
    "contractdata": "CONTRACTDATA_RAW",
    "indexseq": "INDEXSEQ",
    "eqtype": "EQTYPE",
    "seqscheme": "SEQSCHEME",
    "seqnotes": "MACHINE_NOTE",  # 上机备注
    "testno": "TESTNO",
    "baleno": "PACKAGE_LANE_NUMBER",
    "bagfcno": "PACKAGE_FC_NUMBER",
    "isaddbalance": "ISADDBALANCE",
    "balancedata": "BALANCEDATA",
    "addtestsremark": "ADDTESTSREMARK",
    "deliverydate": "DELIVERYDATE",
    "outputrate": "OUTPUTRATE",
    "subprojectname": "SUBPROJECTNAME",
    "productline": "PRODUCTLINE",
    "boardnumber": "BOARDNUMBER",
    "peaksize": "PEAKSIZE",
    "peakmap": "PEAKMAP",
    "isprimers": "ISPRIMERS",
    "primersname": "PRIMERSNAME",
    "issuedbatch": "ISSUEDBATCH",
    "special_splits": "SPECIAL_SPLITS",
    "mismatchs_barcodes": "MISMATCHS_BARCODES",
    "lastlaneid": "LASTLANEID",
    "lastcxms": "LASTCXMS",
    "addnumber": "ADDNUMBER",
    "xpd": "XPD",
    "jtb": "JTB",
    "adaptortype": "ADAPTORTYPE",
    "qubit": "QUBITCONC",
    "qpcr": "QPCR_CONCENTRATION",
    "orderdata": "ORDERDATA",
    "pooling": "POOLING",
    "zsclcv": "ZSCLCV",
    "average_q30": "AVERAGE_Q30",
    "laneid": "LANEID",
    "runid": "RUNID",
    "lane_unique": "LANE_UNIQUE",
    # bi_m_merged历史数据表字段（带wk前缀）
    "wkorigrec": "ORIGREC",
    "wksid": "SAMPLEID",
    "wkqubit": "QUBITCONC",
    "wkqpcr": "QPCR_CONCENTRATION",
    "wkcontractdata": "CONTRACTDATA_RAW",
    "wkpeaksize": "PEAKSIZE",
    "wkseqscheme": "SEQSCHEME",
    "wkeqtype": "EQTYPE",
    "wksamplename": "SAMPLE_NAME",
    "wksampletype": "SAMPLETYPECODE",
    "wkdatatype": "DATATYPE",
    "wkspecies": "SPECIES",
    "wkindexseq": "INDEXSEQ",
    "wkindexcode": "INDEX_NUMBER",
    "wkpeakmap": "PEAKMAP",
    "wkcomplexresult": "COMPLEXRESULT",
    "wkmismatchs_barcode": "MISPLACED_BARCODE_DATA",
    "wkdataunit": "DATA_UNIT",
    "wktotalcontractdata": "TOTAL_CONTRACT_DATA",
    "wkoutputrate": "OUTPUTRATE",
    "wkproductname": "PRODUCT_NAME",
    "wkriskbuildflag": "RISK_BUILD_FLAG",
    "wkbaleno": "PACKAGE_LANE_NUMBER",
    "wkisaddbalance": "IS_ADD_BALANCE",
    "wkbalancedata": "BALANCEDATA",
    "wkseqnotes": "MACHINE_NOTE",
    "wkisprimers": "ISPRIMERS",
    "wkprimersname": "PRIMERSNAME",
    "wkboardnumber": "BOARDNUMBER",
    "wkwellnumber": "WELL_NUMBER",
    "wkissuedbatch": "ISSUEDBATCH",
    "wkaddnumber": "ADDNUMBER",
    "wksjnumber": "SJ_NUMBER",
    "wkproductline": "PRODUCTLINE",
    "wkbagfcno": "PACKAGE_FC_NUMBER",
    "wklargeindexori": "LARGE_INDEX_ORI",
    "wkspecialsplits": "SPECIAL_SPLITS",
    "wkdept": "DEPT",
    "wktestno": "TESTNO",
    "wklastupdate": "LAST_UPDATE",
    "wkqpcrchecktime": "QPCR_CHECK_TIME",
    "wkxpd": "XPD",
    "wkadaptorrate": "ADAPTOR_RATE",
    "wkadaptortype": "ADAPTORTYPE",
    "wkcreatedate": "CREATEDATE",
    "wkdeliverydate": "DELIVERY_DATE",
    "wktaskgroupname": "TASK_GROUP_NAME",
    "wkaddtestsremark": "ADDTESTSREMARK",
    "wksubprojectname": "SUBPROJECTNAME",
    "wksampleid": "SAMPLE_ID",
    "wkserialnumber": "SERIAL_NUMBER",
    "wktubenums": "TUBE_NUMS",
    # bi_m_merged历史数据表输出字段（带l前缀，Lane相关）
    "llaneid": "LANEID",
    "llaneorder": "LANE_ORDER",
    "lcxms": "CURRENT_SEQ_MODE",
    "lpjy": "OPERATOR",
    "lorderdata": "ORDERDATA",
    "lratio_1": "RATIO_1",
    "lratio_2": "RATIO_2",
    "lsample1": "SAMPLE1",
    "lrsb": "RSB",
    "lserialno": "SERIAL_NO",
    "lsample2": "SAMPLE2",
    "lvolume1": "VOLUME1",
    "lvolume2": "VOLUME2",
    "lcontainerstate": "CONTAINER_STATE",
    "loutput": "OUTPUT",
    "lzscl": "TRUE_OUTPUT_RATE",
    "lq30": "AVERAGE_Q30",
    "lcfindexseq": "CF_INDEX_SEQ",
    "lrunid": "RUNID",
    "lruntask": "RUN_TASK",
    "leqrunid": "EQ_RUNID",
    "lstatus": "RUN_STATUS",
    "llables": "RUN_LABELS",
    "lsjfs": "LANE_SJ_MODE",
    "lrowingdate": "ROWING_DATE",
    "llastlaneid": "LAST_LANEID",
    "llastcxms": "LAST_CURRENT_SEQ_MODE",
    # 中文列名
    "ORIGREC": "ORIGREC",
    "SID_文库": "SAMPLEID",
    "合同数据量_文库": "CONTRACTDATA_RAW",
    "Index序列": "INDEXSEQ",
    "机器类型": "EQTYPE",
    "PeakSize": "PEAKSIZE",
    "样本类型": "SAMPLETYPECODE",
    "数据类型": "DATATYPE",
    "是否是客户文库": "CUSTOMERLIBRARY",
    "是否双端index测序": "BASETYPE",
    "index碱基数目": "NUMBEROFBASES",
    "包lane编号": "PACKAGE_LANE_NUMBER",
    "包FC编号": "PACKAGE_FC_NUMBER",
    "错位Barcode数据量": "MISPLACED_BARCODE_DATA",
    "是否包lane": "IS_PACKAGE_LANE",
    "交付时间": "DELIVERY_DATE",
    "文库创建时间_文库": "CREATEDATE",
    "加测备注": "ADDTESTSREMARK",
    "上机备注": "MACHINE_NOTE",
    "运营批次": "ORDTASKORI",
    "产线标识": "PRODUCTLINE",
    "物种": "SPECIES",
    "特殊拆分方式": "DATAFLAG",
    # 历史下单与排机信息
    "运营下单合同数据量": "ORDER_CONTRACTDATA_RAW",
    "下单数据量": "ORDER_DATA_AMOUNT",
    "拆分后下单数据量": "SPLIT_ORDER_AMOUNT",
    "排机操作时间": "SCHEDULING_OPERATION_TIME",
    "文库编号": "LIBRARY_CODE",
    "扣减时间": "DEDUCTION_TIME",
    "扣减后交付时间": "ADJUSTED_DELIVERY_TIME",
}

# 数值字段定义
FLOAT_FIELDS = {
    "CONTRACTDATA_RAW",
    "MISPLACED_BARCODE_DATA",
    "PEAKSIZE",
    "ORDER_CONTRACTDATA_RAW",
    "ORDER_DATA_AMOUNT",
    "SPLIT_ORDER_AMOUNT",
    "QPCR_CONCENTRATION",
    "QUBITCONC",
    "BALANCEDATA",
    "OUTPUTRATE",
    "ORDERDATA",
    "XPD",
    "JTB",
    "POOLING",
    "ZSCLCV",
    "AVERAGE_Q30",
}

INT_FIELDS = {
    "NUMBEROFBASES",
    "ADDNUMBER",
}

# 必须存在的字段
ESSENTIAL_FIELDS = {"ORIGREC", "INDEXSEQ", "CONTRACTDATA_RAW", "EQTYPE"}


def _normalize_base_type(value: object) -> str:
    if value is None:
        return "双"
    text = str(value).strip().lower()
    positives = {"是", "yes", "true", "1", "双", "双端"}
    return "双" if text in positives else "单"


def _normalize_numeric(value: object, *, as_int: bool = False) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        if isinstance(value, str) and not value.strip():
            return None
        number = float(str(value))
        if math.isnan(number) or math.isinf(number):
            return None
        return int(number) if as_int else number
    except (ValueError, TypeError):
        return None


def _infer_customer_library(sampletype: str, sample_id: str) -> str:
    if sampletype.startswith("客户") or sample_id.startswith("FKDL"):
        return "是"
    return "否"


def _map_row(row: pd.Series) -> Optional[Dict[str, object]]:
    mapped: Dict[str, object] = {}

    for csv_key, model_key in CSV_TO_MODEL_MAPPING.items():
        if csv_key not in row or pd.isna(row[csv_key]):
            continue

        value = row[csv_key]

        if model_key == "BASETYPE":
            mapped[model_key] = _normalize_base_type(value)
            continue

        if model_key == "EQTYPE":
            mapped[model_key] = _normalize_machine_type(value)
            continue

        if model_key in INT_FIELDS:
            normalized = _normalize_numeric(value, as_int=True)
            if normalized is not None:
                mapped[model_key] = normalized
            continue

        if model_key in FLOAT_FIELDS:
            normalized = _normalize_numeric(value)
            if normalized is not None:
                mapped[model_key] = normalized
            continue

        mapped[model_key] = str(value).strip()

    # 基础字段补充
    for essential in ESSENTIAL_FIELDS:
        if essential not in mapped:
            return None

    if not mapped.get("CUSTOMERLIBRARY"):
        sampletype = str(mapped.get("SAMPLETYPECODE") or "").strip()
        sample_id = str(mapped.get("SAMPLEID") or "").strip()
        mapped["CUSTOMERLIBRARY"] = _infer_customer_library(sampletype, sample_id)

    # 允许PeakSize缺失，默认0
    mapped.setdefault("PEAKSIZE", 0)

    return mapped


def _iter_csv_rows(csv_path: Path, chunksize: int = 5000) -> Iterable[pd.Series]:
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        for _, row in chunk.iterrows():
            yield row


def load_libraries_from_csv(
    csv_path: str | Path,
    limit: Optional[int] = None,
    drop_invalid: bool = True,
    enable_remark_recognition: bool = True,
    allow_missing: bool = True,
) -> List[EnhancedLibraryInfo]:
    """从CSV文件加载文库列表。

    Args:
        csv_path: CSV 文件路径。
        limit: 最多加载的文库数量；None 表示全部加载。
        drop_invalid: 是否忽略无效行；若为 False，将在遇到无效数据时抛出异常。
        enable_remark_recognition: 是否启用备注识别（默认False）
        allow_missing: 缺少文件或无有效数据时是否降级为返回空列表（默认True）

    Returns:
        List[EnhancedLibraryInfo]: 文库对象列表（已过滤未识别退回的文库）。
    """

    path = Path(csv_path)
    if not path.exists():
        if allow_missing:
            logger.warning(f"未找到CSV文件: {path}，已降级为返回空列表")
            return []
        raise FileNotFoundError(f"未找到CSV文件: {path}")

    libraries: List[EnhancedLibraryInfo] = []
    total_rows = 0
    skipped_rows = 0

    logger.info(f"开始从CSV加载文库数据: {path}")

    for row in _iter_csv_rows(path):
        total_rows += 1

        mapped = _map_row(row)
        if not mapped:
            skipped_rows += 1
            if not drop_invalid:
                raise ValueError(f"第{total_rows}行缺少关键字段，无法转换: {row.to_dict()}")
            continue

        try:
            library = EnhancedLibraryInfo.create_from_dict(mapped)
        except ValueError as exc:
            skipped_rows += 1
            if not drop_invalid:
                raise ValueError(f"第{total_rows}行数据非法: {exc}") from exc
            continue

        libraries.append(library)

        if limit is not None and len(libraries) >= limit:
            break

    logger.info(
        "CSV加载完成 - 共读取{}行，成功{}条，跳过{}条".format(
            total_rows, len(libraries), skipped_rows
        )
    )

    if not libraries:
        if allow_missing and drop_invalid:
            logger.warning(f"文件 {path} 未解析出有效文库数据，已降级为返回空列表")
            return []
        raise ValueError(f"在文件 {path} 中未找到有效的文库数据")

    # 如果启用备注识别，进行识别并过滤未识别的数据
    if enable_remark_recognition:
        logger.info("启用备注识别，开始识别上机备注...")
        try:
            # 检查是否在事件循环中
            try:
                loop = asyncio.get_running_loop()
                # 如果已经在事件循环中，需要在新线程中运行
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(_recognize_and_apply_remarks(libraries, csv_path))
                    )
                    valid_libs, unrecognized_libs = future.result(timeout=REMARK_RECOGNITION_TIMEOUT)
            except RuntimeError:
                # 没有运行的事件循环，直接使用asyncio.run
                valid_libs, unrecognized_libs = asyncio.run(
                    _recognize_and_apply_remarks(libraries, csv_path)
                )
            
            logger.info(
                f"备注识别完成 - 有效: {len(valid_libs)}, 未识别退回: {len(unrecognized_libs)}"
            )
            return valid_libs
        except Exception as e:
            logger.error(f"备注识别失败，返回所有文库: {e}")
            logger.debug(traceback.format_exc())
            return libraries

    return libraries


async def _recognize_and_apply_remarks(
    libraries: List[EnhancedLibraryInfo],
    csv_path: str | Path
) -> Tuple[List[EnhancedLibraryInfo], List[EnhancedLibraryInfo]]:
    """
    识别备注并应用结果
    
    Args:
        libraries: 文库列表
        csv_path: CSV文件路径（用于提取备注）
    
    Returns:
        tuple: (有效文库列表, 未识别退回的文库列表)
    """
    from arrange_library.core.data.remark_processor import extract_remarks_from_libraries
    from arrange_library.core.ai.remark_recognizer import RemarkRecognizer
    from arrange_library.core.preprocessing.remark_intent_applier import RemarkIntentApplier
    import os
    
    # 提取备注
    remarks = extract_remarks_from_libraries(libraries)
    
    if not remarks:
        logger.info("未找到有效备注，跳过识别")
        return libraries, []
    
    # 初始化识别器
    recognizer = RemarkRecognizer(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_version=os.getenv('OPENAI_API_VERSION', '2024-08-01-preview'),
        model="gpt-4o",
        confidence_threshold=0.3
    )
    
    # 批量识别
    recognition_results = await recognizer.recognize_remarks_batch(remarks, libraries)
    
    # 应用结果
    applier = RemarkIntentApplier()
    valid_libs, unrecognized_libs = applier.apply_recognition_results(
        libraries, recognition_results
    )
    
    return valid_libs, unrecognized_libs
