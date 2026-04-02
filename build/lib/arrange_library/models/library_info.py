"""
LibLane V2 文库信息数据模型
基于31字段真实业务数据结构设计
创建时间：2025-12-08 17:02:11
更新时间：2026-01-30 15:00:00

变更记录：
- 2026-01-30: 字段映射精简，以表中实际字段名为准
             - 参考表: data/2026-01-30_v1_standardized_lane_output_v5.csv
             - 每个映射只保留表中存在的字段名，避免误映射
             - 表中不存在的字段保留原映射并标注"表中无此字段，保留兼容"
- 2025-12-24: 新增jjbj（碱基不均衡）、single_index_data、ten_bp_data字段
             新增is_base_imbalance()方法用于判断碱基不均衡文库
             新增product_name、special_splits、sample_number_prefix、large_index_ori字段
             新增is_10x_library()、needs_special_split()、get_special_split_type()方法
             优化is_clinical_by_code()方法使用sample_number_prefix字段
"""

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List, Union

from loguru import logger


# 枚举类型定义
class DataType(Enum):
    """数据类型分类"""
    CLINICAL = "临检"
    YC = "YC" 
    OTHER = "其他"


class MachineType(Enum):
    """测序机器类型"""
    NOVA_X_10B = "Nova X-10B"
    NOVA_X_25B = "Nova X-25B"
    NOVASEQ = "Novaseq"
    NOVASEQ_T7 = "Novaseq-T7"  # 新增T7机器类型
    T7 = "T7"
    SURFSEQ_5000 = "SURFSEQ-5000"
    NOVASEQ_X_PLUS = "NovaSeq X Plus"
    ZM_SURFSEQ_5000 = "ZM SURFSeq5000"
    UNKNOWN = "Unknown"  # 未识别的机器类型


class IndexType(Enum):
    """Index类型"""
    SINGLE = "单"
    DOUBLE = "双"


class ProductLine(Enum):
    """产线类型"""
    S = "S"
    # 其他产线类型可根据实际数据扩展


class DataVolumeType(Enum):
    """数据量类型"""
    SMALL = "小数量"
    # 其他数据量类型可根据实际数据扩展


class LabType(Enum):
    """实验室类型分类"""
    CUSTOMER_SINGLE_CELL = "客户-单细胞文库"
    CUSTOMER_PCR_PRODUCT = "客户-PCR产物"
    CUSTOMER_GENOME_RESEQUENCING = "客户-动植物全基因组重测序文库"
    # 更多实验室类型可根据实际数据扩展


class AddTestRemark(Enum):
    """加测备注类型"""
    RETEST = "补测"
    ADD_TEST = "加测"
    MIXED = "混合"
    EMPTY = "-"
    # 其他类型根据实际数据扩展


@dataclass
class EnhancedLibraryInfo:
    """增强的文库信息模型 - 基于31字段业务数据"""
    
    # === 基础标识信息 ===
    origrec: str                           # ORIGREC - 文库唯一标识符
    sample_id: str                         # SAMPLEID - 样本编号
    sample_type_code: str                  # SAMPLETYPECODE - 样本类型编码
    
    # === 数据分类信息 ===
    data_type: str                         # DATATYPE - 数据类型(临检/YC/其他)
    customer_library: str                  # CUSTOMERLIBRARY - 客户文库标识(是/否)
    
    # === Index相关信息 ===
    base_type: str                         # BASETYPE - Index类型(单/双)
    number_of_bases: int                   # NUMBEROFBASES - Index碱基数目
    index_number: int                      # INDEXNUMBER - Index类型编号  
    index_seq: str                         # INDEXSEQ - Index序列
    
    # === 测试和产线信息 ===
    add_tests_remark: str                  # ADDTESTSREMARK - 加测备注
    product_line: str                      # PRODUCTLINE - 产线
    peak_size: int                         # PEAKSIZE - 峰值(200-600不等)
    
    # === 机器和数据量信息 ===
    eq_type: str                          # EQTYPE - 机器类型
    contract_data_raw: float              # CONTRACTDATA_RAW - 合同数据量
    test_code: Optional[int]              # TESTCODE - 测序编码
    test_no: str                          # TESTNO - 测序策略
    
    # === 项目和合同信息 ===
    sub_project_name: str                 # SUBPROJECTNAME - 合同名称
    
    # === 时间信息 ===
    create_date: str                      # CREATEDATE - 创建日期
    delivery_date: str                    # DELIVERYDATE - 交付日期
    
    # === 文库分类信息 ===
    lab_type: str                         # LABTYPE - 是否客户文库+文库类型
    data_volume_type: str                 # DATAVOLUMETYPE - 数据量类型
    board_number: str                     # BOARDNUMBER - 版号
    # === 唯一标识 ===
    wenku_unique: Optional[str] = None    # 文库级唯一标识（origrec+sid）
    lane_unique: Optional[str] = None     # lane级唯一标识（runid+laneid）
    
    # === 拆分和备注信息 ===
    data_flag: Optional[str] = None       # DATAFLAG - 是否拆分
    add_test_note: Optional[str] = None   # 加测备注 - 加测优先级标识
    run_cycle: Optional[str] = None       # RunCycle - 测序参数备注
    
    # === Lane相关信息 ===
    is_package_lane: Optional[str] = None    # 是否包lane
    package_lane_number: Optional[str] = None # 包lane编号
    package_fc_number: Optional[str] = None   # 包FC编号
    
    # === 实验备注和质量信息 ===
    machine_note: Optional[str] = None       # 上机备注
    qpcr_concentration: Optional[float] = None # QPCR摩尔浓度
    misplaced_barcode_data: Optional[float] = None # 错位Barcode数据量
    
    # === 新增关键业务字段 (2025-08-12) ===
    qpcr_molar: Optional[float] = None          # QPCRMOLAR - QPCR摩尔浓度(新)
    qubit_concentration: Optional[float] = None  # QUBITCONC - Qubit浓度
    species: Optional[str] = None               # SPECIES - 物种信息
    seq_scheme: Optional[str] = None            # SEQSCHEME - 测序方案
    ord_task_ori: Optional[str] = None          # ORDTASKORI - 原始任务订单
    business_line: Optional[str] = None         # BUSINESSLINE - 业务线
    output_rate: Optional[float] = None         # OUTPUTRATE - 产出率
    remarks: Optional[str] = None               # REMARKS - 备注信息
    
    # === 历史下单与排机信息字段 (基于history_data 80+字段) ===
    order_contract_data_raw: Optional[float] = None   # ORDER_CONTRACTDATA_RAW - 运营下单合同数据量
    order_data_amount: Optional[float] = None         # ORDER_DATA_AMOUNT - 下单数据量
    split_order_amount: Optional[float] = None        # SPLIT_ORDER_AMOUNT - 拆分后下单数据量
    schedule_operation_time: Optional[str] = None     # SCHEDULING_OPERATION_TIME - 排机操作时间
    library_code: Optional[str] = None                # LIBRARY_CODE - 文库编号
    deduction_time: Optional[str] = None              # DEDUCTION_TIME - 扣减时间
    adjusted_delivery_time: Optional[str] = None      # ADJUSTED_DELIVERY_TIME - 扣减后交付时间
    
    # === 智能拆分状态字段 (2025-09-10) ===
    is_split: Optional[bool] = None             # 是否已拆分
    split_status: Optional[str] = None          # 拆分状态：pending/processing/completed/failed
    original_library_id: Optional[str] = None  # 原始文库ID（用于拆分片段）
    fragment_id: Optional[str] = None          # 片段ID（用于拆分片段）
    fragment_index: Optional[int] = None       # 片段序号
    total_fragments: Optional[int] = None      # 总片段数
    split_reason: Optional[str] = None         # 拆分原因
    split_strategy: Optional[str] = None       # 拆分策略
    split_timestamp: Optional[str] = None      # 拆分时间戳
    
    # === 备注识别状态字段 (2025-11-17) ===
    remark_recognition_status: Optional[str] = None  # 备注识别状态：recognized/unrecognized/pending
    remark_recognition_reason: Optional[str] = None  # 备注识别原因（未识别时记录原因）

    # === 规则文档新增/缺失字段映射 (排机规则文档 972-1017) ===
    sid: Optional[str] = None                        # SID_文库
    issued_batch: Optional[str] = None               # 运营批次（issuedbatch）
    is_add_balance: Optional[str] = None             # 是否加平衡文库（isaddbalance）
    balance_data: Optional[float] = None             # 平衡文库数据量（balancedata）
    peak_map: Optional[str] = None                   # 峰图描述（peakmap）
    is_primers: Optional[str] = None                 # 是否提供测序引物（isprimers）
    primers_name: Optional[str] = None               # 测序引物名称（primersname）
    last_laneid: Optional[str] = None                # 上次laneid（lastlaneid）
    last_cxms: Optional[str] = None                  # 上次测序模式（lastcxms）
    add_number: Optional[int] = None                 # 加测次数（addnumber）
    xpd: Optional[float] = None                      # 小片段占比（xpd）
    jtb: Optional[float] = None                      # 接头比值（jtb）
    adaptor_type: Optional[str] = None               # 接头类型（adaptortype）
    pooling: Optional[float] = None                  # pooling系数（pooling）
    zsclcv: Optional[float] = None                   # 真实产率cv（zsclcv）
    average_q30: Optional[float] = None              # q30（average_q30）
    
    # === loutput预测模型所需字段 (2025-12-23新增) ===
    sj_number: Optional[int] = None                  # 上机次数（wksjnumber）
    adaptor_rate: Optional[float] = None             # 接头比值（wkadaptorrate）
    complex_result: Optional[str] = None             # 库检综合结果（wkcomplexresult）
    data_unit: Optional[str] = None                  # 数据量单位（wkdataunit）
    risk_build_flag: Optional[str] = None            # 风险建库标识（wkriskbuildflag）
    
    # === 碱基不均衡相关字段 (2025-12-24新增) ===
    jjbj: Optional[str] = None                       # 碱基不均衡标识（是/否）
    single_index_data: Optional[float] = None        # 单端Index数据量
    ten_bp_data: Optional[float] = None              # 10bp Index数据量
    
    # === 产品和特殊处理字段 (2025-12-24新增) ===
    product_name: Optional[str] = None               # 产品名称（用于识别特殊文库）
    special_splits: Optional[str] = None             # 特殊拆分标识（10x_cellranger/UMI）
    sample_number_prefix: Optional[str] = None       # 样本编号前缀（FDYE/FDYT等临检编码）
    large_index_ori: Optional[str] = None            # 超长Index原始标识
    
    def __post_init__(self):
        """数据初始化后的验证和转换"""
        # 转换字符串类型的数值字段
        if isinstance(self.peak_size, str) and self.peak_size.isdigit():
            self.peak_size = int(self.peak_size)
        
        if isinstance(self.contract_data_raw, str):
            try:
                self.contract_data_raw = float(self.contract_data_raw)
            except ValueError:
                self.contract_data_raw = 0.0
                
        if isinstance(self.test_code, str) and self.test_code.isdigit():
            self.test_code = int(self.test_code)

        if isinstance(self.balance_data, str):
            try:
                self.balance_data = float(self.balance_data)
            except ValueError:
                self.balance_data = None

        if isinstance(self.add_number, str):
            try:
                self.add_number = int(float(self.add_number))
            except (TypeError, ValueError):
                self.add_number = None

        if isinstance(self.xpd, str):
            try:
                self.xpd = float(self.xpd)
            except (TypeError, ValueError):
                self.xpd = None

        if isinstance(self.jtb, str):
            try:
                self.jtb = float(self.jtb)
            except (TypeError, ValueError):
                self.jtb = None

        if isinstance(self.pooling, str):
            try:
                self.pooling = float(self.pooling)
            except (TypeError, ValueError):
                self.pooling = None

        if isinstance(self.zsclcv, str):
            try:
                self.zsclcv = float(self.zsclcv)
            except (TypeError, ValueError):
                self.zsclcv = None

        if isinstance(self.average_q30, str):
            try:
                self.average_q30 = float(self.average_q30)
            except (TypeError, ValueError):
                self.average_q30 = None
    
    def parse_index_sequences(self) -> List[str]:
        """解析复杂的Index序列格式
        
        处理格式如：
        - 单个序列: "ATGGCTTGTG;GAATGTTGTG"
        - 多个序列: "GAGGAGTG;GGTACTTA,AAACATCT;GACATATC,CACGCCGT;ACCTCGCA"
        
        Returns:
            List[str]: 解析后的Index序列列表
        """
        if not self.index_seq:
            return []
        if self.index_seq.strip() == "":
            return []
        
        # 按逗号分割多个序列组合
        sequence_groups = self.index_seq.split(',')
        
        parsed_sequences = []
        for group in sequence_groups:
            group = group.strip()
            if group:
                # 处理用分号分隔的序列对
                if ';' in group:
                    sequences = group.split(';')
                    parsed_sequences.append(f"{sequences[0]};{sequences[1]}")
                else:
                    parsed_sequences.append(group)
        
        return parsed_sequences
    
    def is_customer_library(self) -> bool:
        """判断是否为客户文库
        
        优先使用customer_library字段（值为"是"/"否"）
        如果该字段不存在或为空，则通过sampletype字段判断（以"客户-"开头）
        
        Returns:
            bool: True表示客户文库，False表示内部文库
        """
        customer_flag = str(self.customer_library or "").strip()
        if customer_flag:
            flag_upper = customer_flag.upper()
            if customer_flag in {"是", "客户"} or flag_upper in {"Y", "YES", "TRUE"}:
                return True
        
        sampletype = self.sample_type_code or getattr(self, "sampletype", "") or ""
        if str(sampletype).startswith("客户"):
            return True
        
        sample_id = self.sample_id or ""
        if str(sample_id).startswith("FKDL"):
            return True
        
        return False
    
    def is_clinical_by_code(self) -> bool:
        """
        根据样本编号判断是否为临检样本
        规则文档（AI排机-整体说明.csv）：
        - 文库编码前4位是FDYE/FDYT/FDYG/FDYP/FDYK/FDYX/FICR/FIPM
        - 国际临检：英国临检文库首字母为E，美国临检文库首字母为C
        
        优先使用sample_number_prefix字段（预处理好的样本编号前缀）
        """
        # 国内临检编码
        clinical_codes = ['FDYE', 'FDYT', 'FDYG', 'FDYP', 'FDYK', 'FDYX', 'FICR', 'FIPM']
        
        # 优先使用sample_number_prefix字段
        if self.sample_number_prefix:
            prefix = self.sample_number_prefix.upper()
            if prefix in clinical_codes:
                return True
        
        # 回退到sample_id判断
        if self.sample_id and len(self.sample_id) >= 4:
            prefix_4 = self.sample_id[:4].upper()
            if prefix_4 in clinical_codes:
                return True
            
            # 国际临检：英国E开头，美国C开头
            first_letter = self.sample_id[0].upper()
            if first_letter in ['E', 'C']:
                return True
        
        return self.data_type == "临检"
    
    def is_yc_library(self) -> bool:
        """
        判断是否为YC文库
        规则文档：文库编码前4位是FKDL，且子项目名称带YC的文库
        注：由于数据模型限制，目前通过data_type和sample_id综合判断
        """
        if not self.sample_id:
            return self.data_type == "YC"
        
        # 检查编码前4位是否为FKDL
        is_fkdl = len(self.sample_id) >= 4 and self.sample_id[:4].upper() == 'FKDL'
        
        # 检查是否带YC标识
        # 方法1：检查data_type字段
        has_yc_in_type = self.data_type == "YC"
        
        # 方法2：检查batch_code是否包含YC（如果有的话）
        has_yc_in_batch = hasattr(self, 'batch_code') and self.batch_code and 'YC' in self.batch_code.upper()
        
        return is_fkdl and (has_yc_in_type or has_yc_in_batch)
    
    def is_s_level_customer(self) -> bool:
        """
        判断是否为S级客户
        规则文档：项目名称带SJ标识
        
        使用sub_project_name字段进行判断，如果包含SJ标识则认为是S级客户
        
        Returns:
            bool: True表示是S级客户
        """
        if not self.sub_project_name:
            return False
        
        # 检查项目名称中是否包含SJ标识（不区分大小写）
        return 'SJ' in self.sub_project_name.upper()
    
    def is_large_data_library(self) -> bool:
        """
        判断是否为大数据量文库
        规则文档（AI排机-整体说明.csv）：
        - YC文库：单个index合同数据量 > 70G
        - S级客户文库：合同数据量 > 70G
        - 临检文库：大于70G与小于70G可同Lane排机
        """
        return self.contract_data_raw > 70.0
    
    def is_small_data_library(self) -> bool:
        """
        判断是否为小数据量文库
        规则文档：合同数据量 ≤ 70G的文库
        """
        return self.contract_data_raw <= 70.0
    
    def is_base_imbalance(self) -> bool:
        """
        判断是否为碱基不均衡文库
        
        优先使用数据中的jjbj字段（已经预处理好的碱基不均衡标识）
        如果jjbj字段不存在，则根据文库类型关键词判断
        
        Returns:
            bool: True表示是碱基不均衡文库
        """
        # 优先使用jjbj字段
        if self.jjbj is not None and str(self.jjbj).strip() != '':
            return str(self.jjbj).strip() == '是'
        
        # 回退到关键词判断（兼容旧数据）
        lab_type = self.lab_type or ''
        sample_type = self.sample_type_code or ''
        product = self.product_name or ''
        combined = f"{lab_type} {sample_type} {product}".lower()
        
        # 碱基不均衡关键词
        imbalance_keywords = [
            'atac', 'cut tag', 'methylation', 'small rna', '单细胞', '10x',
            '简化基因组', 'gbs', 'rad', 'circrna', '甲基化', 'rrbs',
            'ribo-seq', 'em-seq', '墨卓', 'visium', 'fixed rna', 'mobidrop'
        ]
        
        for keyword in imbalance_keywords:
            if keyword in combined:
                return True
        
        return False
    
    def is_10x_library(self) -> bool:
        """
        判断是否为10X Genomics文库
        
        基于product_name和sample_type_code判断
        
        Returns:
            bool: True表示是10X文库
        """
        product = (self.product_name or '').lower() if self.product_name else ''
        sample_type = (self.sample_type_code or '').lower() if self.sample_type_code else ''
        combined = f"{product} {sample_type}"
        
        # 10X相关关键词
        if '10x' in combined or '10x genomics' in combined:
            return True
        if 'single cell' in combined or '单细胞' in combined:
            return True
        
        # 检查特殊拆分标识（处理nan值）
        if self.special_splits is not None and not (isinstance(self.special_splits, float) and str(self.special_splits) == 'nan'):
            splits_str = str(self.special_splits).lower()
            if '10x' in splits_str:
                return True
        
        return False
    
    def needs_special_split(self) -> bool:
        """
        判断是否需要特殊拆分处理
        
        基于special_splits字段判断（10x_cellranger/UMI等）
        
        Returns:
            bool: True表示需要特殊拆分
        """
        if self.special_splits is None:
            return False
        # 处理nan值
        splits_str = str(self.special_splits).strip()
        if splits_str and splits_str.lower() != 'nan':
            return True
        return False
    
    def get_special_split_type(self) -> Optional[str]:
        """
        获取特殊拆分类型
        
        Returns:
            特殊拆分类型（10x_cellranger/UMI等），无特殊拆分返回None
        """
        if self.special_splits is None:
            return None
        splits_str = str(self.special_splits).strip()
        if splits_str and splits_str.lower() != 'nan':
            return splits_str
        return None
    
    def get_data_type_enum(self) -> DataType:
        """获取数据类型的枚举值"""
        type_mapping = {
            "临检": DataType.CLINICAL,
            "YC": DataType.YC,
            "其他": DataType.OTHER
        }
        return type_mapping.get(self.data_type, DataType.OTHER)
    
    def get_machine_type_enum(self) -> MachineType:
        """获取机器类型的枚举值
        
        Returns:
            MachineType: 机器类型枚举值，未识别的类型返回MachineType.UNKNOWN
            
        Raises:
            ValueError: 当机器类型为空时
            
        Note:
            未知机器类型会记录警告并返回UNKNOWN，保留原始eq_type值不变
        """
        if not self.eq_type:
            raise ValueError(f"文库 {self.origrec} 的机器类型不能为空")
        
        eq_type_lower = self.eq_type.lower().strip()
        
        # 使用字典映射优化，按优先级从具体到通用排序
        # 注意：匹配顺序很重要，必须先匹配更具体的类型
        # 对于多关键词匹配，需要同时满足所有关键词
        machine_patterns = [
            # 最具体的匹配（必须放在前面）
            # ZM SURFSeq需要同时包含zm和surfseq
            (lambda s: "zm" in s and "surfseq" in s, MachineType.ZM_SURFSEQ_5000),
            # Novaseq X Plus需要包含完整短语
            (lambda s: "novaseq x plus" in s or "nova seq x plus" in s, MachineType.NOVASEQ_X_PLUS),
            # Nova X-10B或10B
            (lambda s: "nova x-10b" in s or ("10b" in s and "25b" not in s), MachineType.NOVA_X_10B),
            # Nova X-25B或25B
            (lambda s: "nova x-25b" in s or "25b" in s, MachineType.NOVA_X_25B),
            # 通用匹配（放在后面）
            (lambda s: "novaseq" in s and "x plus" not in s, MachineType.NOVASEQ),
            (lambda s: "surfseq" in s and "zm" not in s, MachineType.SURFSEQ_5000),
            (lambda s: "t7" in s, MachineType.T7),
        ]
        
        # 按顺序检查匹配
        for pattern_func, machine_type in machine_patterns:
            if pattern_func(eq_type_lower):
                return machine_type
        
        # 未匹配到任何类型，记录警告并返回UNKNOWN（保留原始机器类型信息）
        logger.warning(f"文库 {self.origrec} 的机器类型 '{self.eq_type}' 未识别，标记为UNKNOWN")
        return MachineType.UNKNOWN
    
    def get_index_type_enum(self) -> IndexType:
        """获取Index类型的枚举值"""
        if self.base_type == "单":
            return IndexType.SINGLE
        elif self.base_type == "双":
            return IndexType.DOUBLE
        else:
            return IndexType.DOUBLE  # 默认值
    
    def parse_create_date(self) -> Optional[datetime]:
        """解析创建日期字符串为datetime对象
        
        支持多种日期格式:
        - "2023-12-15 08:43:38" (完整日期时间)
        - "2023-12-15" (仅日期)
        - "2023/12/15 08:43:38" (斜杠分隔)
        - "2023/12/15" (仅日期斜杠分隔)
        
        Returns:
            Optional[datetime]: 解析后的日期时间对象，失败返回None
        """
        if not self.create_date:
            return None
            
        # 支持的日期格式列表
        date_formats = [
            "%Y-%m-%d %H:%M:%S",  # 2023-12-15 08:43:38
            "%Y-%m-%d",           # 2023-12-15
            "%Y/%m/%d %H:%M:%S",  # 2023/12/15 08:43:38
            "%Y/%m/%d",           # 2023/12/15
            "%Y%m%d",             # 20231215
            "%m/%d/%Y",           # 12/15/2023
            "%m-%d-%Y",           # 12-15-2023
        ]
        
        for date_format in date_formats:
            try:
                return datetime.strptime(self.create_date.strip(), date_format)
            except (ValueError, TypeError):
                continue
        
        return None
    
    def parse_delivery_date(self) -> Optional[datetime]:
        """解析交付日期字符串为datetime对象
        
        支持多种日期格式(与create_date相同)
        
        Returns:
            Optional[datetime]: 解析后的日期时间对象，失败返回None
        """
        if not self.delivery_date:
            return None
            
        # 支持的日期格式列表
        date_formats = [
            "%Y-%m-%d %H:%M:%S",  # 2023-12-15 08:43:38
            "%Y-%m-%d",           # 2023-12-15
            "%Y/%m/%d %H:%M:%S",  # 2023/12/15 08:43:38
            "%Y/%m/%d",           # 2023/12/15
            "%Y%m%d",             # 20231215
            "%m/%d/%Y",           # 12/15/2023
            "%m-%d-%Y",           # 12-15-2023
        ]
        
        for date_format in date_formats:
            try:
                return datetime.strptime(self.delivery_date.strip(), date_format)
            except (ValueError, TypeError):
                continue
        
        return None
    
    def is_urgent_priority(self) -> bool:
        """判断是否为紧急优先级文库
        
        根据加测备注判断：如果内容为"补测"或"加测"，优先级会高
        
        Returns:
            bool: True表示高优先级
        """
        urgent_keywords = ["补测", "加测"]
        if self.add_tests_remark:
            return any(keyword in self.add_tests_remark for keyword in urgent_keywords)
        if self.add_test_note:
            return any(keyword in self.add_test_note for keyword in urgent_keywords)
        return False
    
    def calculate_priority_score(self, current_time: Optional[datetime] = None) -> float:
        """计算文库的优先级得分
        
        严格按照业务规则文档第87-90行权重分配：
        - 下单时间权重: 40%
        - 交付时间权重: 35%  
        - 数据类型优先级权重: 25%
        
        【增强说明】：完全符合业务规则文档的优先级权重算法
        
        Args:
            current_time: 当前时间，用于计算时间优先级。默认为当前时间
            
        Returns:
            float: 优先级得分，得分越高优先级越高
        """
        if current_time is None:
            current_time = datetime.now()
        
        # 1. 下单时间权重 (40%): 业务规则 - 越早下单越优先排机
        create_time = self.parse_create_date()
        order_time_score = 0.0
        if create_time:
            # 计算天数差异，越早下单得分越高
            days_since_order = (current_time - create_time).days
            # 优化：使用更精细的时间衰减算法
            if days_since_order <= 0:
                order_time_score = 0.1  # 当天下单基础分
            elif days_since_order <= 7:
                order_time_score = 0.3 + (days_since_order / 7.0) * 0.4  # 0.3-0.7
            elif days_since_order <= 30:
                order_time_score = 0.7 + ((days_since_order - 7) / 23.0) * 0.3  # 0.7-1.0
            else:
                order_time_score = 1.0  # 超过30天最高分
        
        # 2. 交付时间权重 (35%): 业务规则 - 离当前时间越近越优先排机
        delivery_time = self.parse_delivery_date()
        delivery_time_score = 0.0
        if delivery_time:
            # 计算到交付时间的天数差异
            days_to_delivery = (delivery_time - current_time).days
            if days_to_delivery < 0:
                # 已经过期，最高优先级
                delivery_time_score = 1.0
            elif days_to_delivery <= 3:
                # 3天内交付，紧急优先级
                delivery_time_score = 0.9 + (3 - days_to_delivery) * 0.1 / 3  # 0.9-1.0
            elif days_to_delivery <= 7:
                # 7天内交付，高优先级
                delivery_time_score = 0.7 + (7 - days_to_delivery) * 0.2 / 4  # 0.7-0.9
            elif days_to_delivery <= 30:
                # 30天内交付，中等优先级
                delivery_time_score = 0.3 + (30 - days_to_delivery) * 0.4 / 23  # 0.3-0.7
            else:
                # 超过30天，低优先级
                delivery_time_score = max(0.0, 0.3 - ((days_to_delivery - 30) / 60.0) * 0.3)
        
        # 3. 数据类型优先级权重 (25%): 业务规则 - 临检>YC>其他
        data_type_score = self._get_data_type_priority_score()
        
        # 4. 紧急加测优先级加成 (业务规则 - 包含扣减时间规则的样本优先处理)
        urgency_bonus = 0.15 if self.is_urgent_priority() else 0.0
        
        # 5. 样本编码特殊处理 (业务规则 - 样本编码扣减时间规则)
        sample_code_bonus = self._get_sample_code_priority_bonus()
        
        # 综合得分计算（严格按照业务规则权重）
        total_score = (
            order_time_score * 0.40 +        # 下单时间权重40%
            delivery_time_score * 0.35 +     # 交付时间权重35%
            data_type_score * 0.25 +         # 数据类型权重25%
            urgency_bonus +                  # 紧急加成
            sample_code_bonus                # 样本编码加成
        )
        
        return min(1.3, max(0.0, total_score))  # 限制在0-1.3范围内（允许加成超过1.0）
    
    def _get_data_type_priority_score(self) -> float:
        """获取数据类型的优先级得分
        
        Returns:
            float: 数据类型优先级得分 (0.0-1.0)
        """
        data_type = self.get_data_type_enum()
        priority_scores = {
            DataType.CLINICAL: 1.0,  # 临检 - 最高优先级
            DataType.YC: 0.7,        # YC - 中等优先级
            DataType.OTHER: 0.4      # 其他 - 普通优先级
        }
        return priority_scores.get(data_type, 0.4)
    
    def _get_sample_code_priority_bonus(self) -> float:
        """获取样本编码的优先级加成
        
        某些特殊样本编码有额外的优先级加成
        
        Returns:
            float: 样本编码优先级加成 (0.0-0.1)
        """
        # 特殊样本编码规则（根据实际业务需要扩展）
        if self.sample_id and 'FKDL' in self.sample_id:
            return 0.05  # 客户文库小幅加成
        return 0.0
    
    def get_pooling_key(self) -> str:
        """获取Pooling分组键
        
        Returns:
            str: Pooling分组键
        """
        return f"{self.sample_type_code}_{self.eq_type}_{self.contract_data_raw}"
    
    def needs_phix(self) -> bool:
        """判断是否需要添加PhiX
        
        Returns:
            bool: True表示需要添加PhiX
        """
        # 甲基化文库需要8.33% PhiX
        if self.sample_type_code and 'methylation' in self.sample_type_code.lower():
            return True
        return False
    
    def get_phix_ratio(self) -> float:
        """获取PhiX添加比例
        
        Returns:
            float: PhiX添加比例
        """
        if self.sample_type_code and 'methylation' in self.sample_type_code.lower():
            return 0.0833
        # 错位Barcode根据比例添加
        if self.is_misaligned_barcode():
            ratio = self.get_misaligned_ratio()
            if ratio >= 0.85:
                return 0.0
            elif ratio >= 0.70:
                return 0.05
            else:
                return 0.33
        return 0.0
    
    def is_misaligned_barcode(self) -> bool:
        """判断是否为错位Barcode文库
        
        Returns:
            bool: True表示是错位Barcode文库
        """
        return self.misplaced_barcode_data is not None and self.misplaced_barcode_data > 0
    
    def get_misaligned_ratio(self) -> float:
        """获取错位Barcode比例
        
        Returns:
            float: 错位Barcode比例
        """
        if self.misplaced_barcode_data and self.contract_data_raw > 0:
            return self.misplaced_barcode_data / self.contract_data_raw
        return 0.0
    
    def get_expected_lane_capacity(self) -> float:
        """根据机器类型、工序、优先级获取预期lane容量
        
        业务规则1：合同数据量校验规则（2025-10-09更新）
        规则更加细化，支持：
        - 工序编码（405, 1595, 876, 1770, 1832, 418, 419, 426）
        - 优先级档位（临检、YC、其他）
        - 产线文库类型（S、Z、ZS）
        - 特殊条件（甲基化、10X、错位barcode等）
        
        详见：config/contract_volume_rules.yaml 和 排机规则文档
        
        【修复说明】：2025-10-09更新，支持更细化的多维度规则
        
        Returns:
            float: Lane容量(GB)
        """
        # 尝试加载详细规则配置
        try:
            from pathlib import Path
            import yaml
            config_path = Path("config/contract_volume_rules.yaml")
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    rules = yaml.safe_load(f)
                    # 根据数据类型（优先级）和机器类型查找容量
                    capacity = self._get_capacity_from_rules(rules)
                    if capacity:
                        return capacity
        except Exception as e:
            # 配置加载失败，使用默认规则
            pass
        
        # 回退到简化规则（向后兼容）
        eq_type_lower = self.eq_type.lower()
        
        # 优先处理特殊机器类型变种（必须在get_machine_type_enum()之前）
        # 处理T7系列变种
        if "stereo-ffpe" in eq_type_lower or "stereo ffpe" in eq_type_lower:
            return 5000.0  # T7-Stereo-FFPE: 5000G
        elif "tz-stereo" in eq_type_lower or ("stereo" in eq_type_lower and ("t7" in eq_type_lower or "tz" in eq_type_lower)):
            return 4000.0  # T7-Stereo/TZ-Stereo: 4000G
        elif "t7-c4" in eq_type_lower or ("c4" in eq_type_lower and "t7" in eq_type_lower):
            return 4500.0  # T7-C4: 4500G
        elif "methylation" in eq_type_lower and "t7" in eq_type_lower:
            return 1580.0  # T7-Methylation: 1580G
        
        # 处理SURFSEQ变种
        if "surfseq-q" in eq_type_lower or "surfseq q" in eq_type_lower or ("surfseq" in eq_type_lower and "q" in eq_type_lower):
            return 750.0  # SURFSEQ-Q: 750G/Lane (8 Lane)
        
        # 处理Novaseq系列变种
        if "se50" in eq_type_lower or "se 50" in eq_type_lower:
            return 435.0  # Novaseq SE50: 435G
        elif "pe250" in eq_type_lower or "pe 250" in eq_type_lower:
            return 400.0  # Novaseq PE250: 400-420G（基础值400G）
        elif "pe50" in eq_type_lower or "pe 50" in eq_type_lower:
            return 400.0  # Novaseq PE50: 400G
        
        # 标准机器类型容量映射
        machine_type = self.get_machine_type_enum()
        capacity_map = {
            MachineType.NOVA_X_10B: 380.0,      # Nova X-10B: 380G ±10G
            MachineType.NOVA_X_25B: 975.0,      # Nova X-25B: 975G ±10G
            MachineType.NOVASEQ: 880.0,         # Novaseq PE150: 默认880G（3.6T-New模式）
            MachineType.T7: 1670.0,             # T7: 1670G ±10G
            MachineType.SURFSEQ_5000: 1200.0,   # SURFSEQ-5000: 1200G ±10G
            MachineType.NOVASEQ_X_PLUS: 975.0,  # NovaSeq X Plus: 975G ±10G
            MachineType.ZM_SURFSEQ_5000: 1200.0 # ZM SURFSeq5000: 1200G ±10G
        }
        return capacity_map.get(machine_type, 975.0)  # 默认值为Nova X-25B容量
    
    def _get_capacity_from_rules(self, rules: dict) -> Optional[float]:
        """从规则配置中获取容量
        
        Args:
            rules: 规则配置字典
            
        Returns:
            Optional[float]: 找到的容量，未找到返回None
        """
        # 首先尝试从default_capacities获取
        default_caps = rules.get('default_capacities', {})
        
        # 匹配机器类型 - 按照长度从长到短排序，优先匹配最具体的类型
        eq_type_lower = self.eq_type.lower()
        
        # 将机器名称按长度排序（长的先匹配）
        sorted_machines = sorted(default_caps.items(), key=lambda x: len(x[0]), reverse=True)
        
        for machine_name, capacity in sorted_machines:
            machine_lower = machine_name.lower()
            if machine_lower in eq_type_lower:
                return float(capacity)
        
        # 如果有工序编码（test_code），可以进一步精确匹配
        # 这里可以扩展更复杂的匹配逻辑
        
        return None
    
    def get_lane_capacity_range(self) -> tuple[float, float]:
        """获取Lane容量的允许范围 (最小值, 最大值)
        
        业务规则1：所有机器类型都有±10G的浮动范围
        
        Returns:
            tuple[float, float]: (最小容量, 最大容量)
        """
        base_capacity = self.get_expected_lane_capacity()
        return (base_capacity - 10.0, base_capacity + 10.0)
    
    def get_machine_capacity_info(self) -> dict:
        """获取机器类型的详细容量信息
        
        【新增功能】：返回机器的完整容量配置信息，支持业务规则验证
        
        Returns:
            dict: 机器容量信息字典
        """
        machine_type = self.get_machine_type_enum()
        base_capacity = self.get_expected_lane_capacity()
        min_capacity, max_capacity = self.get_lane_capacity_range()
        
        # 根据业务规则文档的完整映射表
        machine_info_map = {
            MachineType.NOVA_X_10B: {
                "工序名称": "Novaseq X Plus-PE150",
                "上机方式": "10B",
                "lane数量": 8,
                "base_capacity": 380.0,
                "tolerance": 10.0
            },
            MachineType.NOVA_X_25B: {
                "工序名称": "Novaseq X Plus-PE150", 
                "上机方式": "25B",
                "lane数量": 8,
                "base_capacity": 975.0,
                "tolerance": 10.0
            },
            MachineType.NOVASEQ: {
                "工序名称": "Novaseq PE150",
                "上机方式": "S4 XP", 
                "lane数量": 4,
                "base_capacity": 855.0,  # 780G-930G中值
                "tolerance": 10.0,
                "capacity_range": "780G-930G"
            },
            MachineType.T7: {
                "工序名称": "Novaseq-T7",
                "上机方式": "S4",
                "lane数量": 1,
                "base_capacity": 1670.0,
                "tolerance": 10.0
            },
            MachineType.SURFSEQ_5000: {
                "工序名称": "SURFSEQ-PE150",
                "上机方式": "S4", 
                "lane数量": 1,
                "base_capacity": 1200.0,
                "tolerance": 10.0
            },
            MachineType.NOVASEQ_X_PLUS: {
                "工序名称": "NovaSeq X Plus-PE150",
                "上机方式": "X Plus",
                "lane数量": 8,
                "base_capacity": 975.0,
                "tolerance": 10.0
            },
            MachineType.ZM_SURFSEQ_5000: {
                "工序名称": "ZM SURFSeq5000-PE150",
                "上机方式": "S4",
                "lane数量": 1,
                "base_capacity": 1200.0,
                "tolerance": 10.0
            }
        }
        
        info = machine_info_map.get(machine_type, machine_info_map[MachineType.NOVA_X_25B])
        
        return {
            "machine_type": machine_type.value,
            "工序名称": info["工序名称"],
            "上机方式": info["上机方式"],
            "lane数量": info["lane数量"],
            "base_capacity_gb": info["base_capacity"],
            "min_capacity_gb": info["base_capacity"] - info["tolerance"],
            "max_capacity_gb": info["base_capacity"] + info["tolerance"],
            "tolerance_gb": info["tolerance"],
            "capacity_range": info.get("capacity_range", f"{info['base_capacity']}G±{info['tolerance']}G")
        }
    
    def has_index_conflict(self, other: 'EnhancedLibraryInfo') -> bool:
        """检查与另一个文库是否存在Index冲突
        
        Args:
            other (EnhancedLibraryInfo): 另一个文库信息
            
        Returns:
            bool: True表示存在冲突
        """
        self_sequences = set(self.parse_index_sequences())
        other_sequences = set(other.parse_index_sequences())
        
        # 如果有相同的Index序列，则存在冲突
        return bool(self_sequences.intersection(other_sequences))
    
    def is_same_run_cycle(self, other: 'EnhancedLibraryInfo') -> bool:
        """检查是否具有相同的RunCycle（应排在同一lane）
        
        Args:
            other (EnhancedLibraryInfo): 另一个文库信息
            
        Returns:
            bool: True表示应排在同一lane
        """
        return (self.run_cycle and other.run_cycle and 
                self.run_cycle.strip() == other.run_cycle.strip())
    
    def needs_special_attention(self) -> bool:
        """判断是否需要特殊关注（库检结果要求）
        
        根据QPCR摩尔浓度判断：T7工序需要看，值<=1.5的文库需告知实验员
        
        Returns:
            bool: True表示需要特殊关注
        """
        if self.qpcr_concentration is not None:
            return self.qpcr_concentration <= 1.5
        if self.qpcr_molar is not None:
            return self.qpcr_molar <= 1.5
        return False
    
    def get_data_amount_gb(self) -> float:
        """获取数据量(GB)
        
        Returns:
            float: 数据量，单位GB
        """
        return self.contract_data_raw
    
    def validate_data_integrity(self) -> List[str]:
        """验证数据完整性
        
        Returns:
            List[str]: 错误信息列表，空列表表示无错误
        """
        errors = []
        
        # 必填字段验证
        required_fields = [
            ('origrec', self.origrec),
            ('sample_id', self.sample_id),
            ('data_type', self.data_type),
            ('eq_type', self.eq_type)
        ]
        
        for field_name, field_value in required_fields:
            if not field_value or str(field_value).strip() == "":
                errors.append(f"{field_name} 不能为空")
        
        # 数值字段验证
        if self.contract_data_raw <= 0:
            errors.append("合同数据量必须大于0")
        
        if self.peak_size <= 0:
            errors.append("峰值必须大于0")
        
        # Index序列验证
        if not self.index_seq or self.index_seq.strip() == "":
            errors.append("Index序列不能为空")
        
        # 日期格式验证
        if not self.parse_create_date():
            errors.append("创建日期格式错误")
        
        if not self.parse_delivery_date():
            errors.append("交付日期格式错误")
        
        return errors
    
    def to_dict(self) -> dict:
        """转换为字典格式，便于序列化"""
        return {
            'origrec': self.origrec,
            'sample_id': self.sample_id,
            'sample_type_code': self.sample_type_code,
            'data_type': self.data_type,
            'customer_library': self.customer_library,
            'base_type': self.base_type,
            'number_of_bases': self.number_of_bases,
            'index_number': self.index_number,
            'index_seq': self.index_seq,
            'add_tests_remark': self.add_tests_remark,
            'product_line': self.product_line,
            'peak_size': self.peak_size,
            'eq_type': self.eq_type,
            'contract_data_raw': self.contract_data_raw,
            'test_code': self.test_code,
            'test_no': self.test_no,
            'sub_project_name': self.sub_project_name,
            'create_date': self.create_date,
            'delivery_date': self.delivery_date,
            'lab_type': self.lab_type,
            'data_volume_type': self.data_volume_type,
            'board_number': self.board_number,
            'data_flag': self.data_flag,
            'add_test_note': self.add_test_note,
            'run_cycle': self.run_cycle,
            'is_package_lane': self.is_package_lane,
            'package_lane_number': self.package_lane_number,
            'package_fc_number': self.package_fc_number,
            'machine_note': self.machine_note,
            'qpcr_concentration': self.qpcr_concentration,
            'misplaced_barcode_data': self.misplaced_barcode_data,
            # 新增字段
            'qpcr_molar': self.qpcr_molar,
            'qubit_concentration': self.qubit_concentration,
            'species': self.species,
            'seq_scheme': self.seq_scheme,
            'ord_task_ori': self.ord_task_ori,
            'business_line': self.business_line,
            'output_rate': self.output_rate,
            'remarks': self.remarks,
            # 规则文档补充字段
            'sid': self.sid,
            'issued_batch': self.issued_batch,
            'is_add_balance': self.is_add_balance,
            'balance_data': self.balance_data,
            'peak_map': self.peak_map,
            'is_primers': self.is_primers,
            'primers_name': self.primers_name,
            'last_laneid': self.last_laneid,
            'last_cxms': self.last_cxms,
            'add_number': self.add_number,
            'xpd': self.xpd,
            'jtb': self.jtb,
            'adaptor_type': self.adaptor_type,
            'pooling': self.pooling,
            'zsclcv': self.zsclcv,
            'average_q30': self.average_q30
        }

    def to_doc_fields(self) -> dict:
        """
        按照《排机规则文档》972-1017字段命名导出，便于对齐规则文档。
        """
        return {
            'origrec': self.origrec,
            'sid': self.sid,
            'qubit': self.qubit,
            'qpcr': self.qpcr,
            'contractdata': self.contractdata,
            'species': self.species,
            'addtestsremark': self.addtestsremark,
            'deliverydate': self.deliverydate,
            'outputrate': self.outputrate,
            'sampletype': self.sampletype,
            'sampleid': self.sampleid,
            'indexseq': self.indexseq,
            'seqscheme': self.seqscheme,
            'testno': self.testno,
            'baleno': self.baleno,
            'isaddbalance': self.isaddbalance,
            'balancedata': self.balancedata,
            'subprojectname': self.subprojectname,
            'peaksize': self.peaksize,
            'peakmap': self.peakmap,
            'eqtype': self.eqtype,
            'isprimers': self.isprimers,
            'primersname': self.primersname,
            'issuedbatch': self.issuedbatch,
            'productline': self.productline,
            'boardnumber': self.boardnumber,
            'special_splits': self.special_splits,
            'bagfcno': self.bagfcno,
            'mismatchs_barcodes': self.mismatchs_barcodes,
            'lastlaneid': self.lastlaneid,
            'lastcxms': self.lastcxms,
            'addnumber': self.addnumber,
            'xpd': self.xpd,
            'jtb': self.jtb,
            'adaptortype': self.adaptortype,
            'pooling': self.pooling,
            'zsclcv': self.zsclcv,
            'average_q30': self.average_q30
        }

    # === 规则文档字段名别名，保证命名与排机规则对齐 ===
    @property
    def sampleid(self) -> str:
        return self.sample_id

    @sampleid.setter
    def sampleid(self, value: str) -> None:
        self.sample_id = value or ''

    @property
    def sampletype(self) -> str:
        return self.sample_type_code

    @sampletype.setter
    def sampletype(self, value: str) -> None:
        self.sample_type_code = value or ''

    @property
    def contractdata(self) -> float:
        return self.contract_data_raw

    @contractdata.setter
    def contractdata(self, value: float) -> None:
        self.contract_data_raw = float(value) if value is not None else 0.0

    @property
    def indexseq(self) -> str:
        return self.index_seq

    @indexseq.setter
    def indexseq(self, value: str) -> None:
        self.index_seq = value or ''

    @property
    def addtestsremark(self) -> str:
        return self.add_tests_remark

    @addtestsremark.setter
    def addtestsremark(self, value: str) -> None:
        self.add_tests_remark = value or ''

    @property
    def deliverydate(self) -> str:
        return self.delivery_date

    @deliverydate.setter
    def deliverydate(self, value: str) -> None:
        self.delivery_date = value or ''

    @property
    def outputrate(self) -> Optional[float]:
        return self.output_rate

    @outputrate.setter
    def outputrate(self, value: Optional[float]) -> None:
        self.output_rate = float(value) if value is not None else None

    @property
    def seqscheme(self) -> Optional[str]:
        return self.seq_scheme

    @seqscheme.setter
    def seqscheme(self, value: Optional[str]) -> None:
        self.seq_scheme = value

    @property
    def testno(self) -> str:
        return self.test_no

    @testno.setter
    def testno(self, value: str) -> None:
        self.test_no = value or ''

    @property
    def baleno(self) -> Optional[str]:
        return self.package_lane_number

    @baleno.setter
    def baleno(self, value: Optional[str]) -> None:
        self.package_lane_number = value

    @property
    def isaddbalance(self) -> Optional[str]:
        return self.is_add_balance

    @isaddbalance.setter
    def isaddbalance(self, value: Optional[str]) -> None:
        self.is_add_balance = value

    @property
    def balancedata(self) -> Optional[float]:
        return self.balance_data

    @balancedata.setter
    def balancedata(self, value: Optional[float]) -> None:
        self.balance_data = float(value) if value is not None else None

    @property
    def subprojectname(self) -> str:
        return self.sub_project_name

    @subprojectname.setter
    def subprojectname(self, value: str) -> None:
        self.sub_project_name = value or ''

    @property
    def peaksize(self) -> int:
        return self.peak_size

    @peaksize.setter
    def peaksize(self, value: int) -> None:
        self.peak_size = int(value) if value is not None else 0

    @property
    def peakmap(self) -> Optional[str]:
        return self.peak_map

    @peakmap.setter
    def peakmap(self, value: Optional[str]) -> None:
        self.peak_map = value

    @property
    def eqtype(self) -> str:
        return self.eq_type

    @eqtype.setter
    def eqtype(self, value: str) -> None:
        self.eq_type = value or ''

    @property
    def isprimers(self) -> Optional[str]:
        return self.is_primers

    @isprimers.setter
    def isprimers(self, value: Optional[str]) -> None:
        self.is_primers = value

    @property
    def primersname(self) -> Optional[str]:
        return self.primers_name

    @primersname.setter
    def primersname(self, value: Optional[str]) -> None:
        self.primers_name = value

    @property
    def issuedbatch(self) -> Optional[str]:
        return self.issued_batch

    @issuedbatch.setter
    def issuedbatch(self, value: Optional[str]) -> None:
        self.issued_batch = value

    @property
    def productline(self) -> str:
        return self.product_line

    @productline.setter
    def productline(self, value: str) -> None:
        self.product_line = value or ''

    @property
    def boardnumber(self) -> str:
        return self.board_number

    @boardnumber.setter
    def boardnumber(self, value: str) -> None:
        self.board_number = value or ''

    @property
    def special_splits(self) -> Optional[str]:
        return self.data_flag

    @special_splits.setter
    def special_splits(self, value: Optional[str]) -> None:
        self.data_flag = value

    @property
    def bagfcno(self) -> Optional[str]:
        return self.package_fc_number

    @bagfcno.setter
    def bagfcno(self, value: Optional[str]) -> None:
        self.package_fc_number = value

    @property
    def mismatchs_barcodes(self) -> Optional[float]:
        return self.misplaced_barcode_data

    @mismatchs_barcodes.setter
    def mismatchs_barcodes(self, value: Optional[float]) -> None:
        self.misplaced_barcode_data = float(value) if value is not None else None

    @property
    def qubit(self) -> Optional[float]:
        return self.qubit_concentration

    @qubit.setter
    def qubit(self, value: Optional[float]) -> None:
        self.qubit_concentration = float(value) if value is not None else None

    @property
    def qpcr(self) -> Optional[float]:
        return self.qpcr_concentration

    @qpcr.setter
    def qpcr(self, value: Optional[float]) -> None:
        self.qpcr_concentration = float(value) if value is not None else None

    @property
    def orderdata(self) -> Optional[float]:
        """
        Lane级拆分后下单数据量，优先取拆分值，其次原始下单值。
        """
        return self.split_order_amount or self.order_data_amount

    @orderdata.setter
    def orderdata(self, value: Optional[float]) -> None:
        self.split_order_amount = float(value) if value is not None else None

    @property
    def lastlaneid(self) -> Optional[str]:
        return self.last_laneid

    @lastlaneid.setter
    def lastlaneid(self, value: Optional[str]) -> None:
        self.last_laneid = value

    @property
    def lastcxms(self) -> Optional[str]:
        return self.last_cxms

    @lastcxms.setter
    def lastcxms(self, value: Optional[str]) -> None:
        self.last_cxms = value

    @property
    def addnumber(self) -> Optional[int]:
        return self.add_number

    @addnumber.setter
    def addnumber(self, value: Optional[int]) -> None:
        self.add_number = int(value) if value is not None else None

    @property
    def adaptortype(self) -> Optional[str]:
        return self.adaptor_type

    @adaptortype.setter
    def adaptortype(self, value: Optional[str]) -> None:
        self.adaptor_type = value
    
    @classmethod
    def _validate_input_data(cls, data: dict) -> None:
        """验证输入数据的安全性和有效性
        
        Args:
            data (dict): 输入数据字典
            
        Raises:
            TypeError: 当输入类型错误时
            ValueError: 当输入数据不合法时
        """
        # 1. 基本类型验证
        if not isinstance(data, dict):
            raise TypeError(f"输入必须是字典类型，当前类型: {type(data)}")
        
        if not data:
            raise ValueError("输入字典不能为空")
        
        # 2. 防止过大数据攻击
        MAX_DICT_SIZE = 1000  # 最大字典条目数
        MAX_STRING_LENGTH = 10000  # 最大字符串长度
        
        if len(data) > MAX_DICT_SIZE:
            raise ValueError(f"字典大小超过限制: {len(data)} > {MAX_DICT_SIZE}")
        
        # 3. 验证字典内容安全性
        for key, value in data.items():
            if not isinstance(key, str):
                raise TypeError(f"字典键必须是字符串类型: {key}")
            
            if isinstance(value, str) and len(value) > MAX_STRING_LENGTH:
                raise ValueError(f"字符串长度超过限制: {key} = {len(value)} > {MAX_STRING_LENGTH}")
            
            # 防止恶意脚本注入
            if isinstance(value, str) and any(dangerous in value.lower() for dangerous in 
                ['<script', 'javascript:', 'eval(', 'exec(', '__import__']):
                logger.warning(f"检测到潜在恶意内容，已清理: {key}")
                data[key] = re.sub(r'[<>"\';]', '', str(value))  # 清理危险字符
        
        logger.info(f"安全验证通过，开始创建文库信息实例，数据条目数: {len(data)}")
    
    @classmethod
    def _create_safe_getters(cls, data: dict):
        """创建安全的数据获取函数
        
        Args:
            data (dict): 输入数据字典
            
        Returns:
            tuple: (safe_get, safe_get_str, safe_get_int, safe_get_float) 函数元组
        """
        MAX_STRING_LENGTH = 10000
        MAX_INT_VALUE = 2**31 - 1
        MAX_FLOAT_VALUE = 1e15
        
        def safe_get(key: str, default=None):
            """安全获取字典值，支持大小写不敏感"""
            # 支持ORIGREC或origrec格式
            value = data.get(key, data.get(key.lower(), default))
            
            # 额外安全检查
            if isinstance(value, str):
                # 移除潜在的控制字符
                value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
                # 限制长度
                value = value[:MAX_STRING_LENGTH] if len(value) > MAX_STRING_LENGTH else value
            
            return value
        
        def safe_get_str(key: str, default: str = '') -> str:
            """安全获取字符串值"""
            value = safe_get(key, default)
            if value is None:
                return default
            
            result = str(value).strip()
            # 防止空值攻击
            if not result and default:
                return default
            return result
        
        def safe_get_int(key: str, default: int = 0) -> int:
            """安全获取整数值"""
            try:
                value = safe_get(key, default)
                if value is None or value == '':
                    return default
                
                result = int(float(str(value)))  # 先转float再转int，处理"1.0"格式
                
                # 防止超大数值攻击
                if abs(result) > MAX_INT_VALUE:
                    logger.warning(f"整数值过大，使用默认值: {key} = {result}")
                    return default
                
                return result
            except (ValueError, TypeError, OverflowError) as e:
                logger.warning(f"整数转换失败，使用默认值: {key} = {value}, 错误: {e}")
                return default
        
        def safe_get_float(key: str, default: float = 0.0) -> float:
            """安全获取浮点数值"""
            try:
                value = safe_get(key, default)
                if value is None or value == '':
                    return default
                
                result = float(str(value))
                
                # 防止异常值攻击
                if abs(result) > MAX_FLOAT_VALUE or not (-1e15 <= result <= 1e15):
                    logger.warning(f"浮点数值异常，使用默认值: {key} = {result}")
                    return default
                
                # 检查是否为有效数值
                if not (result == result):  # 检查NaN
                    logger.warning(f"检测到NaN值，使用默认值: {key}")
                    return default
                
                return result
            except (ValueError, TypeError, OverflowError) as e:
                logger.warning(f"浮点数转换失败，使用默认值: {key} = {value}, 错误: {e}")
                return default
        
        return safe_get, safe_get_str, safe_get_int, safe_get_float
    
    @classmethod
    def _build_instance_fields(cls, safe_get, safe_get_str, safe_get_int, safe_get_float, data: dict) -> dict:
        """构建实例字段字典
        
        Args:
            safe_get: 安全获取函数
            safe_get_str: 安全获取字符串函数
            safe_get_int: 安全获取整数函数
            safe_get_float: 安全获取浮点数函数
            data: 原始数据字典
            
        Returns:
            dict: 实例参数字典
        """
        def pick_str(keys: List[str], default: Optional[str] = None) -> Optional[str]:
            for key in keys:
                raw = safe_get(key, None)
                if raw is not None and str(raw).strip() != '':
                    return safe_get_str(key, default if default is not None else '')
            return default

        def pick_int(keys: List[str], default: Optional[int] = 0) -> Optional[int]:
            for key in keys:
                raw = safe_get(key, None)
                if raw is not None and str(raw).strip() != '':
                    return safe_get_int(key, default if default is not None else 0)
            return default

        def pick_float(keys: List[str], default: Optional[float] = 0.0) -> Optional[float]:
            for key in keys:
                raw = safe_get(key, None)
                if raw is not None and str(raw).strip() != '':
                    return safe_get_float(key, default if default is not None else 0.0)
            return default

        # 记录关键字段的获取过程（兼容规则文档命名）
        origrec = pick_str(['ORIGREC', 'origrec', 'wkorigrec'], f'AUTO_{id(data)}')
        sample_id = pick_str(['SAMPLEID', 'sampleid'], f'SAMPLE_{id(data)}')
        sid_value = pick_str(['SID', 'sid', 'SID_文库'])
        eq_type = pick_str(['EQTYPE', 'eqtype'], 'Nova X-25B')
        contract_data = pick_float(['CONTRACTDATA_RAW', 'CONTRACTDATA', 'contractdata', 'wkcontractdata'], 30.0)
        
        logger.debug(f"关键字段获取完成 - ORIGREC: {origrec}, SAMPLEID: {sample_id}, 机器类型: {eq_type}, 数据量: {contract_data}G")
        
        return {
            # 基础标识信息
            'origrec': origrec,
            'sample_id': sample_id,
            'sample_type_code': pick_str(['SAMPLETYPECODE', 'SAMPLETYPE', 'sampletype', 'wksampletype'], 'DNA'),
            'sid': sid_value,
            'wenku_unique': pick_str(['WENKU_UNIQUE', 'wenku_unique'], None),
            'lane_unique': pick_str(['LANE_UNIQUE', 'lane_unique'], None),
            
            # 数据分类信息
            'data_type': pick_str(['DATATYPE', 'datatype'], '其他'),
            'customer_library': pick_str(['CUSTOMERLIBRARY', 'customerlibrary'], '否'),
            
            # Index相关信息
            'base_type': pick_str(['BASETYPE', 'basetype'], '双'),
            'number_of_bases': pick_int(['NUMBEROFBASES', 'numberofbases'], 8),
            'index_number': pick_int(['INDEXNUMBER', 'indexnumber'], 1),
            'index_seq': pick_str(['INDEXSEQ', 'indexseq'], 'ATCGATCG'),
            
            # 测试和产线信息
            'add_tests_remark': pick_str(['ADDTESTSREMARK', 'addtestsremark'], ''),
            'product_line': pick_str(['PRODUCTLINE', 'productline'], 'S'),
            'peak_size': pick_int(['PEAKSIZE', 'peaksize'], 350),
            
            # 机器和数据量信息
            'eq_type': eq_type,
            'contract_data_raw': contract_data,
            'test_code': pick_int(['TESTCODE', 'testcode'], None),
            'test_no': pick_str(['TESTNO', 'testno', 'wktestno'], 'PE150'),
            
            # 项目和合同信息
            'sub_project_name': pick_str(['SUBPROJECTNAME', 'subprojectname'], '测试项目'),
            
            # 时间信息
            'create_date': pick_str(['CREATE_DATE', 'CREATEDATE', 'create_date'], '2025-08-18 10:00:00'),
            'delivery_date': pick_str(['DELIVERY_DATE', 'DELIVERYDATE', 'deliverydate'], '2025-08-25 18:00:00'),
            
            # 文库分类信息
            'lab_type': pick_str(['LABTYPE', 'labtype'], '内部文库'),
            'data_volume_type': pick_str(['DATAVOLUMETYPE', 'datavolumetype'], '标准'),
            'board_number': pick_str(['BOARDNUMBER', 'boardnumber'], 'V1.0'),
            
            # 可选字段/规则文档字段
            'data_flag': pick_str(['DATAFLAG', 'SPECIAL_SPLITS', 'special_splits'], None),
            'add_test_note': safe_get('ADD_TEST_NOTE'),
            'run_cycle': pick_str(['RUNCYCLE', 'runcycle'], None),
            'is_package_lane': pick_str(['IS_PACKAGE_LANE', 'is_package_lane'], None),
            'package_lane_number': pick_str(['PACKAGE_LANE_NUMBER', 'BALENO', 'baleno'], None),
            'package_fc_number': pick_str(['PACKAGE_FC_NUMBER', 'BAGFCNO', 'bagfcno'], None),
            'machine_note': pick_str(['MACHINE_NOTE', 'machine_note', '上机备注'], None),
            'qpcr_concentration': pick_float(['QPCR_CONCENTRATION', 'QPCR', 'qpcr'], None),
            'misplaced_barcode_data': pick_float(['MISPLACED_BARCODE_DATA', 'MISMATCHS_BARCODES', 'mismatchs_barcodes'], None),
            'is_add_balance': pick_str(['ISADDBALANCE', 'isaddbalance'], None),
            'balance_data': pick_float(['BALANCEDATA', 'balancedata'], None),
            'peak_map': pick_str(['PEAKMAP', 'peakmap'], None),
            'is_primers': pick_str(['ISPRIMERS', 'isprimers'], None),
            'primers_name': pick_str(['PRIMERSNAME', 'primersname'], None),
            'complex_result': pick_str(['COMPLEXRESULT', 'complexresult'], None),
            
            # 新增字段
            'qpcr_molar': pick_float(['QPCRMOLAR', 'qpcrmolar'], None),
            'qubit_concentration': pick_float(['QUBITCONC', 'QUBIT', 'qubit'], None),
            'species': pick_str(['SPECIES', 'species'], None),
            'seq_scheme': pick_str(['SEQSCHEME', 'seqscheme'], None),
            'ord_task_ori': pick_str(['ORDTASKORI', 'ordtaskori'], None),
            'issued_batch': pick_str(['ISSUEDBATCH', 'issuedbatch'], None) or pick_str(['ORDTASKORI', 'ordtaskori'], None),
            'business_line': pick_str(['BUSINESSLINE', 'businessline'], None),
            'output_rate': pick_float(['OUTPUTRATE', 'outputrate'], None),
            'remarks': pick_str(['REMARKS', 'remarks'], None),
            # 历史下单与排机信息字段
            'order_contract_data_raw': pick_float(['ORDER_CONTRACTDATA_RAW', 'order_contractdata_raw'], None),
            'order_data_amount': pick_float(['ORDER_DATA_AMOUNT', 'ORDERDATA', 'orderdata'], None),
            'split_order_amount': pick_float(['SPLIT_ORDER_AMOUNT', 'split_order_amount'], None),
            'schedule_operation_time': pick_str(['SCHEDULING_OPERATION_TIME', 'scheduling_operation_time'], ''),
            'library_code': pick_str(['LIBRARY_CODE', 'library_code'], ''),
            'deduction_time': pick_str(['DEDUCTION_TIME', 'deduction_time'], ''),
            'adjusted_delivery_time': pick_str(['ADJUSTED_DELIVERY_TIME', 'adjusted_delivery_time'], ''),
            # 规则文档补充字段
            'last_laneid': pick_str(['LASTLANEID', 'lastlaneid'], None),
            'last_cxms': pick_str(['LASTCXMS', 'lastcxms'], None),
            'add_number': pick_int(['ADDNUMBER', 'addnumber'], None),
            'xpd': pick_float(['XPD', 'xpd'], None),
            'jtb': pick_float(['JTB', 'jtb'], None),
            'adaptor_type': pick_str(['ADAPTORTYPE', 'adaptortype'], None),
            'pooling': pick_float(['POOLING', 'pooling'], None),
            'zsclcv': pick_float(['ZSCLCV', 'zsclcv'], None),
            'average_q30': pick_float(['AVERAGE_Q30', 'average_q30'], None),
        }
    
    @classmethod
    def create_from_dict(cls, data: dict) -> 'EnhancedLibraryInfo':
        """从字典数据创建EnhancedLibraryInfo实例
        
        【安全性增强】：包含完整的输入验证和恶意数据防护
        
        Args:
            data (dict): 包含文库信息的字典
            
        Returns:
            EnhancedLibraryInfo: 创建的实例
            
        Raises:
            ValueError: 当输入数据不合法时
            TypeError: 当输入类型错误时
        """
        # 1. 输入验证
        cls._validate_input_data(data)
        
        # 2. 创建安全获取函数
        safe_get, safe_get_str, safe_get_int, safe_get_float = cls._create_safe_getters(data)
        
        # 3. 构建实例字段
        fields = cls._build_instance_fields(safe_get, safe_get_str, safe_get_int, safe_get_float, data)
        
        # 4. 创建实例并验证
        try:
            logger.info("开始创建EnhancedLibraryInfo实例")
            
            instance = cls(**fields)
            
            # 5. 数据完整性验证和日志记录
            validation_errors = instance.validate_data_integrity()
            if validation_errors:
                logger.warning(f"数据完整性验证发现问题: {validation_errors}")
            else:
                logger.info("数据完整性验证通过")
            
            logger.success(f"EnhancedLibraryInfo实例创建成功 - ORIGREC: {instance.origrec}, 机器类型: {instance.eq_type}")
            return instance
            
        except Exception as e:
            logger.error(f"创建EnhancedLibraryInfo实例失败: {e}")
            logger.error(f"输入数据: {data}")
            raise ValueError(f"无法创建文库信息实例: {e}") from e
    
    @classmethod
    def from_csv_row(cls, row_data: dict) -> 'EnhancedLibraryInfo':
        """从CSV行数据创建EnhancedLibraryInfo实例
        
        Args:
            row_data (dict): CSV行数据字典
            
        Returns:
            EnhancedLibraryInfo: 创建的实例
        """
        def pick(keys, default=None):
            for key in keys:
                value = row_data.get(key)
                if value not in (None, ''):
                    return value
            return default

        def pick_int(keys, default=0):
            value = pick(keys, default=None)
            try:
                return int(value) if value not in (None, '') else default
            except (ValueError, TypeError):
                return default

        def pick_float(keys, default=0.0):
            value = pick(keys, default=None)
            try:
                return float(value) if value not in (None, '') else default
            except (ValueError, TypeError):
                return default

        # [2026-01-30 修正] 字段映射以表中实际字段名为准，避免误映射
        # 表字段参考: data/2026-01-30_v1_standardized_lane_output_v5.csv
        return cls(
            # === 基础标识字段 ===
            origrec=pick(['wkorigrec']),
            sid=pick(['wksid']),
            sample_id=pick(['wksampleid']),
            sample_type_code=pick(['wksampletype']),
            data_type=pick(['wkdatatype']),
            customer_library=pick(['CUSTOMERLIBRARY', 'customerlibrary']),  # 表中无此字段，保留兼容
            base_type=pick(['BASETYPE', 'basetype']),  # 表中无此字段，保留兼容
            number_of_bases=pick_int(['NUMBEROFBASES', 'numberofbases']),  # 表中无此字段，保留兼容
            index_number=pick_int(['INDEXNUMBER', 'indexnumber']),  # 表中无此字段，保留兼容
            
            # === Index相关 ===
            index_seq=pick(['wkindexseq']),
            
            # === 加测相关 ===
            add_tests_remark=pick(['wkaddtestsremark']),
            add_number=pick_int(['wkaddnumber'], None),
            
            # === 产品与设备 ===
            product_line=pick(['wkproductline']),
            eq_type=pick(['wkeqtype']),
            
            # === 数据量 ===
            contract_data_raw=pick_float(['wkcontractdata']),
            peak_size=pick_int(['wkpeaksize']),
            
            # === 测试与项目 ===
            test_code=pick_int(['TESTCODE', 'testcode']),  # 表中无此字段，保留兼容
            test_no=pick(['wktestno']),
            sub_project_name=pick(['wksubprojectname']),
            
            # === 时间字段 ===
            create_date=pick(['wkcreatedate']),
            delivery_date=pick(['wkdeliverydate']),
            
            # === 文库类型与数据单位 ===
            lab_type=pick(['LABTYPE', 'labtype']),  # 表中无此字段，保留兼容
            data_volume_type=pick(['wkdataunit']),
            
            # === 板与序列号 ===
            board_number=pick(['wkboardnumber']),
            wenku_unique=pick(['WENKU_UNIQUE', 'wenku_unique']),  # 表中无此字段，保留兼容
            lane_unique=pick(['LANE_UNIQUE', 'lane_unique']),  # 表中无此字段，保留兼容
            
            # === 特殊处理 ===
            data_flag=pick(['wkspecialsplits']),
            add_test_note=pick(['加测备注', 'ADD_TEST_NOTE']),  # 表中无此字段，保留兼容
            run_cycle=pick(['RunCycle', 'RUNCYCLE']),  # 表中无此字段，保留兼容
            
            # === 包Lane相关 ===
            is_package_lane=pick(['是否包lane', 'IS_PACKAGE_LANE']),  # 表中无此字段，保留兼容
            package_lane_number=pick(['wkbaleno']),
            package_fc_number=pick(['wkbagfcno']),
            machine_note=pick(['wkseqnotes']),
            
            # === 浓度与质控 ===
            qpcr_concentration=pick_float(['wkqpcr'], None),
            misplaced_barcode_data=pick_float(['wkmismatchs_barcode'], None),
            qpcr_molar=pick_float(['QPCRMOLAR', 'qpcrmolar'], None),  # 表中无此字段，保留兼容
            qubit_concentration=pick_float(['wkqubit'], None),
            
            # === 物种与测序方案 ===
            species=pick(['wkspecies']),
            seq_scheme=pick(['wkseqscheme']),
            
            # === 批次与产出 ===
            ord_task_ori=pick(['ORDTASKORI', 'ordtaskori']),  # 表中无此字段，保留兼容
            issued_batch=pick(['wkissuedbatch']),
            business_line=pick(['BUSINESSLINE', 'businessline']),  # 表中无此字段，保留兼容
            output_rate=pick_float(['wkoutputrate'], None),
            
            # === 备注与平衡 ===
            remarks=pick(['REMARKS', 'remarks']),  # 表中无此字段，保留兼容
            is_add_balance=pick(['wkisaddbalance']),
            balance_data=pick_float(['wkbalancedata'], None),
            
            # === Peak与引物 ===
            peak_map=pick(['wkpeakmap']),
            is_primers=pick(['wkisprimers']),
            primers_name=pick(['wkprimersname']),
            
            # === 历史Lane信息 ===
            last_laneid=pick(['llastlaneid']),
            last_cxms=pick(['llastcxms']),
            
            # === 适配器与杂项 ===
            xpd=pick_float(['wkxpd'], None),
            jtb=pick_float(['JTB', 'jtb'], None),  # 表中无此字段，保留兼容
            adaptor_type=pick(['wkadaptortype']),
            pooling=pick_float(['POOLING', 'pooling'], None),  # 表中无此字段，保留兼容
            zsclcv=pick_float(['ZSCLCV', 'zsclcv'], None),  # 表中无此字段，保留兼容
            average_q30=pick_float(['AVERAGE_Q30', 'average_q30'], None),  # 表中无此字段，保留兼容
            
            # === 历史下单与排机信息字段（仅在CSV存在这些列时生效）===
            order_contract_data_raw=pick_float(['运营下单合同数据量', 'ORDER_CONTRACTDATA_RAW'], None),
            order_data_amount=pick_float(['lorderdata'], None),
            split_order_amount=pick_float(['拆分后下单数据量', 'SPLIT_ORDER_AMOUNT'], None),
            schedule_operation_time=pick(['排机操作时间', 'SCHEDULING_OPERATION_TIME']),
            library_code=pick(['文库编号', 'LIBRARY_CODE']),
            deduction_time=pick(['扣减时间', 'DEDUCTION_TIME']),
            adjusted_delivery_time=pick(['扣减后交付时间', 'ADJUSTED_DELIVERY_TIME']),
            
            # === 碱基不均衡相关字段 ===
            jjbj=pick(['wk_jjbj']),
            single_index_data=pick_float(['wk_single_index_data'], None),
            ten_bp_data=pick_float(['wk_10bp_data'], None),
            
            # === 特殊拆分与样本编号 ===
            product_name=pick(['productname', 'PRODUCTNAME', '产品名称']),  # 表中无此字段，保留兼容
            special_splits=pick(['wkspecialsplits']),
            sample_number_prefix=pick(['wksample_number']),
            large_index_ori=pick(['largeindexori', 'LARGEINDEXORI', '超长Index'])  # 表中无此字段，保留兼容
        )
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"EnhancedLibraryInfo(origrec={self.origrec}, sample_id={self.sample_id}, eq_type={self.eq_type})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()


# 工具函数
def create_library_from_csv_dict(csv_dict: dict) -> EnhancedLibraryInfo:
    """便捷函数：从CSV字典创建文库信息对象"""
    return EnhancedLibraryInfo.from_csv_row(csv_dict)


def validate_libraries_batch(libraries: List[EnhancedLibraryInfo]) -> dict:
    """批量验证文库信息
    
    Args:
        libraries (List[EnhancedLibraryInfo]): 文库信息列表
        
    Returns:
        dict: 验证结果统计
    """
    validation_results = {
        'total_libraries': len(libraries),
        'valid_libraries': 0,
        'invalid_libraries': 0,
        'errors_by_library': {}
    }
    
    for i, library in enumerate(libraries):
        errors = library.validate_data_integrity()
        if errors:
            validation_results['invalid_libraries'] += 1
            validation_results['errors_by_library'][library.origrec] = errors
        else:
            validation_results['valid_libraries'] += 1
    
    return validation_results
