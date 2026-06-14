# AI肿瘤阅片实现方案

更新时间：2026-06-11

## 一、最终口径

根据当前提供的表字段，项目当前不是“已有精细 ROI/肿瘤 mask 标注”的阅片训练数据，而是：

1. 样本接收和产品信息表。
2. 病理诊断结果表。
3. 诊断结果文本中的报告数值标签。
4. 另有 `.sdpc` 扫描片文件，可用生强数字阅片软件打开。

因此当前可落地的第一版 AI 阅片模型应定位为：

```text
样本表字段 + 诊断文本结构化 + WSI 图像特征
    -> WSI/样本级弱监督模型
    -> 癌症大类分类、报告面积回归、癌细胞占比回归、坏死识别
```

当前不能直接强交付：

1. 真实 ROI 圈内肿瘤纯度。
2. 单张圈内肿瘤面积。
3. 图像几何意义上的肿瘤面积。
4. 像素级 `tumor_mask`。
5. patch 级坏死定位。

原因是当前字段表没有 ROI 坐标、tumor mask、patch 级肿瘤标签、patch 级坏死标签。

## 二、现有字段能支持什么

### 2.1 样本接收/产品信息字段

| 字段 | 用途 |
| --- | --- |
| 病理编号+样本编号 | 两张表之间的关联键，辅助去重 |
| 病理编号 | 病例级关联键，可用于数据切分 |
| 样本编号 | 样本唯一 ID，建议作为 `sample_id` |
| 样本名称/姓名 | 仅用于关联，训练前必须脱敏 |
| 性别 | 元数据特征或分层统计字段 |
| 年龄 | 解析成数字后可作为元数据特征 |
| 样本类型 | 如石蜡贴片，用于限定样本形态 |
| 产品名称 | 可辅助判断检测场景、癌种范围、产品线 |
| 样本数量 | 可用于推断同病例/同样本张数 |
| 操作日期/送检日期 | 可用于时间切分、数据追溯 |
| 临床诊断 | 癌种标签辅助来源 |
| 病理分型 | 若有值，可作为癌种/亚型标签来源 |
| 送样组织来源 | 器官来源标签 |
| 原发部位 | 器官来源强标签 |
| 诊断结果 | 核心标签来源，需要结构化抽取 |
| 样本属性/样本剩余/核对 | 质控和业务过滤字段 |

以下字段不建议直接作为模型特征：操作人、客户名称、存放位置、返样时间、病人电话、送检医生、门诊号/住院号。它们更多是流程字段或敏感字段。

### 2.2 诊断文本可抽取标签

示例：

```text
非小细胞肺癌，具体诊断请参考原单位意见。
切除组织：119.91mm2；癌组织：21.59mm2；占18.01%。
癌细胞量约2万个，细胞占比约30%，未见坏死，间质炎性纤维组织增生，周围见肺组织。
```

可结构化为：

| 字段 | 示例值 | 含义 |
| --- | ---: | --- |
| `cancer_major` | 非小细胞肺癌 | 癌症大类/病理分型 |
| `tissue_area_report_mm2` | 119.91 | 报告记录的组织面积 |
| `tumor_area_report_mm2` | 21.59 | 报告记录的癌组织面积 |
| `tumor_area_ratio_report_pct` | 18.01 | 癌组织面积占比 |
| `tumor_cellularity_report_pct` | 30 | 癌细胞占比，接近病理纯度 |
| `tumor_cell_count_text` | 约2万个 | 癌细胞量文本 |
| `necrosis_present` | false | “未见坏死”为阴性 |

必须区分：

```text
tumor_area_ratio_report_pct != tumor_cellularity_report_pct
```

前者是面积占比，后者是细胞占比/病理纯度。训练时不能混成一个标签。

## 三、训练信息覆盖与缺口

| 训练目标/模块 | 训练需要的信息 | 当前字段是否具备 | 来源 | 当前结论 |
| --- | --- | --- | --- | --- |
| 样本唯一标识 | 样本编号、病理编号 | 有 | 字段表 | 可建 `sample_id/pathology_id` |
| 病例级切分 | 患者或病例级 ID | 部分有 | 病理编号、姓名脱敏 hash | 可用 `pathology_id` 或脱敏 `patient_key` |
| 器官来源 | 组织来源/原发部位 | 有 | 送样组织来源、原发部位 | 可作为强先验 |
| 癌症大类分类 | 癌种/病理分型 | 部分有 | 临床诊断、病理分型、诊断结果 | 可训练，需文本结构化 |
| 报告组织面积回归 | 组织面积数值 | 部分有 | 诊断结果文本 | 可训练，需抽取 |
| 报告癌组织面积回归 | 癌组织面积数值 | 部分有 | 诊断结果文本 | 可训练，需抽取 |
| 报告面积占比回归 | 癌组织面积占比 | 部分有 | 诊断结果文本/公式 | 可训练，需校验 |
| 癌细胞占比回归 | 癌细胞占比 | 部分有 | 诊断结果文本 | 可训练，需抽取 |
| 低癌细胞占比提示 | 癌细胞占比 < 10% | 部分有 | 癌细胞占比派生 | 可训练二分类辅助头 |
| 坏死识别 | 坏死有无 | 部分有 | 诊断结果文本 | 可训练样本级 MIL |
| WSI 图像模型 | WSI 文件路径 | 字段表没有 | 需要另配图像路径 | 必须补充，否则不能训练图像模型 |
| `.sdpc` 读取 | 原始扫描片和读取方式 | 字段表没有 | 生强软件、厂家 SDK/API、格式转换 | 必须验证程序能否批量读取 |
| patch 坐标 | `x/y/width/height` | 字段表没有 | WSI 切图程序自动生成 | 不需要人工提供 |
| mpp 物理尺度 | `mpp_x/mpp_y` | 字段表没有 | WSI 元数据或扫描系统 | 几何面积计算必须补充 |
| 扫描仪/倍率 | scanner、magnification | 字段表没有或不完整 | WSI 元数据 | 建议补充 |
| 染色类型 | H&E/IHC 等 | 字段表没有或不完整 | 产品名称/实验记录/人工确认 | 建议补充 |
| 组织 mask | `tissue_mask` | 字段表没有 | WSI 图像自动分割 | 可由算法生成 |
| 肿瘤 mask | `tumor_mask` | 没有 | 需要人工标注或分割标签 | 当前不能可靠训练 |
| ROI 圈选坐标 | ROI polygon/mask | 没有 | 需要医生圈选文件 | 当前不能训练真实 ROI 指标 |
| patch 级坏死标注 | patch label | 没有 | 需要局部标注 | 当前只能做样本级坏死 |
| 蔓延趋势标签 | 标准化类别 | 没有 | 需病理医生制定并标注 | 当前不训练 |

结论：

1. 字段表足够支持样本级弱监督标签构建。
2. 字段表不包含 WSI 路径、mpp、ROI 坐标、tumor mask、patch 级标签。
3. `.sdpc` 说明有扫描片，但必须验证算法程序能否读取图像、level、缩略图和 mpp。
4. 没有 WSI 路径，只能做文本结构化和表格统计，不能训练图像阅片模型。
5. 有 WSI 但没有 mpp，可以训练分类/回归/坏死模型，但不能输出可信几何面积。
6. 没有 tumor mask 或肿瘤局部标注，不能可靠计算图像肿瘤面积。
7. 没有 ROI 坐标，不能训练或计算真实圈内指标。

## 四、`.sdpc`、mpp 与面积计算

### 4.1 `.sdpc` 代表什么

`.sdpc` 是扫描片文件格式。生强数字阅片软件能打开，说明该软件能识别图像内容，但不等于算法程序已经能批量读取。

必须确认：

1. 生强是否提供 SDK/API 读取 `.sdpc`。
2. SDK/API 能否读取指定区域图像 tile。
3. 能否读取缩略图、level 0 宽高、各层 downsample。
4. 能否读取或导出 `mpp_x/mpp_y`、扫描倍率、扫描仪型号。
5. 是否支持批量导出 metadata。
6. 如果不能 SDK 读取，是否能批量转成 `.svs`、`.tif`、`.dcm`、OME-TIFF 等通用格式。

`.sdpc` 能帮助解决“读图”和“读 mpp”的问题，但不能直接告诉模型哪里是肿瘤。

### 4.2 mpp 是什么

`mpp_x/mpp_y` 是整张扫描切片图像的物理像素尺度，不是癌种属性。

```text
mpp = microns per pixel
```

例如：

```text
mpp_x = 0.5
mpp_y = 0.5
```

表示：

```text
1 个像素宽 = 0.5 微米
1 个像素高 = 0.5 微米
```

用途：

```text
面积(mm2) = 像素数 * mpp_x * mpp_y / 1,000,000
```

如果没有 mpp：

1. 可以训练癌种分类。
2. 可以训练报告数值回归。
3. 可以训练坏死识别。
4. 不能输出可信的图像几何面积。

建议在 manifest 里增加：

| 字段 | 说明 |
| --- | --- |
| `mpp_x` | x 方向每像素微米数 |
| `mpp_y` | y 方向每像素微米数 |
| `mpp_source` | `wsi_metadata/scanner_metadata/estimated/missing` |
| `mpp_valid` | 是否可信 |

### 4.3 面积为什么还需要 mask

mpp 只是尺子，mask 才是边界。

```text
面积 = mask 内像素数 * mpp_x * mpp_y / 1,000,000
```

| 面积类型 | 需要什么 mask | 当前能否做 |
| --- | --- | --- |
| 图像组织面积 | `tissue_mask` | 有 WSI + mpp 时可做 |
| 图像肿瘤面积 | `tumor_mask` | 当前不能强交付 |
| ROI 圈内面积 | ROI mask + tissue/tumor mask | 当前不能做 |
| 坏死面积 | necrosis mask | 当前不能做 |

## 五、tissue_mask 与 tumor_mask

### 5.1 tissue_mask

`tissue_mask` 用于识别哪些像素是组织、哪些是白背景/空白。

它通常可以通过图像算法自动生成：

1. 读取 WSI 缩略图。
2. HSV/LAB 颜色空间背景分离。
3. 去掉空白、气泡、笔迹、折叠、模糊区域。
4. 形态学处理。
5. 统计组织像素数。

有 WSI 和 mpp 时，可以计算：

```text
tissue_area_image_mm2 = tissue_mask 像素数 * mpp_x * mpp_y / 1,000,000
```

### 5.2 tumor_mask

`tumor_mask` 用于识别组织内部哪些像素是肿瘤。

它不能由 `tissue_mask` 直接生成。`tissue_mask` 只回答“有没有组织”，不回答“是不是肿瘤”。

| 路线 | 需要的数据 | 能否得到 tumor_mask | 可靠性 |
| --- | --- | --- | --- |
| 精细分割训练 | 医生/标注员画肿瘤 polygon/mask | 可以 | 最高 |
| patch 级分类 | 标注肿瘤 patch 和非肿瘤 patch | 生成 tumor probability map 后阈值成 mask | 中等 |
| 只有报告数值 | 癌组织面积、癌细胞占比等整张片标签 | 只能弱监督定位，不能稳定生成可靠 mask | 较低 |

当前字段表属于第三种。可以训练 WSI 级报告数值回归，但不能直接得到可靠像素级 `tumor_mask`。

建议路线：

1. 第一阶段：用现有字段训练癌种分类、报告面积回归、癌细胞占比回归、坏死识别。
2. 第二阶段：选 100 到 300 张典型切片，让医生或标注员画肿瘤区域。
3. 第三阶段：训练 patch/tile 级肿瘤识别模型，输出 `tumor_prob_map`。
4. 第四阶段：阈值和形态学处理生成候选 `tumor_mask`。
5. 第五阶段：用人工标注评估 Dice、IoU、面积误差。
6. 只有 `tumor_mask` 质量达标后，才用于自动肿瘤面积计算。

## 六、patch 坐标说明

字段表里没有 patch 坐标是正常的。patch 坐标由 WSI 切图程序自动生成。

```text
WSI level 0 图像
    -> 按 patch_size 和 stride 切网格
    -> 生成每个 patch 的 x/y/width/height
```

patch 中间表：

| 字段 | 来源 |
| --- | --- |
| `patch_id` | 系统生成 |
| `sample_id/wsi_id` | 由样本和 WSI 路径关联 |
| `x/y/width/height` | 切图程序生成 |
| `mpp_x/mpp_y` | WSI 元数据继承 |
| `tissue_ratio` | tissue_mask 与 patch 网格相交计算 |
| `embedding_path` | UNI2-h 特征提取后生成 |

patch 坐标的作用是把模型注意力、热力图、证据 patch 映射回 WSI 原图。它不能代替 mpp，也不能单独计算真实 mm2 面积。

## 七、当前可训练任务

### 7.1 癌症大类分类

输入：

1. WSI patch embedding。
2. 器官来源：`送样组织来源`、`原发部位`。
3. 癌种标签：`临床诊断`、`病理分型`、`诊断结果` 结构化结果。

输出：

```text
cancer_major
```

前提是能匹配到 WSI 文件。

### 7.2 报告面积回归

标签：

```text
tissue_area_report_mm2
tumor_area_report_mm2
tumor_area_ratio_report_pct
```

这些是诊断报告里的数值标签，不是模型根据图像几何直接算出来的面积。

### 7.3 癌细胞占比回归

标签：

```text
tumor_cellularity_report_pct
```

低值提示：

```text
low_cellularity = tumor_cellularity_report_pct < 10
```

### 7.4 坏死识别

标签：

```text
necrosis_present
```

当前只能做样本级弱监督。没有坏死 ROI 或 patch 标注时，不能训练精确坏死定位模型。

### 7.5 图像组织面积计算

前提：

1. WSI 可读取。
2. mpp 可读取且可信。
3. tissue_mask 可生成。

输出：

```text
tissue_area_image_mm2
```

## 八、当前不能训练或不能强交付的任务

| 任务 | 为什么不能做 | 需要补什么 |
| --- | --- | --- |
| 真实 ROI 圈内肿瘤纯度 | 没有 ROI 坐标，不知道圈在哪里 | 医生圈选 polygon/mask |
| 单张圈内肿瘤面积 | 没有 ROI 坐标，也没有 tumor_mask | ROI polygon + tumor_mask |
| 图像肿瘤面积几何计算 | 没有 tumor_mask | tumor polygon/mask 标注 |
| 可靠 tumor_mask 自动生成 | 只有报告级数值，没有局部肿瘤标注 | tumor polygon/mask 或 patch 级肿瘤/非肿瘤标签 |
| patch 级肿瘤分类 | 没有 patch 级肿瘤标签 | patch 或区域标注 |
| patch 级坏死定位 | 没有局部坏死标注 | patch/ROI 坏死标注 |
| 蔓延趋势 | 没有标准标签定义和训练标签 | 病理医生定义类别并标注 |
| 临床独立诊断 | 当前只是辅助分析 | 合规、临床验证、注册路径 |

## 九、数据与标注设计

### 9.1 统一训练 manifest

建议生成 `sample_manifest.csv`：

| 字段 | 来源 | 是否已有 |
| --- | --- | --- |
| `sample_id` | 样本编号 | 有 |
| `pathology_id` | 病理编号 | 有 |
| `patient_key` | 病理编号或姓名脱敏 hash | 可生成 |
| `wsi_path` | WSI 文件路径 | 需要补充 |
| `wsi_format` | 例如 `.sdpc` | 需要补充 |
| `wsi_reader` | 生强 SDK/API、格式转换、其他库 | 需要确认 |
| `sample_name_hash` | 样本名称/姓名脱敏 | 可生成 |
| `sex` | 性别 | 有 |
| `age` | 年龄 | 有 |
| `sample_type` | 样本类型 | 有 |
| `product_name` | 产品名称 | 有 |
| `clinical_diagnosis` | 临床诊断 | 有 |
| `diagnosis_text` | 诊断结果 | 有 |
| `organ_site` | 送样组织来源/原发部位 | 有 |
| `cancer_major` | 诊断文本/病理分型抽取 | 可生成 |
| `tissue_area_report_mm2` | 诊断文本抽取 | 可生成 |
| `tumor_area_report_mm2` | 诊断文本抽取 | 可生成 |
| `tumor_area_ratio_report_pct` | 诊断文本抽取/公式计算 | 可生成 |
| `tumor_cellularity_report_pct` | 诊断文本抽取 | 可生成 |
| `necrosis_present` | 诊断文本抽取 | 可生成 |
| `mpp_x/mpp_y` | WSI 元数据 | 需要补充 |
| `mpp_source` | `wsi_metadata/scanner_metadata/estimated/missing` | 需要补充 |
| `mpp_valid` | mpp 是否可信 | 需要补充 |
| `scanner_vendor` | WSI 元数据 | 建议补充 |
| `magnification` | WSI 元数据 | 建议补充 |
| `stain_type` | 实验记录/产品信息 | 建议补充 |
| `has_roi_geometry` | 是否有 ROI 坐标 | 当前默认 false |
| `split` | 训练集/验证集/测试集 | 可生成 |

### 9.2 诊断文本抽取表

建议生成 `diagnosis_extraction.parquet`：

| 字段 | 说明 |
| --- | --- |
| `sample_id` | 样本编号 |
| `diagnosis_text` | 原始诊断结果 |
| `extracted_json` | 抽取出的结构化字段 |
| `extraction_version` | 抽取规则版本 |
| `extraction_confidence` | 抽取置信度 |
| `need_manual_review` | 是否需要人工复核 |

必须做公式校验：

```text
tumor_area_report_mm2 / tissue_area_report_mm2 * 100
    是否接近 tumor_area_ratio_report_pct
```

不一致样本进入人工复核或 ignore。

### 9.3 patch_index 中间表

这是系统生成表，不是原始字段表。

| 字段 | 来源 |
| --- | --- |
| `patch_id` | 系统生成 |
| `sample_id` | manifest 关联 |
| `wsi_id` | WSI 路径或文件名生成 |
| `x/y/width/height` | WSI 切图程序生成 |
| `mpp_x/mpp_y` | WSI 元数据继承 |
| `tissue_ratio` | tissue_mask 计算 |
| `patch_label_source` | 当前多为 weak_label/unlabeled |

## 十、模型训练方案

### 10.1 训练目标

| 模型 | 输入 | 输出 | 标签来源 | 当前状态 |
| --- | --- | --- | --- | --- |
| 癌症大类 MIL 分类 | patch embedding + organ_site | cancer_major | 临床诊断/病理分型/诊断结果 | 可训练 |
| 报告组织面积回归 | patch embedding | tissue_area_report_mm2 | 诊断文本 | 可训练 |
| 报告癌组织面积回归 | patch embedding | tumor_area_report_mm2 | 诊断文本 | 可训练 |
| 报告面积占比回归 | patch embedding | tumor_area_ratio_report_pct | 诊断文本/公式 | 可训练 |
| 癌细胞占比回归 | patch embedding | tumor_cellularity_report_pct | 诊断文本 | 可训练 |
| 低癌细胞占比分类 | patch embedding | <10% 二分类 | 癌细胞占比派生 | 可训练 |
| 坏死识别 | patch embedding | necrosis_present | 诊断文本 | 可训练 |
| 图像组织面积计算 | tissue_mask + mpp | tissue_area_image_mm2 | WSI + mpp | 有 WSI/mpp 时可做 |
| tumor_mask 生成模型 | tumor polygon/mask 或 patch 级标签 | tumor_prob_map/tumor_mask | 额外局部标注 | 当前不能可靠训练 |
| ROI 圈内指标 | ROI mask + tumor mask | ROI purity/area | 额外 ROI 标注 | 当前不能训练 |

### 10.2 训练阶段

阶段 0：数据盘点与字段补齐。

1. 样本表和诊断表关联。
2. WSI 文件路径匹配。
3. 验证 `.sdpc` 是否能被算法程序批量读取。
4. 检查 WSI 是否能打开并读取 level 信息。
5. 统计 mpp 缺失率和 mpp 来源。
6. 抽取诊断文本标签。

阶段 1：文本标签结构化。

1. 抽取癌症大类。
2. 抽取组织面积、癌组织面积、面积占比。
3. 抽取癌细胞占比。
4. 抽取坏死有无。
5. 做公式一致性校验和人工抽查。

阶段 2：WSI 特征提取。

1. 读取 WSI。
2. 生成 tissue_mask。
3. 切 patch。
4. 生成 patch_index。
5. 用 UNI2-h 提取 patch embedding。

阶段 3：单任务模型。

1. 训练癌症大类分类。
2. 训练报告数值回归。
3. 训练癌细胞占比回归和低值分类。
4. 训练坏死识别。

阶段 4：联合模型与校准。

1. 多任务共享 embedding。
2. 输出分类、回归、坏死结果。
3. 对低癌细胞占比和坏死阈值做验证集校准。
4. 输出热力图和证据 patch，但不把热力图当作真实 ROI 或 tumor_mask。

### 10.3 损失函数

建议先单任务训练，再多任务联合。

```text
loss_total =
    1.0 * loss_cancer_ce
  + 0.5 * loss_tissue_area_huber
  + 0.5 * loss_tumor_area_huber
  + 0.5 * loss_area_ratio_huber
  + 0.5 * loss_cellularity_huber
  + 0.5 * loss_low_cellularity_bce
  + 0.5 * loss_necrosis_bce
```

没有某个标签的样本，不参与该任务 loss。

## 十一、推理输出建议

当前版本输出应避免声称有 ROI 几何结果或可靠 tumor_mask。

```json
{
  "sample_id": "YKHS250030773-1A",
  "pathology_id": "Y20251130028",
  "model_version": "tumor-reader-v1.0.0",
  "qc": {
    "wsi_readable": true,
    "wsi_format": ".sdpc",
    "mpp_valid": true,
    "tissue_area_image_mm2": 126.4
  },
  "diagnosis_extraction": {
    "cancer_major_text": "非小细胞肺癌",
    "tissue_area_report_mm2": 119.91,
    "tumor_area_report_mm2": 21.59,
    "tumor_area_ratio_report_pct": 18.01,
    "tumor_cellularity_report_pct": 30.0,
    "necrosis_present": false
  },
  "classification": {
    "organ_site": "lung",
    "cancer_major_top1": "non_small_cell_lung_cancer",
    "confidence": 0.87
  },
  "report_regression": {
    "tissue_area_report_pred_mm2": 118.6,
    "tumor_area_report_pred_mm2": 22.4,
    "tumor_area_ratio_report_pred_pct": 18.9,
    "tumor_cellularity_report_pred_pct": 31.2,
    "low_cellularity_flag": false
  },
  "abnormal": {
    "necrosis_present": false,
    "necrosis_confidence": 0.91
  },
  "tumor_mask": {
    "available": false,
    "reason": "no_tumor_annotation"
  },
  "roi_results": {
    "available": false,
    "reason": "no_roi_geometry"
  }
}
```

## 十二、验收标准

当前版本至少满足：

1. 样本表和诊断结果表能稳定关联。
2. 诊断文本抽取有准确率评估和人工抽查。
3. `.sdpc` 读取方式明确，能批量读取或已确定转换方案。
4. WSI 文件路径能和样本编号匹配。
5. patch_index、embedding_index 可复现生成。
6. 癌种分类在 patient_key/pathology_id 级测试集上评估。
7. 报告面积、面积占比、癌细胞占比回归有 MAE/RMSE/Spearman。
8. 癌细胞占比 < 10% 有专项召回率评估。
9. 坏死识别有 AUC/PR-AUC/Recall。
10. 无 ROI 坐标时，不输出真实圈内面积和圈内纯度。
11. 无 mpp 时，不输出可信图像几何面积。
12. 无 tumor_mask 标注或验证时，不输出自动肿瘤面积强结论。

## 十三、需要优先补充的信息

按优先级排序：

1. WSI 文件路径与样本编号的映射。
2. `.sdpc` 的程序读取方式：生强 SDK/API、批量 metadata 导出或格式转换。
3. WSI 元数据里的 `mpp_x/mpp_y`，以及 `mpp_source/mpp_valid`。
4. WSI 是否为 H&E、IHC 或其他染色类型。
5. 扫描仪厂商、倍率、扫描日期。
6. 诊断文本抽取规则和人工复核样本。
7. 若要做图像肿瘤面积，补 tumor polygon/mask 或 patch 级肿瘤/非肿瘤标签。
8. 若要做真实圈内指标，补 ROI polygon/mask。
9. 若要做坏死定位，补 patch/ROI 级坏死标注。

## 十四、参考资料

1. UNI GitHub 项目：https://github.com/mahmoodlab/UNI
2. UNI2-h 模型页：https://huggingface.co/MahmoodLab/UNI2-h
3. OpenSlide Python 文档：https://openslide.org/api/python/
4. cuCIM 文档：https://docs.rapids.ai/api/cucim/stable/
