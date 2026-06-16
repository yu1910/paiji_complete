# WSI 肿瘤区域识别与定量分析开发方案

**项目概述**：基于 CLAM 弱监督框架和 UNI 病理预训练模型，在没有人工 ROI/tumor mask 标注的前提下，构建 WSI 肿瘤候选区域识别、面积估算和样本级报告分析系统。

**文档版本**：1.1
**更新日期**：2026年6月15日

---

## 目录

1. [项目需求](#1-项目需求)
2. [技术方案](#2-技术方案)
3. [系统架构](#3-系统架构)
4. [开发阶段](#4-开发阶段)
5. [数据准备](#5-数据准备)
6. [实现细节](#6-实现细节)
7. [测试方案](#7-测试方案)
8. [风险评估](#8-风险评估)
9. [时间计划](#9-时间计划)
10. [资源需求](#10-资源需求)

---

## 1. 项目需求

### 1.1 当前数据口径

当前项目已有的信息主要包括：

1. 样本接收和产品信息表。
2. 病理诊断结果表。
3. 诊断结果文本中的报告数值标签。
4. `.sdpc` 扫描片文件，可通过生强数字阅片软件打开。

当前没有人工精细标注：

1. 没有医生圈选 ROI 坐标。
2. 没有像素级 `tumor_mask`。
3. 没有 patch 级肿瘤/非肿瘤标签。
4. 没有 patch 级坏死标签。

因此第一版应定位为：

```text
样本表字段 + 诊断文本结构化 + WSI 图像特征
    -> CLAM/多任务 MIL 弱监督模型
    -> 癌症大类分类、肿瘤候选热力图、模型估算肿瘤面积、报告数值回归、癌细胞占比回归、坏死识别
```

### 1.2 核心指标

| 序号 | 指标 | 第一版交付口径 | 必要条件 | 实现方式 |
| --- | --- | --- | --- | --- |
| 1 | 常见癌症大类识别 | 样本/WSI 级分类结果 | WSI + 诊断文本癌种标签 | CLAM/多任务 MIL |
| 2 | 图像组织面积 | 几何意义上的组织面积 `tissue_area_image_mm2` | WSI 可读 + mpp 可信 + tissue_mask | 自动组织分割 + mpp |
| 3 | 肿瘤候选区域 | 模型生成 `tumor_prob_map/tumor_candidate_mask` | 训练好的 CLAM/MIL 模型 | attention/patch score 阈值化 |
| 4 | 模型估算肿瘤面积 | 基于候选 mask 的估算面积，不等同人工 GT mask | WSI 可读 + mpp 可信 + 肿瘤候选 mask | patch/tile 概率图 + mpp |
| 5 | 肿瘤纯度/占比 | 区分报告面积占比与癌细胞占比；输出模型估算值和报告回归值 | 诊断文本标签 + WSI 特征 | 多任务回归 + 候选区域统计 |
| 6 | ROI 内肿瘤面积/纯度 | 若用户提供 ROI polygon，则可计算 ROI 内候选肿瘤指标 | ROI polygon/mask + 肿瘤候选 mask + mpp | 多边形相交 + 面积统计 |
| 7 | 异常表现识别 | 样本级坏死/出血等提示 | 诊断文本有弱标签 | MIL 分类头 |

### 1.3 关键边界

CLAM 可以解决“没有人工像素级标注时的弱监督肿瘤候选区域识别”，但需要明确以下边界：

1. CLAM attention/patch score 是弱监督定位结果，可以用于生成候选区域和面积估算。
2. CLAM 生成的候选 mask 不是人工标注的真实 `tumor_mask`，需要用报告面积、人工抽查或小规模标注集做校准和验证。
3. 没有 mpp 时，可以输出 patch 数、像素面积或报告数值回归，不能输出可信 mm2 几何面积。
4. 没有 ROI 坐标时，不能自动知道“圈内”范围；但如果前端/用户提供 ROI polygon，可以在候选 mask 上计算 ROI 内指标。
5. 诊断文本中的 `tumor_area_ratio_report_pct` 和 `tumor_cellularity_report_pct` 不能混用。前者是面积占比，后者更接近病理纯度/癌细胞占比。

### 1.4 输出格式

```text
results/
├── slide_001/
│   ├── metadata.json
│   ├── diagnosis_extraction.json
│   ├── cancer_classification.json
│   ├── tissue_analysis.json
│   ├── tumor_candidate_analysis.json
│   ├── tumor_area_estimation.json
│   ├── roi_analysis.json
│   ├── abnormality_detection.json
│   ├── heatmap.png
│   ├── tumor_candidate_overlay.png
│   └── report.pdf
└── batch_summary.csv
```

示例输出应包含可用性说明：

```json
{
  "tumor_candidate_mask": {
    "available": true,
    "source": "clam_attention_threshold",
    "calibrated": false,
    "note": "weakly supervised candidate mask, not manual ground truth"
  },
  "tumor_area_estimation": {
    "available": true,
    "area_mm2": 21.8,
    "mpp_valid": true
  },
  "roi_results": {
    "available": false,
    "reason": "no_roi_geometry"
  }
}
```

---

## 2. 技术方案

### 2.1 为什么选 CLAM

| 特性 | CLAM/MIL 路线 | 像素级分割路线 |
| --- | --- | --- |
| 标注需求 | 只需要 WSI/样本级标签 | 需要 ROI 或像素级 tumor mask |
| 当前数据适配度 | 高 | 低 |
| 输出 | 分类、attention 热力图、候选区域 | 精细分割 mask |
| 面积能力 | 可估算，需要校准 | 标注充分时更准确 |
| 可解释性 | 可输出高注意力证据 patch | 可输出像素边界 |

CLAM 是第一版的主路线。它不要求人工画出肿瘤区域，可通过样本级标签学习哪些 patch 与肿瘤诊断最相关，再生成热力图和候选区域。

**GitHub**: <https://github.com/mahmoodlab/CLAM>

### 2.2 为什么选 UNI v1/UNI2-h

| 模型 | 作用 | 第一版建议 |
| --- | --- | --- |
| ResNet50 | CLAM 默认基线 | 仅作为 fallback |
| UNI v1 | 成熟病理特征提取器，1024-dim | 优先落地 |
| UNI2-h | 更新的病理 foundation model | 在特征提取流程稳定后升级 |
| CONCH | 图文多模态，可结合诊断文本 | 后续探索 |

第一版建议使用 UNI v1 快速打通流程，同时将 embedding 维度和模型接口设计成可切换，便于后续替换 UNI2-h。

### 2.3 标签来源

诊断文本示例：

```text
非小细胞肺癌，具体诊断请参考原单位意见。
切除组织：119.91mm2；癌组织：21.59mm2；占18.01%。
癌细胞量约2万个，细胞占比约30%，未见坏死，间质炎性纤维组织增生，周围见肺组织。
```

可结构化为：

| 字段 | 示例 | 用途 |
| --- | ---: | --- |
| `cancer_major` | 非小细胞肺癌 | 癌症大类分类 |
| `tissue_area_report_mm2` | 119.91 | 报告组织面积回归标签 |
| `tumor_area_report_mm2` | 21.59 | 报告癌组织面积回归标签 |
| `tumor_area_ratio_report_pct` | 18.01 | 报告面积占比回归标签 |
| `tumor_cellularity_report_pct` | 30 | 癌细胞占比/低纯度提示 |
| `necrosis_present` | false | 坏死识别弱标签 |

### 2.4 模型任务设计

| 模型任务 | 输入 | 输出 | 标签来源 | 第一版状态 |
| --- | --- | --- | --- | --- |
| 癌症大类分类 | WSI patch embedding + 器官来源 | `cancer_major` | 临床诊断/病理分型/诊断文本 | 可训练 |
| 肿瘤候选定位 | CLAM attention/instance score | `tumor_prob_map` | 样本级弱标签 | 可训练和估算 |
| 报告组织面积回归 | WSI patch embedding | `tissue_area_report_mm2` | 诊断文本 | 可训练 |
| 报告癌组织面积回归 | WSI patch embedding | `tumor_area_report_mm2` | 诊断文本 | 可训练 |
| 报告面积占比回归 | WSI patch embedding | `tumor_area_ratio_report_pct` | 诊断文本/公式校验 | 可训练 |
| 癌细胞占比回归 | WSI patch embedding | `tumor_cellularity_report_pct` | 诊断文本 | 可训练 |
| 低癌细胞占比分类 | WSI patch embedding | `<10%` 二分类 | 癌细胞占比派生 | 可训练 |
| 坏死识别 | WSI patch embedding | `necrosis_present` | 诊断文本 | 可训练 |

---

## 3. 系统架构

### 3.1 处理流程

```text
样本表 / 诊断表 / WSI 文件
    ↓
[Phase 0] 数据盘点与 manifest 构建
    ├─ 样本编号、病理编号关联
    ├─ WSI 路径匹配
    ├─ .sdpc 读取或格式转换验证
    └─ mpp、倍率、扫描仪 metadata 读取
    ↓
[Phase 1] 诊断文本结构化
    ├─ 癌症大类
    ├─ 组织面积、癌组织面积、面积占比
    ├─ 癌细胞占比
    └─ 坏死有无
    ↓
[Phase 2] WSI 预处理
    ├─ 自动 tissue_mask
    ├─ patch 坐标生成
    ├─ patch_index 生成
    └─ UNI 特征提取
    ↓
[Phase 3] CLAM/MIL 训练与推理
    ├─ 癌症大类分类
    ├─ 报告数值回归
    ├─ 癌细胞占比/低值提示
    ├─ 坏死识别
    └─ attention/patch score 热力图
    ↓
[Phase 4] 后处理与面积估算
    ├─ tumor_prob_map 生成
    ├─ 阈值和形态学处理
    ├─ tumor_candidate_mask
    ├─ 组织面积和肿瘤候选面积
    └─ ROI 内候选肿瘤统计（有 ROI 时）
    ↓
[Phase 5] 可视化与报告
    ├─ 热力图
    ├─ 候选肿瘤 overlay
    ├─ JSON/CSV
    └─ PDF 报告
```

### 3.2 模块划分

| 模块 | 来源 | 功能 | 优先级 |
| --- | --- | --- | --- |
| `manifest_builder.py` | 自开发 | 样本表、诊断表、WSI 路径、mpp 汇总 | P0 |
| `diagnosis_extractor.py` | 自开发 | 诊断文本结构化和公式校验 | P0 |
| `sdpc_reader_adapter.py` | 自开发/SDK | `.sdpc` 读取、metadata 导出或格式转换 | P0 |
| `create_patches_fp.py` | CLAM 官方改造 | tissue_mask、patch 提取、坐标生成 | P0 |
| `extract_features_fp.py` | CLAM 官方改造 | UNI v1/UNI2-h 特征提取 | P0 |
| `main.py` | CLAM 官方改造 | CLAM 分类训练 | P0 |
| `multi_task_mil.py` | 自开发 | 报告数值回归、癌细胞占比、坏死多任务头 | P1 |
| `inference.py` | 自开发 | 单张/批量 WSI 推理 | P1 |
| `tumor_candidate_analyzer.py` | 自开发 | heatmap、候选 mask、面积估算、ROI 统计 | P1 |
| `report_generator.py` | 自开发 | JSON/CSV/PDF 报告 | P2 |

---

## 4. 开发阶段

### Phase 0：数据盘点与读取验证（Week 1）

目标：确认图像链路可以跑通，建立训练 manifest。

1. 关联样本接收表和病理诊断结果表。
2. 建立 `sample_id/pathology_id/patient_key`。
3. 补充 WSI 文件路径映射。
4. 验证 `.sdpc` 是否能通过 SDK/API 批量读取。
5. 若 `.sdpc` 不能直接读取，确认批量转 `.svs/.tif/.dcm/OME-TIFF` 的方案。
6. 读取 level 0 宽高、各层 downsample、mpp、倍率、扫描仪厂商。

验收标准：

1. 至少 10 张 WSI 可批量打开并读取缩略图、指定区域 tile。
2. `sample_manifest.csv` 可生成。
3. mpp 缺失率和来源统计明确。

### Phase 1：诊断文本结构化（Week 2）

目标：从诊断文本生成弱监督训练标签。

1. 抽取 `cancer_major`。
2. 抽取 `tissue_area_report_mm2`、`tumor_area_report_mm2`、`tumor_area_ratio_report_pct`。
3. 抽取 `tumor_cellularity_report_pct`。
4. 抽取 `necrosis_present`。
5. 校验 `tumor_area_report_mm2 / tissue_area_report_mm2 * 100` 是否接近报告占比。

验收标准：

1. 结构化结果保存为 `diagnosis_extraction.parquet`。
2. 抽取规则有版本号。
3. 随机抽查样本有准确率记录。
4. 公式不一致样本进入人工复核或训练忽略列表。

### Phase 2：WSI 预处理与特征提取（Week 3-4）

目标：生成 patch_index 和 UNI embedding。

```bash
python create_patches_fp.py \
  --source data/wsi_files \
  --save_dir data/patches_output \
  --patch_size 256 \
  --seg --patch --stitch
```

```bash
export UNI_CKPT_PATH=models/uni/pytorch_model.bin

python extract_features_fp.py \
  --data_h5_dir data/patches_output/patches \
  --data_slide_dir data/wsi_files \
  --feat_dir data/features \
  --model_name uni_v1 \
  --batch_size 256
```

验收标准：

1. `patch_index.parquet` 包含 `sample_id/wsi_id/x/y/width/height/tissue_ratio/mpp_x/mpp_y`。
2. 10 张 WSI 能成功完成 tissue_mask、patch 提取和 embedding。
3. patch 坐标能映射回 WSI 原图。

### Phase 3：CLAM/MIL 模型训练（Week 5-6）

目标：训练样本级弱监督模型，并获得 attention/patch score。

```bash
python create_splits_seq.py \
  --task task_cancer_major \
  --seed 1 \
  --k 10
```

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --drop_out 0.25 \
  --early_stopping \
  --lr 2e-4 \
  --k 10 \
  --exp_code cancer_major_UNI \
  --weighted_sample \
  --bag_loss ce \
  --inst_loss svm \
  --task task_cancer_major \
  --model_type clam_sb \
  --log_data \
  --data_root_dir data/features \
  --embed_dim 1024
```

扩展训练：

1. 报告面积/占比回归头。
2. 癌细胞占比回归和 `<10%` 低值分类头。
3. 坏死识别二分类头。

验收标准：

1. 训练/验证/测试按 `pathology_id` 或 `patient_key` 切分，避免同病例泄漏。
2. 分类输出 AUC、macro F1、敏感度、特异度。
3. 回归输出 MAE、RMSE、Spearman。
4. 可导出每个 patch 的 attention/score。

### Phase 4：肿瘤候选区域与面积估算（Week 7-8）

目标：把 CLAM 的 attention/patch score 转成可解释的候选区域和面积。

1. 将 patch score 映射回 WSI 坐标。
2. 生成 `tumor_prob_map`。
3. 基于验证集进行阈值校准。
4. 形态学处理生成 `tumor_candidate_mask`。
5. 结合 `tissue_mask` 和 mpp 计算：
   - `tissue_area_image_mm2`
   - `tumor_area_estimated_mm2`
   - `tumor_area_estimated_ratio_pct`
6. 与诊断文本中的报告面积和面积占比做误差分析。

验收标准：

1. 每张 WSI 有 heatmap 和 overlay。
2. 面积估算输出明确标记为 `estimated_by_weak_supervision`。
3. 和报告面积的 MAE/RMSE/相关性有评估。
4. 随机抽查高/中/低分样本，人工确认热力图是否大体落在疑似肿瘤区域。

### Phase 5：报告生成与系统集成（Week 9）

目标：输出 JSON、CSV、PDF 和可视化结果。

验收标准：

1. 单张 WSI 可完整输出分析目录。
2. 批量汇总 CSV 包含可用性、置信度和异常提示。
3. 无 ROI、无 mpp 或 WSI 读取失败时，报告能给出明确原因，而不是输出错误面积。

### Phase 6：小规模人工校准与升级（Week 10-12）

目标：提高肿瘤候选 mask 和面积估算可信度。

1. 选 100-300 张典型切片做人工 ROI/tumor polygon 或 patch 级标注。
2. 评估候选 mask 的 Dice、IoU、面积误差。
3. 调整阈值、训练 patch 分类器或弱监督分割模型。
4. 升级 UNI2-h 或引入 CONCH。

---

## 5. 数据准备

### 5.1 统一训练 manifest

建议生成 `sample_manifest.csv`：

| 字段 | 来源 | 是否已有 |
| --- | --- | --- |
| `sample_id` | 样本编号 | 有 |
| `pathology_id` | 病理编号 | 有 |
| `patient_key` | 病理编号或姓名脱敏 hash | 可生成 |
| `wsi_path` | WSI 文件路径 | 需要补充 |
| `wsi_format` | `.sdpc/.svs/.tif` 等 | 需要补充 |
| `wsi_reader` | 生强 SDK/API、格式转换、OpenSlide/cuCIM | 需要确认 |
| `sex` | 性别 | 有 |
| `age` | 年龄 | 有 |
| `sample_type` | 样本类型 | 有 |
| `product_name` | 产品名称 | 有 |
| `clinical_diagnosis` | 临床诊断 | 有 |
| `diagnosis_text` | 诊断结果 | 有 |
| `organ_site` | 送样组织来源/原发部位 | 有 |
| `cancer_major` | 文本抽取 | 可生成 |
| `tissue_area_report_mm2` | 文本抽取 | 可生成 |
| `tumor_area_report_mm2` | 文本抽取 | 可生成 |
| `tumor_area_ratio_report_pct` | 文本抽取/公式计算 | 可生成 |
| `tumor_cellularity_report_pct` | 文本抽取 | 可生成 |
| `necrosis_present` | 文本抽取 | 可生成 |
| `mpp_x/mpp_y` | WSI 元数据 | 需要补充 |
| `mpp_source` | `wsi_metadata/scanner_metadata/estimated/missing` | 需要补充 |
| `mpp_valid` | mpp 是否可信 | 需要补充 |
| `scanner_vendor` | WSI 元数据 | 建议补充 |
| `magnification` | WSI 元数据 | 建议补充 |
| `stain_type` | 实验记录/产品信息 | 建议补充 |
| `has_roi_geometry` | 是否有 ROI 坐标 | 默认 false |
| `split` | train/val/test | 可生成 |

### 5.2 `.sdpc` 与 mpp

`.sdpc` 能帮助解决读图和读 metadata，但不能直接告诉模型哪里是肿瘤。必须确认：

1. 生强是否提供 SDK/API。
2. 是否能读取指定区域 tile。
3. 是否能读取缩略图、level 0 宽高、各层 downsample。
4. 是否能读取或导出 `mpp_x/mpp_y`。
5. 是否能批量导出 metadata。
6. 如果不能直接读取，是否能批量转换为通用格式。

mpp 是面积计算的尺子：

```text
面积(mm2) = 像素数 * mpp_x * mpp_y / 1,000,000
```

没有 mpp 时，不输出可信 mm2 面积；可以输出 patch 数、像素数或报告数值回归。

### 5.3 patch_index 中间表

patch 坐标由切图程序自动生成，不需要原始字段表提供。

| 字段 | 来源 |
| --- | --- |
| `patch_id` | 系统生成 |
| `sample_id` | manifest 关联 |
| `wsi_id` | WSI 路径或文件名生成 |
| `x/y/width/height` | WSI 切图程序生成 |
| `level` | 切图层级 |
| `downsample` | WSI 元数据 |
| `mpp_x/mpp_y` | WSI 元数据继承 |
| `tissue_ratio` | tissue_mask 与 patch 网格相交 |
| `embedding_path` | UNI 特征文件 |
| `patch_score` | CLAM/MIL 推理生成 |

---

## 6. 实现细节

### 6.1 tissue_mask 与 tumor_candidate_mask

`tissue_mask` 回答“哪里有组织”，通常可自动生成：

1. 读取 WSI 缩略图。
2. HSV/LAB 背景分离。
3. 去除空白、气泡、笔迹、折叠、模糊区域。
4. 形态学处理。
5. 映射回目标 level 或 level 0。

`tumor_candidate_mask` 回答“模型认为哪里疑似肿瘤”，由 CLAM/MIL 推理生成：

1. patch embedding 输入 CLAM。
2. 获取 patch attention 或 instance score。
3. 映射成 `tumor_prob_map`。
4. 用验证集阈值转为候选 mask。
5. 用形态学处理平滑边界并去除小噪声。

### 6.2 面积计算

不要用倍率直接折算面积。面积应以 WSI metadata 中的 mpp 和坐标层级为准。

```python
def patch_area_mm2(width_px, height_px, mpp_x, mpp_y, downsample=1.0, tissue_ratio=1.0):
    level0_width = width_px * downsample
    level0_height = height_px * downsample
    area_um2 = level0_width * level0_height * mpp_x * mpp_y * tissue_ratio
    return area_um2 / 1_000_000
```

候选肿瘤面积：

```text
tumor_area_estimated_mm2 =
    sum(patch_area_mm2_i * tumor_probability_i)
```

或阈值化后：

```text
tumor_area_estimated_mm2 =
    sum(patch_area_mm2_i for patch_i in tumor_candidate_mask)
```

### 6.3 肿瘤纯度和占比

需要同时输出三类值，避免混淆：

| 字段 | 含义 |
| --- | --- |
| `tumor_area_ratio_report_pct` | 诊断报告里的癌组织面积占比 |
| `tumor_cellularity_report_pct` | 诊断报告里的癌细胞占比/病理纯度 |
| `tumor_area_estimated_ratio_pct` | 模型候选肿瘤面积 / 图像组织面积 |

低癌细胞占比提示：

```text
low_cellularity_flag = tumor_cellularity_report_pred_pct < 10
```

### 6.4 ROI 分析

ROI 不是模型凭空生成的输入。第一版支持两种情况：

1. 没有 ROI polygon：报告中标记 `roi_results.available=false`。
2. 用户在前端圈选 ROI 或导入 ROI polygon：计算 ROI 内组织面积、候选肿瘤面积和候选肿瘤占比。

```python
def analyze_roi(roi_polygon, patch_coords, patch_scores, patch_areas_mm2, threshold):
    inside_roi = point_in_polygon(patch_coords, roi_polygon)
    tumor_like = patch_scores >= threshold
    roi_total_area = patch_areas_mm2[inside_roi].sum()
    roi_tumor_area = patch_areas_mm2[inside_roi & tumor_like].sum()
    roi_tumor_ratio = roi_tumor_area / roi_total_area * 100 if roi_total_area > 0 else 0
    return roi_tumor_area, roi_tumor_ratio
```

### 6.5 多任务损失

建议先单任务训练，再多任务联合：

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

---

## 7. 测试方案

### 7.1 单元测试

1. 诊断文本抽取：面积、占比、癌细胞占比、坏死阴阳性。
2. 面积公式：mpp、downsample、tissue_ratio。
3. patch 坐标映射：patch 到 WSI 原图/缩略图。
4. ROI polygon 相交判断。
5. 无 mpp、无 ROI、WSI 读取失败时的输出降级。

### 7.2 集成测试

```bash
pytest tests/ -v
```

集成样本至少覆盖：

1. `.sdpc` 正常读取。
2. mpp 缺失。
3. 诊断文本缺少面积。
4. 诊断文本有“未见坏死”。
5. 用户提供 ROI polygon。

### 7.3 模型评估

| 任务 | 指标 |
| --- | --- |
| 癌症大类分类 | AUC、macro F1、敏感度、特异度 |
| 报告面积回归 | MAE、RMSE、Spearman |
| 癌细胞占比回归 | MAE、RMSE、低值召回率 |
| 坏死识别 | AUC、PR-AUC、Recall |
| 肿瘤候选面积估算 | 与报告面积的 MAE/RMSE/相关性，人工抽查一致性 |

---

## 8. 风险评估

| 风险 | 影响 | 缓解措施 |
| --- | --- | --- |
| CLAM attention 只关注判别性区域，未覆盖完整肿瘤 | 面积低估 | 用报告面积校准阈值，引入 top-k/概率积分，小规模人工标注验证 |
| 诊断文本标签噪声 | 分类/回归偏差 | 规则版本化、公式校验、人工抽查、低置信样本忽略 |
| `.sdpc` 无法程序读取 | 流程无法自动化 | SDK/API、批量导出 metadata、格式转换三路并行验证 |
| mpp 缺失或不可信 | mm2 面积不可用 | 输出降级为 patch/像素统计，补 metadata |
| 同病例数据泄漏 | 测试指标虚高 | 按 `pathology_id/patient_key` 切分 |
| 跨扫描仪/染色差异 | 泛化下降 | 扫描仪分层评估，颜色标准化，增加多中心数据 |
| ROI 指标被误解为自动圈选 | 交付口径不清 | 报告中明确 ROI 需要外部 polygon；无 ROI 时不输出圈内指标 |

---

## 9. 时间计划

| 周数 | 阶段 | 交付物 | 状态 |
| --- | --- | --- | --- |
| 1 | Phase 0：数据盘点与读取验证 | manifest、`.sdpc` 读取结论、mpp 统计 | 计划 |
| 2 | Phase 1：诊断文本结构化 | `diagnosis_extraction.parquet` | 计划 |
| 3-4 | Phase 2：WSI 预处理与特征提取 | patch_index、UNI embedding | 计划 |
| 5-6 | Phase 3：CLAM/MIL 训练 | 分类/回归/坏死模型 | 计划 |
| 7-8 | Phase 4：候选区域与面积估算 | heatmap、candidate mask、面积估算 | 计划 |
| 9 | Phase 5：报告生成与集成 | JSON/CSV/PDF 报告 | 计划 |
| 10-12 | Phase 6：小规模人工校准与升级 | 评估集、阈值校准、UNI2-h 升级 | 计划 |

---

## 10. 资源需求

### 10.1 硬件

| 资源 | 建议 |
| --- | --- |
| GPU | RTX 4090/RTX A6000/A100，至少 24GB 显存 |
| CPU | 16-32 核 |
| 内存 | 128-256GB |
| 存储 | 10TB SSD/NAS，按 WSI 数量扩展 |

### 10.2 软件与依赖

1. Python 3.9+。
2. PyTorch。
3. OpenSlide/cuCIM，或生强 `.sdpc` SDK/API。
4. CLAM。
5. UNI/UNI2-h 权重。
6. pandas、pyarrow、opencv、scikit-image、shapely、matplotlib。

---

## 附录 A：命令速查表

```bash
# 环境
conda activate wsi_analysis

# 数据 manifest
python manifest_builder.py --sample_table data/sample.xlsx --diagnosis_table data/diagnosis.xlsx --wsi_root data/wsi --out data/sample_manifest.csv

# 诊断文本抽取
python diagnosis_extractor.py --manifest data/sample_manifest.csv --out data/diagnosis_extraction.parquet

# Patch 提取
python create_patches_fp.py --source data/wsi --save_dir data/patches --patch_size 256 --seg --patch --stitch

# 特征提取
export UNI_CKPT_PATH=models/uni/pytorch_model.bin
python extract_features_fp.py --data_h5_dir data/patches/patches --feat_dir data/features --model_name uni_v1

# CLAM 训练
python main.py --exp_code cancer_major_UNI --task task_cancer_major --model_type clam_sb --embed_dim 1024

# 推理 + 候选肿瘤面积估算
python batch_analyze.py --manifest data/sample_manifest.csv --input_dir data/wsi --output_dir results --model_path results/best_model.pt

# 测试
pytest tests/ -v
```

---

**最后更新**：2026-06-15
