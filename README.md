# paiji_complete 1.1 使用说明

更新时间：2026-04-23 10:56:22

这份 README 只写当前仓库里已经落地、并且按现状可以直接使用的 1.1 相关入口，不写理想设计稿口径。

先说最容易误解的一点：

- `arrange_library_model6.py` 现在会完成排机与结果导出。
- 当前版本的 `prediction_delivery` 实际是跳过状态。
- 所以下单量、预测产出量相关字段目前默认是空值，不要把当前输出理解成“已经做完真实预测”。

## 1. 当前有哪些入口

### `arrange_library/arrange_library_model6.py`

主流程入口，负责整条排机链路编排。

当前实际行为：

- 读取输入 CSV。
- 执行 3.6T-NEW 与 1.1 的排机编排。
- 识别 1.1 第二轮候选。
- 输出排机明细文件。
- 调用 `_run_prediction_delivery(...)` 做统一收口，但当前这一步不会执行真实预测，只会保留下单量/产出相关空字段。

命令行示例：

```bash
python ./arrange_library/arrange_library_model6.py \
  --mode arrange \
  --data-file ./1.1测试数据.csv \
  --output-file ./tmp/1.1_arrange_output.csv
```

适用场景：

- 想跑完整排机流程。
- 想看最终 lane 明细输出。
- 想验证 1.1 首轮是否成 lane。

### `arrange_library/mode_1_1_service.py`

1.1 的对外服务封装层，适合被 Python 程序直接调用。

当前主要入口有 4 个：

- `run_mode_1_1_round1`
  - 只跑 1.1 第一轮分流和排机。
- `run_mode_1_1_round2`
  - 只做 1.1 第二轮候选识别；`schedule_candidates=True` 时会继续做第二轮编排。
- `run_mode_1_1_round2_export`
  - 只做 1.1 第二轮识别、直出与导出。
  - 当前同样不会做真实预测，下单量/产出量字段默认还是空值。
- `run_mode_1_1_full`
  - 直接走 `arrange_library(...)`，也就是当前完整链路入口。

Python 调用示例：

```python
from pathlib import Path

from arrange_library import run_mode_1_1_round2_export

result = run_mode_1_1_round2_export(
    data_file=Path("./1.1测试数据.csv"),
    output_file=Path("./tmp/1.1_round2_output.csv"),
)

print(result.output_path)
```

适用场景：

- Python 程序集成。
- 外部系统稳定调用。
- 不想直接依赖主流程大函数。

### `arrange_library/mode_1_1_cli.py`

1.1 第二轮轻量命令行入口。

当前实际行为：

- 不跑整条主流程。
- 只识别 1.1 第二轮候选。
- 按历史 `llastlaneid` 直出第二轮 lane。
- 导出最终 CSV。
- 当前不会做真实预测。

命令行示例：

```bash
python -m arrange_library.mode_1_1_cli \
  --data-file ./1.1测试数据.csv \
  --output-file ./tmp/1.1_round2_output.csv
```

适用场景：

- 只想跑 1.1 第二轮。
- 不想带着 3.6T-NEW 主流程一起跑。

### `arrange_library/core/scheduling/mode_1_1_round2.py`

1.1 第二轮核心规则文件。

负责：

- 第二轮候选识别。
- 按 `llastlaneid` 分组。
- 复用历史 lane 关系生成第二轮 lane。

说明：

- 这是内部实现层。
- 一般不要直接调用，通常通过 `mode_1_1_service.py` 或 `mode_1_1_cli.py` 间接调用。

### `arrange_library/core/scheduling/mode_allocator.py`

1.1 第一轮分流器。

负责：

- 3.6T-NEW / 1.1 首轮分池。
- 高优样本预消耗相关分流。
- 1.1 禁排样本回退。
- 质量分组池拆分。

说明：

- 这是内部实现层。
- 一般不直接作为外部入口调用。

### `arrange_library/config/mode_1_1_rules.json`

1.1 专属规则配置文件。

主要放：

- 第一轮规则。
- 第二轮规则。
- pooling 相关参数。
- 首轮下单量除 2 等配置项。

说明：

- 需要改 1.1 阈值、标签、规则参数时，先看这里。

### `test_1.1.sh`

简单的快捷测试脚本，底层调用的还是主流程 `arrange_library_model6.py`。

脚本默认输入是 `./113_sta.csv`，不是 `./1.1测试数据.csv`。

调用示例：

```bash
bash ./test_1.1.sh ./1.1测试数据.csv ./tmp/1.1_arrange_output.csv
```

## 2. 1.1 第二轮怎么调用

当前推荐两种方式。

### 方式一：命令行

```bash
python -m arrange_library.mode_1_1_cli \
  --data-file ./1.1测试数据.csv \
  --output-file ./tmp/1.1_round2_output.csv
```

适合：

- 手工联调。
- 临时验证第二轮输出。

### 方式二：Python

```python
from pathlib import Path

from arrange_library import run_mode_1_1_round2_export

result = run_mode_1_1_round2_export(
    data_file=Path("./1.1测试数据.csv"),
    output_file=Path("./tmp/1.1_round2_output.csv"),
)

print(result.output_path)
```

适合：

- 系统对接。
- 程序里直接拿返回对象继续处理。

## 3. 1.1 第二轮输入要求

输入 CSV 至少要具备这些历史字段：

- `lastlaneround`
- `llastlaneid`
- `llastcxms` 或 `lastcxms`

当前识别口径：

- `lastlaneround = 1.1第一轮`
- `llastlaneid` 不为空
- 历史模式属于 `1`、`1.0`、`1.1`

满足这些条件，才会被识别为 1.1 第二轮候选。

如果输入数据里没有这些历史字段，或者这些字段没有值，那么第二轮不会触发。这种情况下主流程只会表现为普通排机输出，不会生成 `1.1第二轮` lane。

## 4. 现在该怎么选入口

如果你只是想：

- 跑完整排机流程：用 `arrange_library/arrange_library_model6.py`
- 只跑 1.1 第二轮：用 `arrange_library/mode_1_1_cli.py`
- 在 Python 里集成 1.1 第二轮：用 `run_mode_1_1_round2_export`

如果你关心的是“有没有真实下单量/产出预测”，当前答案是：

- 现在的重点是排机编排和结果导出。
- 真实 prediction 目前没有在这条链路里启用。
- 输出文件里的相关预测字段默认应视为空值。
