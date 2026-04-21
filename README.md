# paiji_complete 1.1 简要说明

这份说明只回答 3 件事：

1. `1.1` 相关程序有哪些。
2. 每个程序是干什么的。
3. 怎么调用，尤其是 `1.1 第二轮` 怎么调用。

## 1.1 相关程序

### 1. `arrange_library/arrange_library_model6.py`

功能：

- 主程序。
- 跑完整排机流程。
- 包含 1.1 首轮、1.1 第二轮、prediction 后处理。

怎么调用：

```bash
python ./arrange_library/arrange_library_model6.py \
  --mode arrange \
  --data-file ./1.1测试数据.csv \
  --output-file ./tmp/1.1_arrange_output.csv
```

适合：

- 跑完整流程时用。

### 2. `arrange_library/mode_1_1_service.py`

功能：

- 1.1 的服务层封装。
- 给外部程序稳定调用，不用直接改主程序。

里面主要有 4 个入口：

- `run_mode_1_1_round1`
  - 只跑 1.1 第一轮。
- `run_mode_1_1_round2`
  - 只识别 1.1 第二轮候选。
- `run_mode_1_1_round2_export`
  - 直接导出 1.1 第二轮结果。
- `run_mode_1_1_full`
  - 走完整 1.1 流程。

怎么调用：

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

- Python 程序集成时用。

### 3. `arrange_library/mode_1_1_cli.py`

功能：

- 1.1 第二轮的命令行入口。
- 不跑整条大流程。
- 只做第二轮识别、第二轮导出。

怎么调用：

```bash
python -m arrange_library.mode_1_1_cli \
  --data-file ./1.1测试数据.csv \
  --output-file ./tmp/1.1_round2_output.csv
```


- 命令行直接跑第二轮。

### 4. `arrange_library/core/scheduling/mode_1_1_round2.py`

功能：

- 1.1 第二轮核心逻辑。
- 识别哪些文库属于第二轮。
- 按 `llastlaneid` 分组。
- 按历史 lane 直接生成第二轮 lane。

说明：

- 这是内部逻辑文件。
- 一般不要直接调它，通常通过 `mode_1_1_service.py` 或 `mode_1_1_cli.py` 调用。

### 5. `arrange_library/core/scheduling/mode_allocator.py`

功能：

- 1.1 第一轮分流。
- 把文库分到不同池子，比如 1.1 池、3.6T 池、禁排回退池。

说明：

- 这是内部逻辑文件。
- 一般不直接调用。

### 6. `arrange_library/config/mode_1_1_rules.json`

功能：

- 1.1 规则配置文件。
- 放第一轮、第二轮、pooling 等参数。

说明：

- 改规则时看这里。

### 7. `test_1.1.sh`

功能：

- 1.1 的快捷测试脚本。
- 实际调用的还是主程序 `arrange_library_model6.py`。

怎么调用：

```bash
bash ./test_1.1.sh ./1.1测试数据.csv ./tmp/1.1_arrange_output.csv
```

## 1.1 第二轮怎么调用

最推荐两种方式。

### 方式 1：命令行调用

```bash
python -m arrange_library.mode_1_1_cli \
  --data-file ./1.1测试数据.csv \
  --output-file ./tmp/1.1_round2_output.csv
```

说明：

- 这是最简单的调用方式。
- 适合直接跑第二轮。

### 方式 2：Python 调用

```python
from pathlib import Path
from arrange_library import run_mode_1_1_round2_export

result = run_mode_1_1_round2_export(
    data_file=Path("./1.1测试数据.csv"),
    output_file=Path("./tmp/1.1_round2_output.csv"),
)

print(result.output_path)
```

说明：

- 适合系统对接。
- 程序里直接拿返回结果。

## 1.1 第二轮输入要求

输入 CSV 里至少要有这些字段：

- `lastlaneround`
- `llastlaneid`
- `llastcxms`

第二轮识别逻辑是：

- `lastlaneround = 1.1第一轮`
- `llastlaneid` 不为空
- `llastcxms` 属于 `1`、`1.0`、`1.1`

满足这些条件，就会被当成 1.1 第二轮文库处理。

## 最简单结论

如果你只是想：

- 跑完整 1.1 流程：用 `arrange_library/arrange_library_model6.py`
- 只跑 1.1 第二轮：用 `arrange_library/mode_1_1_cli.py`
- 在 Python 里集成 1.1 第二轮：用 `run_mode_1_1_round2_export`
