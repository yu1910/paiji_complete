#!/usr/bin/env bash
# 请用 bash 执行：bash test_1.1.sh 或 chmod +x test_1.1.sh && ./test_1.1.sh（不要用 python 运行本文件）

set -euo pipefail
# ./53_sta.csv，，没有第二轮数据
# 1.1测试数据_原始待排清洗版_补必要列.csv，没有第二轮数据
DATA_FILE="${1:-./53_sta.csv}"
OUTPUT_FILE="${2:-./tmp/53_sta_arrange.csv}"

python "./arrange_library/arrange_library_model6.py" \
  --mode arrange \
  --data-file "${DATA_FILE}" \
  --output-file "${OUTPUT_FILE}"
