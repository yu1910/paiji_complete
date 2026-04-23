#!/usr/bin/env bash
# 请用 bash 执行：bash test_1.1.sh 或 chmod +x test_1.1.sh && ./test_1.1.sh（不要用 python 运行本文件）

# 53_sta.csv 验证1.1第一轮的数据， 1.1第二轮测试数据_补列.csv 验证1.1第二轮
python "./arrange_library/arrange_library_model6.py" \
  --mode arrange \
  --data-file "./53_sta.csv" \
  --output-file "./tmp/53_sta_arrange.csv"
