"""
1.1 模式命令行入口。

提供第二轮轻量导出的独立 CLI，避免联调方手写 Python 调用。

推荐在 ``paiji_complete`` 目录下使用模块方式执行：
python -m arrange_library.mode_1_1_cli --data-file ./1.1测试数据.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from arrange_library.mode_1_1_service import run_mode_1_1_round2_export


def build_round2_export_parser() -> argparse.ArgumentParser:
    """构建 1.1 第二轮轻量导出 CLI 参数。"""
    parser = argparse.ArgumentParser(
        description="执行 1.1 第二轮轻量导出，只处理历史第二轮候选并输出最终 CSV。",
    )
    parser.add_argument(
        "--data-file",
        required=True,
        help="输入 CSV 文件路径，必须包含 lastlaneround/llastlaneid/llastcxms 等历史字段。",
    )
    parser.add_argument(
        "--output-file",
        help="输出 CSV 文件完整路径；如不指定，则自动输出到输入文件同目录。",
    )
    parser.add_argument(
        "--output-detail-dir",
        help="输出目录；仅在未传 --output-file 时生效。",
    )
    return parser


def run_mode_1_1_round2_export_cli(argv: Optional[Sequence[str]] = None) -> Path:
    """执行 1.1 第二轮轻量导出 CLI。"""
    parser = build_round2_export_parser()
    args = parser.parse_args(argv)

    result = run_mode_1_1_round2_export(
        data_file=Path(args.data_file),
        output_detail_dir=Path(args.output_detail_dir) if args.output_detail_dir else None,
        output_file=Path(args.output_file) if args.output_file else None,
    )

    print(f"output_path={result.output_path}")
    print(f"candidates={result.identification_result.total_candidates}")
    print(f"groups={len(result.identification_result.candidate_groups)}")
    print(f"lanes={result.exported_lane_count}")
    print(f"libraries={result.exported_library_count}")
    return result.output_path


def main() -> None:
    """CLI 主入口。"""
    run_mode_1_1_round2_export_cli()


__all__ = [
    "build_round2_export_parser",
    "main",
    "run_mode_1_1_round2_export_cli",
]


if __name__ == "__main__":
    main()
