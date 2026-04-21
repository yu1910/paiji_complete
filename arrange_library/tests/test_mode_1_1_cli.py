"""
1.1 CLI 单元测试。
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from arrange_library import mode_1_1_cli


def test_run_mode_1_1_round2_export_cli_delegates_to_service(monkeypatch, capsys, tmp_path):
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"

    class _FakeIdentificationResult:
        total_candidates = 8
        candidate_groups = [object(), object(), object(), object()]

    class _FakeResult:
        output_path = output_file
        identification_result = _FakeIdentificationResult()
        exported_lane_count = 4
        exported_library_count = 8

    captured = {}

    def _fake_export(**kwargs):
        captured.update(kwargs)
        return _FakeResult()

    monkeypatch.setattr(mode_1_1_cli, "run_mode_1_1_round2_export", _fake_export)

    result_path = mode_1_1_cli.run_mode_1_1_round2_export_cli(
        [
            "--data-file",
            str(input_file),
            "--output-file",
            str(output_file),
        ]
    )

    stdout = capsys.readouterr().out
    assert result_path == output_file
    assert captured["data_file"] == input_file
    assert captured["output_file"] == output_file
    assert captured["output_detail_dir"] is None
    assert "output_path=" in stdout
    assert "candidates=8" in stdout
    assert "groups=4" in stdout
    assert "lanes=4" in stdout
    assert "libraries=8" in stdout
