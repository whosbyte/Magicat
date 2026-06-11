# tests/test_cli.py
from typer.testing import CliRunner

from magicat.cli import app

runner = CliRunner()


def test_run_command_prints_summary(fixture_video, tmp_path):
    workdir = tmp_path / "job"
    result = runner.invoke(
        app, ["run", str(fixture_video), "--workdir", str(workdir)])
    assert result.exit_code == 0
    assert "shots: 3" in result.output
    assert "preview_mp4" in result.output
    assert "cut_detection: ok" in result.output   # live stage progress
    assert str(workdir.resolve()) in result.output   # absolute manifest path


def test_run_command_missing_input_fails():
    result = runner.invoke(app, ["run", "C:/nope/missing.mp4"])
    assert result.exit_code != 0
