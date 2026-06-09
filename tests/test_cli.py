# tests/test_cli.py
from typer.testing import CliRunner

from magicat.cli import app

runner = CliRunner()


def test_run_command_prints_summary(fixture_video, tmp_path):
    result = runner.invoke(
        app, ["run", str(fixture_video), "--workdir", str(tmp_path / "job")])
    assert result.exit_code == 0
    assert "shots: 3" in result.output
    assert "preview_mp4" in result.output


def test_run_command_missing_input_fails():
    result = runner.invoke(app, ["run", "C:/nope/missing.mp4"])
    assert result.exit_code != 0
