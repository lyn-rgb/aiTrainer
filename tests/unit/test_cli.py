from aitrainer.cli import main


def test_cli_dry_run(capsys):
    assert main(["dry-run"]) == 0
    assert '"status": "ok"' in capsys.readouterr().out
