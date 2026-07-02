from __future__ import annotations

import importlib
import sys


def test_package_cli_help_prints_usage(capsys) -> None:
    module = importlib.import_module("dynsys_econometrics.__main__")
    original_argv = sys.argv[:]
    try:
        sys.argv = ["python -m dynsys_econometrics"]
        module.main()
    finally:
        sys.argv = original_argv
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower()


def test_package_cli_smoke_runs(capsys) -> None:
    module = importlib.import_module("dynsys_econometrics.__main__")
    original_argv = sys.argv[:]
    try:
        sys.argv = ["python -m dynsys_econometrics", "smoke"]
        module.main()
    finally:
        sys.argv = original_argv
    captured = capsys.readouterr()
    assert "logistic_map:" in captured.out
