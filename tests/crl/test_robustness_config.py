import json
import subprocess
import sys
from pathlib import Path

from crl.experiment import GridCalibrationConfig


def test_grid_calibration_uses_validated_defaults():
    calibration = GridCalibrationConfig()

    assert calibration.n_calib_steps == 10_000
    assert calibration.min_calib == 100


def test_cli_reports_defaults_and_accepts_grid_bin_override():
    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "cli.py"),
            "--env-name",
            "LunarLander-v3",
            "--grid-bins",
            "3",
            "--print-config-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    config = json.loads(completed.stdout)

    assert config["n_calib_steps"] == 10_000
    assert config["min_calib"] == 100
    assert config["grid_bins"] == 3
    assert config["max_workers"] == 4
