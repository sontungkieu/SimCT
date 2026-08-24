from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_kaggle_entrypoint_fails_closed_before_hardware_probe(tmp_path):
    output = tmp_path / "canary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "tpu" / "kaggle_v5e8_canary.py"),
            "--config",
            str(
                REPO_ROOT
                / "scripts"
                / "tpu"
                / "kaggle_v5e8_canary.example.json"
            ),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 69
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["phase"] == "real_model_integration"
    assert payload["status"] == "blocked"
    assert payload["configuration_validated"] is True
    assert payload["real_model_integration"] is False
    assert payload["hardware_probe_attempted"] is False
    assert payload["simct_update_executed"] is False
    assert payload["scientific_evidence"] is False
    assert "real backend unavailable" in payload["error"]
