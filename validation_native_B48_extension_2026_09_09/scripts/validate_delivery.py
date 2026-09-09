from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    json_files = list(HERE.glob("*.json"))
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))
    csv_files = [
        HERE / "B32_COLUMN_ERRORS.csv", HERE / "B48_PHI_MANIFEST.csv",
        HERE / "B48_NEW_MODE_STEP_REFINEMENT.csv", HERE / "B48_NEW_MODE_FD_QUALIFICATION.csv",
        HERE / "H48_ADJOINT_IDENTITY.csv",
    ]
    for path in csv_files:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert rows, path
    freeze = json.loads((HERE / "R6P_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    assert freeze["verdict"] == "PASS"
    for row in freeze["artifacts"]:
        path = HERE / row["path"]
        assert path.is_file() and path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"], path
    decision = json.loads((HERE / "R6P_FINAL_DECISION.json").read_text(encoding="utf-8"))
    assert decision["verdict"] == "PASS" and all(decision["gates"].values())
    b48 = np.fromfile(HERE / "B48_native_49536x48.bin", "<f8").reshape(49536, 48)
    h48 = np.fromfile(HERE / "H48_native_49824x48.bin", "<f8").reshape(49824, 48)
    b32 = np.column_stack([np.fromfile(ROOT / f"native_M32A_blind/run/Ra{mode}_native.bin", "<f8") for mode in range(1, 33)])
    h32 = np.fromfile(ROOT / "native_M32A_blind/H_native_49824x32.bin", "<f8").reshape(49824, 32)
    assert np.array_equal(b48[:, :32], b32)
    assert np.array_equal(h48[:, :32], h32)
    handoff = (HERE / "NATIVE_B48_EXTENSION_HANDOFF.md").read_text(encoding="utf-8")
    for section in range(1, 18):
        assert f"## {section}." in handoff
    print(json.dumps({
        "verdict": "PASS",
        "json_files_parsed": len(json_files),
        "csv_files_checked": len(csv_files),
        "freeze_artifacts_verified": len(freeze["artifacts"]),
        "B48_shape": list(b48.shape),
        "H48_shape": list(h48.shape),
        "B48_prefix_byte_equal": True,
        "H48_prefix_byte_equal": True,
    }, indent=2))


if __name__ == "__main__":
    main()
