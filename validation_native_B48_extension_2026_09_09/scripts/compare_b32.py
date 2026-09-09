from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]
AUTH = ROOT / "native_M32A_blind/run"
RUN = HERE / "run_M32"
TOL = 1.0e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = []
    authoritative = []
    generalized = []
    for mode in range(1, 33):
        left_path = AUTH / f"Ra{mode}_native.bin"
        right_path = RUN / f"Ra{mode}_native.bin"
        left = np.fromfile(left_path, "<f8")
        right = np.fromfile(right_path, "<f8")
        assert left.shape == right.shape == (49536,)
        error = float(np.linalg.norm(right - left) / np.linalg.norm(left))
        rows.append({
            "mode": mode,
            "relative_error": error,
            "authoritative_sha256": sha256(left_path),
            "generalized_sha256": sha256(right_path),
            "byte_equal": left_path.read_bytes() == right_path.read_bytes(),
            "pass_1e_10": error <= TOL,
        })
        authoritative.append(left)
        generalized.append(right)
    b_auth = np.column_stack(authoritative)
    b_gen = np.column_stack(generalized)
    errors = np.array([row["relative_error"] for row in rows])
    frobenius = float(np.linalg.norm(b_gen - b_auth) / np.linalg.norm(b_auth))
    runtime = dict(line.strip().split("=", 1) for line in (RUN / "native_M32A_runtime.txt").read_text().splitlines() if "=" in line)
    result = {
        "verdict": "PASS" if frobenius <= TOL and np.all(errors <= TOL) else "FAIL",
        "preregistered_relative_tolerance": TOL,
        "frobenius_relative_error": frobenius,
        "maximum_column_relative_error": float(errors.max()),
        "median_column_relative_error": float(np.median(errors)),
        "all_columns_byte_equal": all(row["byte_equal"] for row in rows),
        "column_order": list(range(1, 33)),
        "phi_order": list(range(1, 33)),
        "rhs_filename_mapping": "mode j -> Ra{j}_native.bin",
        "output_order": list(range(1, 33)),
        "runtime_manifest_fields": runtime,
    }
    with (HERE / "B32_COLUMN_ERRORS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "B32_BACKWARD_COMPATIBILITY.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["verdict"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
