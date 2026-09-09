from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]
RUN48 = HERE / "run_M48"
AUTH = ROOT / "native_M32A_blind" / "run"
TOL = 1.0e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    columns = []
    column_records = []
    for mode in range(1, 49):
        path = RUN48 / f"Ra{mode}_native.bin"
        column = np.fromfile(path, "<f8")
        assert column.shape == (49536,)
        columns.append(column)
        column_records.append({
            "mode": mode,
            "rhs_file": path.name,
            "sha256": sha256(path),
            "n_values": int(column.size),
        })
    b48 = np.column_stack(columns)
    binary = HERE / "B48_native_49536x48.bin"
    npy = HERE / "B48_native_49536x48.npy"
    b48.astype("<f8", copy=False).tofile(binary)
    np.save(npy, b48, allow_pickle=False)

    prefix_rows = []
    authoritative = []
    for mode in range(1, 33):
        path = AUTH / f"Ra{mode}_native.bin"
        reference = np.fromfile(path, "<f8")
        authoritative.append(reference)
        error = float(np.linalg.norm(b48[:, mode - 1] - reference) / np.linalg.norm(reference))
        prefix_rows.append({
            "mode": mode,
            "relative_error": error,
            "byte_equal": (RUN48 / f"Ra{mode}_native.bin").read_bytes() == path.read_bytes(),
            "authoritative_sha256": sha256(path),
            "b48_column_sha256": column_records[mode - 1]["sha256"],
        })
    b32 = np.column_stack(authoritative)
    errors = np.asarray([row["relative_error"] for row in prefix_rows])
    frobenius = float(np.linalg.norm(b48[:, :32] - b32) / np.linalg.norm(b32))
    prefix = {
        "verdict": "PASS" if frobenius <= TOL and np.all(errors <= TOL) else "FAIL",
        "preregistered_relative_tolerance": TOL,
        "frobenius_relative_prefix_error": frobenius,
        "maximum_prefix_column_relative_error": float(errors.max()),
        "median_prefix_column_relative_error": float(np.median(errors)),
        "all_prefix_columns_byte_equal": all(row["byte_equal"] for row in prefix_rows),
        "column_order": list(range(1, 49)),
        "prefix_column_details": prefix_rows,
    }

    with (RUN48 / "native_M32A_state_hashes.csv").open(newline="", encoding="utf-8") as stream:
        state_rows = list(csv.DictReader(stream))
    state_hashes = {row["hash"] for row in state_rows}
    with (RUN48 / "native_M32A_private_input_hashes.csv").open(newline="", encoding="utf-8") as stream:
        private_rows = list(csv.DictReader(stream))
    private_hash_tuples = {
        (row["dprestrw"], row["dstiw"], row["dielmatw"], row["dstiiniw"])
        for row in private_rows
    }
    provenance = {
        "construction_count": 1,
        "shape": list(b48.shape),
        "dtype": "little-endian float64",
        "storage_order": "row-major matrix with mode columns",
        "binary_path": binary.name,
        "binary_sha256": sha256(binary),
        "npy_path": npy.name,
        "npy_sha256": sha256(npy),
        "generalized_source_path": "generalized_source/nonlingeo.c",
        "generalized_source_sha256": sha256(HERE / "generalized_source" / "nonlingeo.c"),
        "executable_path": "generalized_source/native_B48Q_i4.exe",
        "executable_sha256": sha256(HERE / "generalized_source" / "native_B48Q_i4.exe"),
        "phi_storage": "phi48/modal_phi{mode}.bin for modes 1..48",
        "phi_column_sha256": {
            str(mode): sha256(HERE / "phi48" / f"modal_phi{mode}.bin")
            for mode in range(1, 49)
        },
        "mode_rhs_manifest": column_records,
        "state_hash_invariant": len(state_hashes) == 1,
        "state_hash": next(iter(state_hashes)) if len(state_hashes) == 1 else None,
        "private_input_hashes_invariant": len(private_hash_tuples) == 1,
        "private_input_hashes": list(next(iter(private_hash_tuples))) if len(private_hash_tuples) == 1 else None,
        "runtime_manifest": dict(
            line.strip().split("=", 1)
            for line in (RUN48 / "native_M32A_runtime.txt").read_text().splitlines()
            if "=" in line
        ),
    }
    (HERE / "B48_PREFIX_AUDIT.json").write_text(json.dumps(prefix, indent=2) + "\n", encoding="utf-8")
    (HERE / "B48_PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prefix": prefix, "provenance": provenance}, indent=2))
    if prefix["verdict"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
