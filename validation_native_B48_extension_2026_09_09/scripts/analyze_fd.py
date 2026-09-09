from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
FDROOT = HERE / "fd_new_modes"
RUN48 = HERE / "run_M48"
H0 = 0.08957732492949613
STEPS = [H0, H0 / 2.0, H0 / 4.0]
REPRESENTATIVES = [33, 40, 48]
TOL = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> None:
    native = {
        mode: np.fromfile(RUN48 / f"Ra{mode}_native.bin", "<f8")
        for mode in range(33, 49)
    }
    fd = {
        (mode, index): np.fromfile(FDROOT / f"mode_{mode:02d}" / f"R_a_centered_{index}.bin", "<f8")
        for mode in range(33, 49)
        for index in range(3)
    }
    step_rows = []
    for mode in REPRESENTATIVES:
        previous = None
        previous_error = None
        for index, step in enumerate(STEPS):
            error = rel(native[mode], fd[mode, index])
            change = None if previous is None else rel(fd[mode, index], previous)
            error_ratio = None if previous_error is None or previous_error == 0 else error / previous_error
            step_rows.append({
                "mode": mode,
                "step_label": ["h0", "h0/2", "h0/4"][index],
                "step": step,
                "native_norm": float(np.linalg.norm(native[mode])),
                "fd_norm": float(np.linalg.norm(fd[mode, index])),
                "native_vs_fd_relative_error": error,
                "fd_change_from_coarser": change,
                "native_error_ratio_to_coarser": error_ratio,
                "native_sha256": sha256(RUN48 / f"Ra{mode}_native.bin"),
                "fd_sha256": sha256(FDROOT / f"mode_{mode:02d}" / f"R_a_centered_{index}.bin"),
            })
            previous = fd[mode, index]
            previous_error = error

    candidates = []
    for index in (1, 2):
        selected = [row for row in step_rows if row["step_label"] == ["h0", "h0/2", "h0/4"][index]]
        stable = all(
            row["native_vs_fd_relative_error"] <= TOL
            and row["fd_change_from_coarser"] <= TOL
            and row["native_error_ratio_to_coarser"] <= 2.0
            for row in selected
        )
        candidates.append((index, stable))
    stable_indices = [index for index, stable in candidates if stable]
    confirmation_index = max(stable_indices) if stable_indices else None

    fieldnames = list(step_rows[0])
    with (HERE / "B48_NEW_MODE_STEP_REFINEMENT.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(step_rows)

    qualification_rows = []
    if confirmation_index is not None:
        for mode in range(33, 49):
            error = rel(native[mode], fd[mode, confirmation_index])
            qualification_rows.append({
                "mode": mode,
                "confirmation_step_label": ["h0", "h0/2", "h0/4"][confirmation_index],
                "confirmation_step": STEPS[confirmation_index],
                "native_column_norm": float(np.linalg.norm(native[mode])),
                "fd_column_norm": float(np.linalg.norm(fd[mode, confirmation_index])),
                "native_vs_fd_relative_error": error,
                "tolerance": TOL,
                "passed": error <= TOL,
            })
    with (HERE / "B48_NEW_MODE_FD_QUALIFICATION.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(qualification_rows[0]) if qualification_rows else [
            "mode", "confirmation_step_label", "confirmation_step", "native_column_norm",
            "fd_column_norm", "native_vs_fd_relative_error", "tolerance", "passed",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(qualification_rows)

    errors = np.asarray([row["native_vs_fd_relative_error"] for row in qualification_rows])
    result = {
        "verdict": "PASS" if confirmation_index is not None and all(row["passed"] for row in qualification_rows) else "FAIL",
        "h0": H0,
        "representative_modes": REPRESENTATIVES,
        "preregistered_steps": STEPS,
        "inherited_relative_tolerance": TOL,
        "stable_candidates": {
            ["h0", "h0/2", "h0/4"][index]: stable for index, stable in candidates
        },
        "confirmation_step_label": None if confirmation_index is None else ["h0", "h0/2", "h0/4"][confirmation_index],
        "confirmation_step": None if confirmation_index is None else STEPS[confirmation_index],
        "new_mode_count": len(qualification_rows),
        "median_relative_error": None if not len(errors) else float(np.median(errors)),
        "p90_relative_error": None if not len(errors) else float(np.percentile(errors, 90)),
        "maximum_relative_error": None if not len(errors) else float(errors.max()),
        "worst_mode": None if not len(errors) else qualification_rows[int(np.argmax(errors))]["mode"],
        "all_new_modes_passed": bool(len(qualification_rows) == 16 and all(row["passed"] for row in qualification_rows)),
        "cancellation_diagnostic": "The inherited hook also emitted h0/8 raw files; those are retained but excluded from selection and gates per preregistration.",
    }
    (HERE / "B48_DERIVATIVE_QUALIFICATION.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["verdict"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
