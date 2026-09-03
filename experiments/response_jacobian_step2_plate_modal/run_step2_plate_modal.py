"""Run Step 2 fixed-connectivity thin-plate modal geometry diagnostic."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from plate_mesh import create_plate_mesh
from plate_modes import create_plate_modes
from plate_responses import C0, make_responses, response_diagnostics
from plate_solvers import run_all_methods

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROOT = PROJECT_ROOT
RESULT_DIR = PROJECT_ROOT / "results" / "response_jacobian_step2_plate_modal"
REPORT_PATH = RESULT_DIR / "STEP2_PLATE_MODAL_REPORT.md"
SUMMARY_PATH = RESULT_DIR / "summary.csv"
HISTORY_PATH = RESULT_DIR / "iteration_history.csv"
DIAG_PATH = RESULT_DIR / "diagnostics.json"


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (list, tuple, dict)):
                    out[key] = json.dumps(value, default=_json_default)
                elif value is None:
                    out[key] = ""
                else:
                    out[key] = value
            writer.writerow(out)


def _fmt_matrix(M) -> str:
    return np.array2string(np.asarray(M), precision=6, suppress_small=False)


def _method_table(rows: List[Dict[str, object]]) -> str:
    rows = sorted(rows, key=lambda r: float(r["final_residual_ratio"]))
    lines = ["| method | converged | diverged | iters | final ratio | best ratio | error to best |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['method_name']} | {r['converged']} | {r['diverged']} | {r['iterations_run']} | "
            f"{float(r['final_residual_ratio']):.3e} | {float(r['best_residual_ratio_seen']):.3e} | "
            f"{float(r.get('error_to_best_solution_norm') or 0.0):.3e} |"
        )
    return "\n".join(lines)


def _compact_table(rows: List[Dict[str, object]]) -> str:
    lines = [
        "case | rho(I-J0) | coupling | direct_ratio | oracle_scalar_ratio | diagonal_ratio | trust_region_ratio | GO/NO-GO",
        "---|---:|---:|---:|---:|---:|---:|---",
    ]
    for r in rows:
        lines.append(
            f"{r['case']} | {r['rho(I-J0)']:.3f} | {r['coupling']:.3f} | {r['direct_ratio']:.3e} | "
            f"{r['oracle_scalar_ratio']:.3e} | {r['diagonal_ratio']:.3e} | {r['trust_region_ratio']:.3e} | {r['go_no_go']}"
        )
    return "\n".join(lines)


def _checks(project_root: Path, mesh_diag, mode_diag, diagnostics, summary_rows) -> Dict[str, bool]:
    by_case: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in summary_rows:
        by_case.setdefault(row["case_name"], {})[row["method_name"]] = row
    hard_cases = [name for name in diagnostics if name.startswith("geom_hard")]
    checks = {
        "project_root_is_correct": bool(project_root.resolve() == EXPECTED_ROOT.resolve()),
        "plate_mesh_expected_node_count": bool(mesh_diag["num_nodes"] == 31 * 25),
        "basis_orthonormality_close_to_I": bool(mode_diag["orthonormality_error_fro"] < 1e-10),
        "frozen_response_J_close_to_I": bool(diagnostics["frozen_response"]["identity_deviation"] < 1e-6),
        "hard_case_has_rho_gt_one": any(diagnostics[name]["rho_I_minus_J0"] > 1.0 for name in hard_cases),
        "direct_divergence_not_labeled_converged": all(not by_case[name]["direct_modal_inversion"]["converged"] for name in hard_cases if by_case[name]["direct_modal_inversion"]["diverged"]),
        "trust_region_improves_over_direct_in_hard_cases": all(
            float(by_case[name]["full_jacobian_trust_region"]["final_residual_ratio"]) < float(by_case[name]["direct_modal_inversion"]["final_residual_ratio"])
            for name in hard_cases
        ),
    }
    return checks


def _go_no_go(checks, diagnostics, summary_rows) -> Dict[str, object]:
    by_case: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in summary_rows:
        by_case.setdefault(row["case_name"], {})[row["method_name"]] = row
    hard_names = [name for name in diagnostics if name.startswith("geom_hard")]
    criteria = {
        "frozen_J_close_to_I": diagnostics["frozen_response"]["identity_deviation"] < 1e-6,
        "frozen_direct_succeeds": bool(by_case["frozen_response"]["direct_modal_inversion"]["converged"]),
        "at_least_one_hard_rho_gt_one": any(diagnostics[name]["rho_I_minus_J0"] > 1.0 for name in hard_names),
        "direct_fails_or_unstable_in_hard_case": any(
            by_case[name]["direct_modal_inversion"]["diverged"] or float(by_case[name]["direct_modal_inversion"]["final_residual_ratio"]) > 1.0
            for name in hard_names
        ),
        "trust_region_succeeds_or_improves": all(
            by_case[name]["full_jacobian_trust_region"]["converged"] or
            float(by_case[name]["full_jacobian_trust_region"]["final_residual_ratio"]) < float(by_case[name]["direct_modal_inversion"]["final_residual_ratio"])
            for name in hard_names
        ),
        "modal_basis_normalization_passes": checks["basis_orthonormality_close_to_I"],
    }
    return {"decision": "GO" if all(criteria.values()) and all(checks.values()) else "NO-GO", "criteria": criteria}


def _interpret(summary_rows, diagnostics) -> List[str]:
    by_case: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in summary_rows:
        by_case.setdefault(row["case_name"], {})[row["method_name"]] = row
    scalar_sensitive = False
    for methods in by_case.values():
        ratios = [float(v["final_residual_ratio"]) for k, v in methods.items() if k.startswith("scalar_alpha_") and math.isfinite(float(v["final_residual_ratio"]))]
        if ratios and max(ratios) / max(min(ratios), 1e-15) > 1e3:
            scalar_sensitive = True
    hard_direct_failed = any(
        by_case[name]["direct_modal_inversion"]["diverged"] or float(by_case[name]["direct_modal_inversion"]["final_residual_ratio"]) > 1.0
        for name in diagnostics if name.startswith("geom_hard")
    )
    diagonal_under = any(
        diagnostics[name]["coupling_ratio"] > 0.2 and
        float(by_case[name]["diagonal_modal_factor_initial"]["final_residual_ratio"]) > 10.0 * float(by_case[name]["full_jacobian_trust_region"]["final_residual_ratio"])
        for name in diagnostics
    )
    return [
        f"- Frozen response produced J approximately I: {'YES' if diagnostics['frozen_response']['identity_deviation'] < 1e-6 else 'NO'}.",
        f"- Direct inversion worked in frozen_response: {'YES' if by_case['frozen_response']['direct_modal_inversion']['converged'] else 'NO'}.",
        f"- Geometry-dependent hard cases produced rho(I-J0) > 1: {'YES' if any(diagnostics[n]['rho_I_minus_J0'] > 1 for n in diagnostics if n.startswith('geom_hard')) else 'NO'}.",
        f"- Direct inversion failed or was unstable in hard cases: {'YES' if hard_direct_failed else 'NO'}.",
        f"- Scalar factor behavior was sensitive to alpha: {'YES' if scalar_sensitive else 'NO'}.",
        f"- Diagonal factor underperformed when coupling was high: {'YES' if diagonal_under else 'NO'}.",
        f"- Full Jacobian trust-region remained stable: {'YES' if all(float(by_case[n]['full_jacobian_trust_region']['final_residual_ratio']) < 1e-6 for n in by_case) else 'NO'}.",
    ]


def _build_report(mesh_diag, mode_diag, diagnostics, summary_rows, checks, go_info, compact_rows) -> str:
    lines = [
        "# Step 2 Plate Modal Geometry Diagnostic",
        "",
        "## Purpose",
        "Step 2 moves the response-Jacobian diagnostic from pure modal-coordinate algebra into a fixed-connectivity thin-plate geometry space. The geometry is parameterized as `X(c) = X0 + B c`, residuals are projected as `b(c) = B^T M r(c)`, and all solver behavior is still evaluated in a controlled non-FEM setting.",
        "",
        "## Difference From Step 1",
        "Step 1 directly prescribed `b(c)` in low-dimensional modal coordinates. Step 2 constructs a plate mesh, analytical geometry modes, lumped area weights, physical residual fields, and then recovers the modal residual by projection. This verifies that the modal intervention and projection machinery behaves correctly before any FEM/JAX work.",
        "",
        "## Plate Mesh",
        f"- nodes: {mesh_diag['num_nodes']}",
        f"- DOFs: {mesh_diag['num_dofs']}",
        f"- Lx, Ly: {mesh_diag['Lx']}, {mesh_diag['Ly']}",
        f"- nx, ny: {mesh_diag['nx']}, {mesh_diag['ny']}",
        f"- dx, dy: {mesh_diag['dx']:.6g}, {mesh_diag['dy']:.6g}",
        f"- total area: {mesh_diag['total_area']:.6g}",
        f"- min coordinates: {mesh_diag['min_coordinates']}",
        f"- max coordinates: {mesh_diag['max_coordinates']}",
        "",
        "## Modal Basis",
        f"- k: {mode_diag['k']}",
        f"- orthonormality error `||B^T M B - I||_F`: {mode_diag['orthonormality_error_fro']:.6e}",
    ]
    for desc in mode_diag["mode_descriptions"]:
        lines.append(f"- {desc}")
    lines.extend([
        "",
        "## Response Definitions",
        "- `frozen_response`: `Xp = Xc + u0`, so `b(c) = c + B^T M u0` and `J approximately I`.",
        "- `geometry_dependent_response`: `Xp = Xc + B(G c_geom + beta h(c_geom) + q0)`, where `c_geom = B^T M (Xc - X0)`. This creates controlled gain mismatch, coupling, and one nonlinear hard case without FEM.",
        "",
        "## Internal Checks",
    ])
    for name, ok in checks.items():
        lines.append(f"- {name}: {'PASS' if ok else 'FAIL'}")
    lines.extend(["", "## Compact Result", _compact_table(compact_rows), ""])

    for case_name, d in diagnostics.items():
        rows = [r for r in summary_rows if r["case_name"] == case_name]
        eig_text = ", ".join(f"{e[0]:.6g}{'+' if e[1] >= 0 else ''}{e[1]:.6g}j" for e in d["eigenvalues"])
        lines.extend([
            f"## Case: {case_name}",
            "",
            f"- response type: {d['response_type']}",
            f"- beta: {d['beta']}",
            "- finite difference eps: {:.1e}".format(d["finite_difference_eps"]),
            f"- initial residual norm: {d['initial_residual_norm']:.6e}",
            f"- eigenvalues of J0: {eig_text}",
            f"- rho(I - J0): {d['rho_I_minus_J0']:.6e}",
            f"- identity deviation: {d['identity_deviation']:.6e}",
            f"- coupling ratio: {d['coupling_ratio']:.6e}",
            f"- condition number: {d['condition_number']:.6e}",
            "",
            "### J0 Matrix",
            "```text",
            _fmt_matrix(d["J0"]),
            "```",
            "",
            "### Method Summary",
            _method_table(rows),
            "",
        ])
    lines.extend(["## Automatic Interpretation", *_interpret(summary_rows, diagnostics), "", "## GO / NO-GO", f"Decision: **{go_info['decision']}**"])
    for key, value in go_info["criteria"].items():
        lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    mesh = create_plate_mesh()
    B, mode_diag = create_plate_modes(mesh)
    mesh_diag = mesh.diagnostics()
    responses = make_responses(mesh, B)
    diagnostics = {name: response_diagnostics(resp, mode_diag["orthonormality_error_fro"]) for name, resp in responses.items()}

    summary_rows: List[Dict[str, object]] = []
    history_rows: List[Dict[str, object]] = []
    for response in responses.values():
        results, histories = run_all_methods(response)
        summary_rows.extend([r.as_row() for r in results])
        history_rows.extend(histories)

    checks = _checks(PROJECT_ROOT, mesh_diag, mode_diag, diagnostics, summary_rows)
    go_info = _go_no_go(checks, diagnostics, summary_rows)

    compact_rows = []
    for case_name in responses.keys():
        rows = {r["method_name"]: r for r in summary_rows if r["case_name"] == case_name}
        compact_rows.append({
            "case": case_name,
            "rho(I-J0)": float(diagnostics[case_name]["rho_I_minus_J0"]),
            "coupling": float(diagnostics[case_name]["coupling_ratio"]),
            "direct_ratio": float(rows["direct_modal_inversion"]["final_residual_ratio"]),
            "oracle_scalar_ratio": float(rows["oracle_scalar_scale_factor"]["final_residual_ratio"]),
            "diagonal_ratio": float(rows["diagonal_modal_factor_initial"]["final_residual_ratio"]),
            "trust_region_ratio": float(rows["full_jacobian_trust_region"]["final_residual_ratio"]),
            "go_no_go": go_info["decision"],
        })

    _write_csv(SUMMARY_PATH, summary_rows)
    _write_csv(HISTORY_PATH, history_rows)
    DIAG_PATH.write_text(json.dumps({"mesh": mesh_diag, "basis": mode_diag, "cases": diagnostics, "checks": checks, "go_no_go": go_info}, indent=2, default=_json_default), encoding="utf-8")
    REPORT_PATH.write_text(_build_report(mesh_diag, mode_diag, diagnostics, summary_rows, checks, go_info, compact_rows), encoding="utf-8")

    print(f"STEP2_PLATE_MODAL_REPORT.md: {REPORT_PATH}")
    print(f"summary.csv: {SUMMARY_PATH}")
    print(f"iteration_history.csv: {HISTORY_PATH}")
    print(f"diagnostics.json: {DIAG_PATH}")
    print()
    print("Internal checks:")
    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print()
    print(_compact_table(compact_rows))
    print()
    print(f"Overall Step 2: {go_info['decision']}")


if __name__ == "__main__":
    main()
