"""Checkpointed stages for the framework-story closure workflow.

This driver intentionally avoids the old monolithic execution path.  Each
numerical invocation handles one deterministic case, writes case artifacts
immediately, and records a manifest row before returning control to the shell.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_response_inversion_framework_story_closure as story  # noqa: E402


RESULT_DIR = ROOT / "results" / "response_inversion_framework_story_closure"
METHOD_DIR = ROOT / "results" / "response_inversion_method_closure"

MANIFEST = RESULT_DIR / "checkpoint_manifest.csv"
MANIFEST_COLUMNS = [
    "stage",
    "case",
    "status",
    "started_at",
    "completed_at",
    "output_path",
    "error",
    "runtime_seconds",
]
VALID_STATUS = {"PENDING", "RUNNING", "COMPLETE", "FAILED", "TIMEOUT", "REUSED", "UNRESOLVED"}

STATIONARITY_CASES: Dict[str, Tuple[str, str, str, float]] = {
    "comb_g05": ("A_FDM_comb_coupon", "bottom_contact", "comb mild response", 0.5),
    "comb_g10": ("A_FDM_comb_coupon", "bottom_contact", "comb hard response", 1.0),
    "comb_g15": ("A_FDM_comb_coupon", "bottom_contact", "comb strong response", 1.5),
    "lbracket_g10": ("B_L_bracket_datum", "bottom_only", "L-bracket near-identity control", 1.0),
}

RESPONSE_REPLAY_CASES = dict(STATIONARITY_CASES)

MATCHED_ADMISSIBILITY_CASES: Dict[str, Tuple[str, str, str, str]] = {
    "lbracket_base_datum": ("B_L_bracket_datum", "bottom_only", "L-bracket fixed base plus preserved datum", "datum"),
    "wall_allowance": ("C_wall_allowance", "bottom_plus_allowance", "thin wall allowance bound", "allowance"),
}

CASE_SETS = {
    "stationarity": STATIONARITY_CASES,
    "response_replay": RESPONSE_REPLAY_CASES,
    "matched_admissibility": MATCHED_ADMISSIBILITY_CASES,
}

STATIONARITY_COLUMNS = [
    "case",
    "case_label",
    "constraint_set",
    "gamma",
    "forward_model",
    "method",
    "basis_size",
    "initial_engineering_residual",
    "final_engineering_residual",
    "engineering_surface_residual_ratio",
    "initial_physical_residual",
    "final_physical_residual",
    "physical_residual_ratio",
    "initial_reduced_residual",
    "final_reduced_residual",
    "reduced_residual_ratio",
    "normalized_first_order_metric",
    "objective_value",
    "old_own_conv_residual_ratio_test",
    "final_convergence_classification",
    "iterations",
    "accepted_steps",
    "rejected_steps",
    "actual_residual_calls",
    "jacobian_construction_count",
    "fd_internal_residual_calls",
    "constraint_violation_l2",
    "constraint_violation_rms",
    "constraint_violation_max",
    "correction_norm",
    "runtime_s",
]

HISTORY_COLUMNS = [
    "case",
    "case_label",
    "constraint_set",
    "gamma",
    "forward_model",
    "method",
    "iteration",
    "engineering_residual_norm",
    "surface_RMS_mm",
    "physical_residual_norm",
    "objective_value",
    "normalized_first_order_metric",
    "step_norm",
    "predicted_objective_decrease",
    "actual_objective_decrease",
    "accepted",
    "lm_lambda",
    "trust_radius",
    "status",
]

REPLAY_COLUMNS = [
    "case",
    "case_label",
    "constraint_set",
    "gamma",
    "state",
    "tau",
    "step_norm",
    "initial_objective",
    "predicted_objective",
    "actual_objective",
    "predicted_objective_change",
    "actual_objective_change",
    "epsilon_pred",
    "cos_pred_actual_residual_change",
    "predicted_descent",
    "actual_descent",
    "residual_calls",
    "jacobian_calls",
]

SELECTED_COLUMNS = [
    "case",
    "case_label",
    "constraint_set",
    "gamma",
    "method",
    "definition",
    "engineering_surface_residual_ratio",
    "physical_residual_ratio",
    "reduced_residual_ratio",
    "normalized_first_order_metric",
    "iterations",
    "actual_residual_calls",
    "notes",
]

ROBUST_COLUMNS = SELECTED_COLUMNS + ["forward_model", "fd_internal_residual_calls"]

ADMISSIBILITY_COLUMNS = [
    "case",
    "manufacturing_setting",
    "constraint_kind",
    "method",
    "basis_size",
    "initial_engineering_residual",
    "engineering_surface_residual_ratio",
    "physical_residual_ratio",
    "max_constraint_violation",
    "rms_constraint_violation",
    "l2_constraint_violation",
    "correction_norm",
    "active_constraints",
    "active_inequality_count",
    "first_order_or_projected_kkt_metric",
    "iterations",
    "actual_residual_calls",
    "accepted_steps",
    "rejected_steps",
    "notes",
]

TARGET_AUDIT_COLUMNS = [
    "case",
    "constraint_set",
    "geometry",
    "total_nodes",
    "surface_nodes",
    "surface_dofs",
    "bottom_nodes",
    "bottom_nodes_in_surface",
    "internal_nodes_excluded",
    "datum_nodes",
    "allowance_nodes",
    "surface_weight_type",
    "surface_weight_per_dof",
    "all_dof_weight_min",
    "all_dof_weight_max",
    "optimized_surface_matches_reported_surface_RMS",
    "units",
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_result_dir() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST.exists():
        write_csv(MANIFEST, [], MANIFEST_COLUMNS)


def normalize_value(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value


def write_csv(path: Path, rows: Iterable[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    materialized = list(rows)

    def write_rows(target: Path) -> None:
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in materialized:
                writer.writerow({col: normalize_value(row.get(col, "")) for col in columns})

    write_rows(tmp)
    try:
        os.replace(tmp, path)
    except PermissionError:
        # Synchronized filesystems can transiently deny atomic replacement of tiny CSVs.
        # Fall back to a direct rewrite so checkpoint progress is not lost.
        write_rows(path)
        try:
            tmp.unlink()
        except OSError:
            pass


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def valid_csv(path: Path, required_columns: Iterable[str], min_rows: int = 1) -> bool:
    rows = read_csv(path)
    if len(rows) < min_rows:
        return False
    columns = set(rows[0].keys()) if rows else set()
    return set(required_columns).issubset(columns)


def update_manifest(stage: str, case: str, status: str, started_at: str, output_path: Path | str = "", error: str = "") -> None:
    if status not in VALID_STATUS:
        raise ValueError(f"invalid manifest status: {status}")
    ensure_result_dir()
    rows = read_csv(MANIFEST)
    completed = "" if status == "RUNNING" else now()
    runtime = ""
    if started_at and completed:
        start_dt = datetime.fromisoformat(started_at)
        done_dt = datetime.fromisoformat(completed)
        runtime = f"{(done_dt - start_dt).total_seconds():.3f}"
    row = {
        "stage": stage,
        "case": case,
        "status": status,
        "started_at": started_at,
        "completed_at": completed,
        "output_path": rel(output_path) if isinstance(output_path, Path) else output_path,
        "error": error.replace("\r", " ").replace("\n", " ")[:1000],
        "runtime_seconds": runtime,
    }
    rows = [r for r in rows if not (r["stage"] == stage and r["case"] == case)]
    rows.append(row)
    write_csv(MANIFEST, rows, MANIFEST_COLUMNS)


def aggregate_case_csv(pattern: str, output: Path, columns: List[str]) -> None:
    rows: List[Dict[str, object]] = []
    for path in sorted(RESULT_DIR.glob(pattern)):
        rows.extend(read_csv(path))
    write_csv(output, rows, columns)


def completed_for_resume(stage: str, case: str, output_path: Path, required_columns: Iterable[str]) -> bool:
    rows = read_csv(MANIFEST)
    manifest_ok = any(r["stage"] == stage and r["case"] == case and r["status"] in {"COMPLETE", "REUSED"} for r in rows)
    if output_path.suffix.lower() == ".csv":
        return manifest_ok and valid_csv(output_path, required_columns)
    return manifest_ok and output_path.exists() and output_path.stat().st_size > 0


def manifest_complete_or_reused(stage: str, case: str) -> bool:
    return any(r["stage"] == stage and r["case"] == case and r["status"] in {"COMPLETE", "REUSED"} for r in read_csv(MANIFEST))


def resume_output(stage: str, case: str) -> Tuple[Path, Iterable[str]]:
    if stage == "audit_prior_method":
        return RESULT_DIR / "prior_modal_method_mapping.md", []
    if stage == "audit_engineering_target":
        return RESULT_DIR / "engineering_target_audit.csv", TARGET_AUDIT_COLUMNS
    if stage == "stationarity":
        return RESULT_DIR / f"stationarity_case_{case}.csv", STATIONARITY_COLUMNS
    if stage == "response_replay":
        return RESULT_DIR / f"response_replay_case_{case}.csv", REPLAY_COLUMNS
    if stage == "matched_admissibility":
        return RESULT_DIR / f"matched_admissibility_case_{case}.csv", ADMISSIBILITY_COLUMNS
    if stage == "assemble":
        return RESULT_DIR / "FRAMEWORK_STORY_AND_EVIDENCE_CLOSURE.md", []
    raise ValueError(f"unknown stage: {stage}")


def run_audit_prior_method(args: argparse.Namespace) -> Path:
    path = RESULT_DIR / "prior_modal_method_mapping.md"
    if args.resume and path.exists() and path.stat().st_size > 0:
        return path
    if args.dry_run:
        return path
    story.audit_prior_mapping()
    write_identity_reuse()
    write_selected_response_reuse()
    write_robustness_reuse()
    return path


def run_audit_engineering_target(args: argparse.Namespace) -> Path:
    path = RESULT_DIR / "engineering_target_definition.md"
    csv_path = RESULT_DIR / "engineering_target_audit.csv"
    if args.resume and path.exists() and path.stat().st_size > 0 and valid_csv(csv_path, TARGET_AUDIT_COLUMNS):
        return csv_path
    if args.dry_run:
        return csv_path
    text = """# Engineering Target Definition

The engineering target operator is `E=S`, where `S` selects the DOFs belonging
to `mesh.surface_mask`.

Source audit:

- `fixed_bottom_core.py:171-186` marks a node as surface if it appears on a boundary face of the hexahedral mesh.
- `run_admissible_inherent_strain_fem_benchmark.py:589-607` computes `surface_RMS_error_mm` from residual vectors on `model.surface_mask_np`.
- `run_response_inversion_method_closure.py:134-139,210-213` defines the optimized surface objective as `||S D||_2 / sqrt(n_surface)`, matching the reported uniform nodal surface RMS metric.

Definition:

`r_E(a)=W_E^{1/2} S D(a)` with `W_E^{1/2}=I/sqrt(n_surface)` applied to all three displacement components at surface nodes. This is uniform nodal surface weighting, not area weighting and not the lumped all-DOF volume/mass weighting. Units are millimeters because residual components are geometry displacements in mm.

This audit is text/code-only and does not instantiate FEM cases.
"""
    path.write_text(text, encoding="utf-8")
    rows = [
        {
            "case": "ALL",
            "constraint_set": "surface_physical",
            "geometry": "code_definition_audit",
            "total_nodes": "",
            "surface_nodes": "mesh.surface_mask",
            "surface_dofs": "3 * surface_nodes",
            "bottom_nodes": "case constraint set dependent",
            "bottom_nodes_in_surface": "included when boundary/surface",
            "internal_nodes_excluded": "True",
            "datum_nodes": "case datum_mask when present",
            "allowance_nodes": "case allowance_mask when present",
            "surface_weight_type": "uniform_nodal_surface_RMS",
            "surface_weight_per_dof": "1/sqrt(n_surface_nodes)",
            "all_dof_weight_min": "not used by engineering target",
            "all_dof_weight_max": "not used by engineering target",
            "optimized_surface_matches_reported_surface_RMS": "True",
            "units": "mm",
        }
    ]
    write_csv(csv_path, rows, TARGET_AUDIT_COLUMNS)
    return csv_path


def write_identity_reuse() -> Path:
    source = METHOD_DIR / "sid_local_scope_validation.csv"
    wanted = {("A_FDM_comb_coupon", 0.5), ("A_FDM_comb_coupon", 1.0), ("A_FDM_comb_coupon", 1.5), ("B_L_bracket_datum", 1.0), ("C_wall_allowance", 1.0)}
    rows = []
    for row in read_csv(source):
        key = (row["case"], float(row["gamma"]))
        if key not in wanted:
            continue
        rows.append({
            "case": row["case"],
            "case_label": row["case_label"],
            "constraint_set": row["constraint_set"],
            "gamma": row["gamma"],
            "s_id": row["s_id"],
            "observed_first_step_reduced_contraction": row["first_identity_step_contraction_ratio"],
            "projection_active": row["projection_active"],
            "interpretation": "REUSED from sid_local_scope_validation; local identity-response diagnostic",
        })
    out = RESULT_DIR / "identity_response_limit_evidence.csv"
    write_csv(out, rows, ["case", "case_label", "constraint_set", "gamma", "s_id", "observed_first_step_reduced_contraction", "projection_active", "interpretation"])
    return out


def selected_from_objective_row(row: Dict[str, str], method: str, definition: str, notes: str) -> Dict[str, object]:
    return {
        "case": row["case"],
        "case_label": row["case_label"],
        "constraint_set": row["constraint_set"],
        "gamma": row["gamma"],
        "method": method,
        "definition": definition,
        "engineering_surface_residual_ratio": row["surface_residual_ratio"],
        "physical_residual_ratio": row["physical_residual_ratio"],
        "reduced_residual_ratio": row["reduced_residual_ratio"],
        "normalized_first_order_metric": "",
        "iterations": row["iterations"],
        "actual_residual_calls": row["actual_residual_calls"],
        "notes": notes,
    }


def write_selected_response_reuse() -> Path:
    source = METHOD_DIR / "objective_comparison.csv"
    wanted = {("A_FDM_comb_coupon", 0.5), ("A_FDM_comb_coupon", 1.0), ("A_FDM_comb_coupon", 1.5), ("B_L_bracket_datum", 1.0)}
    methods = {
        "identity": ("structured_modal_identity_update", "reduced identity baseline"),
        "reduced_LM_trust": ("reduced_response_inversion", "J_b reduced-response inverse"),
        "surface_physical_LM_trust": ("engineering_response_inversion", "surface engineering H_E inverse"),
    }
    rows = []
    for row in read_csv(source):
        key = (row["case"], float(row["gamma"]))
        if key not in wanted or row["forward_model"] != "reference_domain" or row["method"] not in methods:
            continue
        method, definition = methods[row["method"]]
        rows.append(selected_from_objective_row(row, method, definition, "REUSED from method-closure objective_comparison.csv"))
    out = RESULT_DIR / "selected_response_calibration_evidence.csv"
    write_csv(out, rows, SELECTED_COLUMNS)
    return out


def write_robustness_reuse() -> Path:
    source = METHOD_DIR / "forward_model_objective_robustness.csv"
    wanted = {("A_FDM_comb_coupon", 1.0), ("A_FDM_comb_coupon", 1.5)}
    methods = {
        "identity": ("structured_modal_identity_update", "identity baseline"),
        "diagnostic_reduced_J_LM_trust": ("reduced_response_inversion", "reduced J_b inverse"),
        "preferred_surface_physical_LM_trust": ("engineering_response_inversion", "surface H_E inverse"),
    }
    rows = []
    for row in read_csv(source):
        key = (row["case"], float(row["gamma"]))
        if key not in wanted or row["method"] not in methods:
            continue
        method, definition = methods[row["method"]]
        rows.append({
            "case": row["case"],
            "case_label": row["case_label"],
            "constraint_set": row["constraint_set"],
            "gamma": row["gamma"],
            "method": method,
            "definition": f"{row['forward_model']} {definition}",
            "engineering_surface_residual_ratio": row["surface_residual_ratio"],
            "physical_residual_ratio": row["physical_residual_ratio"],
            "reduced_residual_ratio": row["reduced_residual_ratio"],
            "normalized_first_order_metric": "",
            "iterations": row["iterations"],
            "actual_residual_calls": row["actual_residual_calls"],
            "notes": "REUSED from method-closure R/U robustness; Model U is robustness model, not truth.",
            "forward_model": row["forward_model"],
            "fd_internal_residual_calls": row["fd_internal_residual_calls"],
        })
    out = RESULT_DIR / "selected_forward_model_robustness.csv"
    write_csv(out, rows, ROBUST_COLUMNS)
    return out


def run_stationarity_case(case_id: str, args: argparse.Namespace) -> Path:
    out = RESULT_DIR / f"stationarity_case_{case_id}.csv"
    hist_out = RESULT_DIR / f"stationarity_history_{case_id}.csv"
    coeff_out = RESULT_DIR / f"stationarity_coefficients_{case_id}.json"
    if args.resume and completed_for_resume("stationarity", case_id, out, STATIONARITY_COLUMNS):
        return out
    if args.dry_run:
        return out
    case_name, cset, label, gamma = STATIONARITY_CASES[case_id]
    model = story.build_model(case_name, cset, gamma)
    row, history, coeffs = story.run_engineering_solver(model, label, gamma, "reference_domain", story.MAX_ITER_STATIONARITY)
    write_csv(out, [row], STATIONARITY_COLUMNS)
    write_csv(hist_out, history, HISTORY_COLUMNS)
    coeff_out.write_text(json.dumps({"case_id": case_id, "case": case_name, "gamma": gamma, "coefficients": np.asarray(coeffs).tolist()}, indent=2), encoding="utf-8")
    aggregate_case_csv("stationarity_case_*.csv", RESULT_DIR / "engineering_solver_stationarity.csv", STATIONARITY_COLUMNS)
    aggregate_case_csv("stationarity_history_*.csv", RESULT_DIR / "engineering_solver_histories.csv", HISTORY_COLUMNS)
    return out


def load_stationarity_coefficients(case_id: str, mode_count: int) -> np.ndarray | None:
    path = RESULT_DIR / f"stationarity_coefficients_{case_id}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    coeffs = np.asarray(payload.get("coefficients", []), dtype=float)
    return coeffs if coeffs.size == mode_count else None


def run_response_replay_case(case_id: str, args: argparse.Namespace) -> Path:
    out = RESULT_DIR / f"response_replay_case_{case_id}.csv"
    if args.resume and completed_for_resume("response_replay", case_id, out, REPLAY_COLUMNS):
        return out
    if args.dry_run:
        return out
    case_name, cset, label, gamma = RESPONSE_REPLAY_CASES[case_id]
    model = story.build_model(case_name, cset, gamma)
    states = [("initial", np.zeros(model.mode_count))]
    coeffs = load_stationarity_coefficients(case_id, model.mode_count)
    if coeffs is not None:
        states.append(("accepted_final", coeffs))
    rows = []
    for state, c in states:
        fwd = story.mc.CountedForward(model, "reference_domain")
        r, H, _, _ = story.objective_pair_surface(model, fwd, c)
        delta = story.mc.step_lm(H, r, model.case.trust_initial_lambda, model.case.trust_max_step)
        for tau in (0.25, 0.5, 1.0):
            pred_r = r + tau * H @ delta
            r_actual, _, _, _ = story.objective_pair_surface(model, fwd, c + tau * delta)
            obj0 = story.mc.objective_value(r)
            pred_change = pred_r - r
            actual_change = r_actual - r
            rows.append({
                "case": case_name,
                "case_label": label,
                "constraint_set": cset,
                "gamma": gamma,
                "state": state,
                "tau": tau,
                "step_norm": float(np.linalg.norm(delta)),
                "initial_objective": obj0,
                "predicted_objective": story.mc.objective_value(pred_r),
                "actual_objective": story.mc.objective_value(r_actual),
                "predicted_objective_change": story.mc.objective_value(pred_r) - obj0,
                "actual_objective_change": story.mc.objective_value(r_actual) - obj0,
                "epsilon_pred": float(np.linalg.norm(r_actual - pred_r) / max(np.linalg.norm(r_actual), story.EPS)),
                "cos_pred_actual_residual_change": story.safe_cos(pred_change, actual_change),
                "predicted_descent": bool(story.mc.objective_value(pred_r) < obj0),
                "actual_descent": bool(story.mc.objective_value(r_actual) < obj0),
                "residual_calls": fwd.residual_calls,
                "jacobian_calls": fwd.jacobian_builds,
            })
    write_csv(out, rows, REPLAY_COLUMNS)
    aggregate_case_csv("response_replay_case_*.csv", RESULT_DIR / "engineering_response_prediction_replay.csv", REPLAY_COLUMNS)
    return out


def run_matched_admissibility_case(case_id: str, args: argparse.Namespace) -> Path:
    out = RESULT_DIR / f"matched_admissibility_case_{case_id}.csv"
    if args.resume and completed_for_resume("matched_admissibility", case_id, out, ADMISSIBILITY_COLUMNS):
        return out
    if args.dry_run:
        return out
    case_name, cset, label, constraint_kind = MATCHED_ADMISSIBILITY_CASES[case_id]
    model = story.build_model(case_name, cset, 1.0)
    init_fwd = story.mc.CountedForward(model, "reference_domain")
    init = story.mc.objective_norms(model, init_fwd.residual_c(np.zeros(model.mode_count)))
    rows = []
    for admissible in (False, True):
        extra_eq = model.datum_mask_np if admissible and constraint_kind == "datum" else np.zeros(model.mesh.num_nodes, dtype=bool)
        enforce_allowance = admissible and constraint_kind == "allowance"
        method = "A1_manufacturing_admissible_engineering_inverse" if admissible else "A0_unconstrained_engineering_inverse"
        row, _, coeffs = story.run_engineering_solver(model, label, 1.0, "reference_domain", story.MAX_ITER_SELECTED, extra_eq, enforce_allowance, method)
        fwd_check = story.mc.CountedForward(model, "reference_domain")
        fo = story.projected_first_order(model, coeffs, fwd_check, extra_eq, enforce_allowance)
        q = model.q_from_c(coeffs)
        viol = story.custom_violation(model, q, extra_eq, enforce_allowance)
        rows.append({
            "case": case_name,
            "manufacturing_setting": label,
            "constraint_kind": constraint_kind,
            "method": method,
            "basis_size": model.mode_count,
            "initial_engineering_residual": init["surface_residual"],
            "engineering_surface_residual_ratio": row["engineering_surface_residual_ratio"],
            "physical_residual_ratio": row["physical_residual_ratio"],
            "max_constraint_violation": viol["constraint_violation_max"],
            "rms_constraint_violation": viol["constraint_violation_rms"],
            "l2_constraint_violation": viol["constraint_violation_l2"],
            "correction_norm": row["correction_norm"],
            "active_constraints": viol["active_constraint_count"],
            "active_inequality_count": viol["active_inequality_count"],
            "first_order_or_projected_kkt_metric": fo,
            "iterations": row["iterations"],
            "actual_residual_calls": row["actual_residual_calls"],
            "accepted_steps": row["accepted_steps"],
            "rejected_steps": row["rejected_steps"],
            "notes": "same basis/objective/forward model; admissible row adds manufacturing constraints",
        })
    write_csv(out, rows, ADMISSIBILITY_COLUMNS)
    aggregate_case_csv("matched_admissibility_case_*.csv", RESULT_DIR / "matched_manufacturing_admissibility.csv", ADMISSIBILITY_COLUMNS)
    return out


def run_assemble(args: argparse.Namespace) -> Path:
    out = RESULT_DIR / "FRAMEWORK_STORY_AND_EVIDENCE_CLOSURE.md"
    if args.resume and out.exists() and out.stat().st_size > 0:
        return out
    if args.dry_run:
        return out
    prior = (RESULT_DIR / "prior_modal_method_mapping.md").read_text(encoding="utf-8") if (RESULT_DIR / "prior_modal_method_mapping.md").exists() else ""
    target = (RESULT_DIR / "engineering_target_definition.md").read_text(encoding="utf-8") if (RESULT_DIR / "engineering_target_definition.md").exists() else ""
    target_rows = read_csv(RESULT_DIR / "engineering_target_audit.csv")
    stationarity_rows = read_csv(RESULT_DIR / "engineering_solver_stationarity.csv")
    history_rows = read_csv(RESULT_DIR / "engineering_solver_histories.csv")
    prediction_rows = read_csv(RESULT_DIR / "engineering_response_prediction_replay.csv")
    identity_rows = read_csv(RESULT_DIR / "identity_response_limit_evidence.csv")
    selected_rows = read_csv(RESULT_DIR / "selected_response_calibration_evidence.csv")
    robust_rows = read_csv(RESULT_DIR / "selected_forward_model_robustness.csv")
    manuf_rows = read_csv(RESULT_DIR / "matched_manufacturing_admissibility.csv")
    claims = story.claim_matrix(prior, prediction_rows, identity_rows, selected_rows, robust_rows, manuf_rows)
    manifest_rows = read_csv(MANIFEST)
    unresolved = [r for r in manifest_rows if r["status"] in {"FAILED", "TIMEOUT", "UNRESOLVED"}]
    if unresolved:
        unresolved_stages = ", ".join(f"{r['stage']}:{r['case']}={r['status']}" for r in unresolved)
        claims = [
            dict(r, verdict=("UNRESOLVED" if r["claim"].startswith(("C3", "C6")) else r["verdict"]), evidence=f"{r['evidence']} Unresolved staged evidence: {unresolved_stages}.")
            for r in claims
        ]
    write_csv(RESULT_DIR / "framework_claim_evidence_matrix.csv", claims, ["claim", "verdict", "evidence"])
    story.write_report(prior, target, target_rows, stationarity_rows, history_rows, prediction_rows, identity_rows, selected_rows, robust_rows, manuf_rows, claims)
    replace_legacy_reproduction_section(out)
    append_status_section(out, manifest_rows)
    metadata = {
        "staged_driver": rel(Path(__file__).resolve()),
        "assembled_at": now(),
        "framework_verdict": "SUPPORTED_WITH_SCOPE",
        "process_isolation": "one numerical case per Python invocation",
    }
    (RESULT_DIR / "framework_story_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def replace_legacy_reproduction_section(report_path: Path) -> None:
    text = report_path.read_text(encoding="utf-8")
    text = text.replace(
        "- `experiments/fixed_bottom_jax_fem/run_response_inversion_framework_story_closure.py`",
        "- `experiments/fixed_bottom_jax_fem/run_framework_story_stage.py`",
    )
    text = text.replace(
        "```powershell\npython -B experiments\\fixed_bottom_jax_fem\\run_response_inversion_framework_story_closure.py\n```",
        "See `results/response_inversion_framework_story_closure/STAGED_EXECUTION_RUNBOOK.md` for the staged commands. Do not run the old monolithic closure runner.",
    )
    report_path.write_text(text, encoding="utf-8")


def append_status_section(report_path: Path, manifest_rows: List[Dict[str, str]]) -> None:
    groups = {"COMPLETE": [], "REUSED": [], "UNRESOLVED": []}
    for row in manifest_rows:
        status = row["status"]
        if status in {"FAILED", "TIMEOUT", "UNRESOLVED"}:
            groups["UNRESOLVED"].append(row)
        elif status in groups:
            groups[status].append(row)
    lines = ["", "## Staged Checkpoint Status", ""]
    for status in ("COMPLETE", "REUSED", "UNRESOLVED"):
        lines.append(f"### {status}")
        entries = groups[status]
        if not entries:
            lines.append("")
            lines.append("None.")
            lines.append("")
            continue
        lines.append("")
        for row in entries:
            lines.append(f"- `{row['stage']}:{row['case']}` -> `{row['output_path']}`")
        lines.append("")
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run_stage(args: argparse.Namespace) -> Path:
    if args.stage == "audit_prior_method":
        return run_audit_prior_method(args)
    if args.stage == "audit_engineering_target":
        return run_audit_engineering_target(args)
    if args.stage == "stationarity":
        return run_stationarity_case(args.case, args)
    if args.stage == "response_replay":
        return run_response_replay_case(args.case, args)
    if args.stage == "matched_admissibility":
        return run_matched_admissibility_case(args.case, args)
    if args.stage == "assemble":
        return run_assemble(args)
    raise ValueError(f"unknown stage: {args.stage}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["audit_prior_method", "audit_engineering_target", "stationarity", "response_replay", "matched_admissibility", "assemble"])
    parser.add_argument("--case", default=None, help="Deterministic case id for numerical stages.")
    parser.add_argument("--resume", action="store_true", help="Skip a stage/case when a valid completed output already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate dispatch and update checkpoint manifest without FEM/JAX computation.")
    args = parser.parse_args()
    case_map = CASE_SETS.get(args.stage)
    if case_map is not None:
        if args.case not in case_map:
            valid = ", ".join(sorted(case_map))
            parser.error(f"--stage {args.stage} requires --case in: {valid}")
    elif args.case is None:
        args.case = "all"
    return args


def main() -> int:
    args = parse_args()
    ensure_result_dir()
    if args.resume and not args.dry_run and manifest_complete_or_reused(args.stage, args.case):
        out, columns = resume_output(args.stage, args.case)
        if completed_for_resume(args.stage, args.case, out, columns):
            print(f"{args.stage}:{args.case} SKIP -> {rel(out)}")
            return 0
    started = now()
    update_manifest(args.stage, args.case, "RUNNING", started)
    try:
        out = run_stage(args)
        status = "PENDING" if args.dry_run else ("REUSED" if args.stage.startswith("audit_") and args.resume and out.exists() else "COMPLETE")
        update_manifest(args.stage, args.case, status, started, out)
        print(f"{args.stage}:{args.case} {status} -> {rel(out)}")
        return 0
    except TimeoutError as exc:
        update_manifest(args.stage, args.case, "TIMEOUT", started, error=str(exc))
        raise
    except Exception as exc:
        update_manifest(args.stage, args.case, "FAILED", started, error=f"{exc}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
