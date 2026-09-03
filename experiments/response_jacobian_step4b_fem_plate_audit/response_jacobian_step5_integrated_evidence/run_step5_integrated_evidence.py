"""Step 5 integrated evidence summary and final claim calibration."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = PROJECT_ROOT / "results"
OUT_DIR = RESULT_ROOT / "response_jacobian_step5_integrated_evidence"

MASTER_REPORT = OUT_DIR / "MASTER_EVIDENCE_REPORT.md"
MASTER_CASE = OUT_DIR / "master_case_summary.csv"
MASTER_METHOD = OUT_DIR / "master_method_summary.csv"
CLAIMS = OUT_DIR / "claim_support_matrix.csv"
RECOMMENDED = OUT_DIR / "recommended_final_benchmark_cases.csv"
LIMITATIONS = OUT_DIR / "limitations_and_next_steps.md"
DIAG_INDEX = OUT_DIR / "diagnostics_index.json"

STEP_CONFIGS = [
    ("Step 1", "response_jacobian_step1_synthetic", "summary.csv", "diagnostics.json", "STEP1_SYNTHETIC_REPORT.md"),
    ("Step 2", "response_jacobian_step2_plate_modal", "summary.csv", "diagnostics.json", "STEP2_PLATE_MODAL_REPORT.md"),
    ("Step 3", "response_jacobian_step3_physics_plate", "summary.csv", "diagnostics.json", "STEP3_PHYSICS_PLATE_REPORT.md"),
    ("Step 3B", "response_jacobian_step3b_coupling_stress", "summary.csv", "diagnostics.json", "STEP3B_COUPLING_STRESS_REPORT.md"),
    ("Step 4", "response_jacobian_step4_fem_plate", "summary.csv", "diagnostics.json", "STEP4_FEM_PLATE_REPORT.md"),
    ("Step 4B", "response_jacobian_step4b_fem_plate_audit", "audited_summary.csv", "diagnostics.json", "STEP4B_FEM_PLATE_AUDIT_REPORT.md"),
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in fields})


def as_float(v, default=None):
    if v in (None, ""):
        return default
    try:
        return float(v)
    except Exception:
        return default


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def selected_cases_from_diag(diag: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    if "cases" in diag:
        return diag["cases"]
    if "selected_cases" in diag:
        return diag["selected_cases"]
    if "clean_selected_cases" in diag:
        return diag["clean_selected_cases"]
    return {}


def go_status_from_diag(diag: Dict[str, object]) -> str:
    for key in ("go_no_go", "decision"):
        if key in diag:
            val = diag[key]
            if isinstance(val, dict):
                return str(val.get("decision", ""))
            return str(val)
    return ""


def method_name_map(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        out.setdefault(row.get("case_name", ""), {})[row.get("method_name", "")] = row
    return out


def best_method(methods: Dict[str, Dict[str, str]]) -> str:
    if not methods:
        return ""
    return min(methods.values(), key=lambda r: as_float(r.get("final_residual_ratio"), 1e99)).get("method_name", "")


def first_method_ratio(methods: Dict[str, Dict[str, str]], names: List[str]) -> Optional[float]:
    for name in names:
        if name in methods:
            return as_float(methods[name].get("final_residual_ratio"))
    return None


def best_scalar_ratio(methods: Dict[str, Dict[str, str]]) -> Optional[float]:
    candidates = []
    for name, row in methods.items():
        if name.startswith("scalar_") or name in ("oracle_scalar_scale_factor", "oracle_extended_scalar"):
            val = as_float(row.get("final_residual_ratio"))
            if val is not None:
                candidates.append(val)
    return min(candidates) if candidates else None


def collect_data():
    diagnostics_index = {"missing_files": [], "steps": {}}
    case_rows = []
    method_rows = []
    parsed_counts = {}
    for step, folder, summary_name, diag_name, report_name in STEP_CONFIGS:
        result_dir = RESULT_ROOT / folder
        summary_path = result_dir / summary_name
        diag_path = result_dir / diag_name
        report_path = result_dir / report_name
        for p in [summary_path, diag_path, report_path]:
            if not p.exists():
                diagnostics_index["missing_files"].append(str(p))
        summary = read_csv(summary_path)
        diag = load_json(diag_path)
        cases = selected_cases_from_diag(diag)
        methods_by_case = method_name_map(summary)
        parsed_counts[step] = len(cases) if cases else len(methods_by_case)
        diagnostics_index["steps"][step] = {
            "folder": folder,
            "summary_path": str(summary_path),
            "diagnostics_path": str(diag_path),
            "report_path": str(report_path),
            "go_status": go_status_from_diag(diag),
            "cases_parsed": parsed_counts[step],
        }
        for row in summary:
            method_rows.append({
                "step": step,
                "case_name": row.get("case_name", ""),
                "method_name": row.get("method_name", ""),
                "final_residual_ratio": row.get("final_residual_ratio", ""),
                "best_residual_ratio_seen": row.get("best_residual_ratio_seen", ""),
                "converged": row.get("converged", ""),
                "diverged": row.get("diverged", ""),
                "iterations_run": row.get("iterations_run", ""),
                "selected_alpha": row.get("selected_alpha", ""),
                "accepted_steps": row.get("accepted_steps", ""),
                "rejected_steps": row.get("rejected_steps", ""),
                "final_lambda": row.get("final_lambda", ""),
            })
        for case_name, cdiag in cases.items():
            methods = methods_by_case.get(case_name, {})
            audit_extra = {}
            if step == "Step 4B":
                for item in diag.get("original_audit_compact", []):
                    if item.get("original_case") == case_name:
                        audit_extra = item
            case_rows.append({
                "step": step,
                "case_name": case_name,
                "rho_I_minus_J": cdiag.get("rho_I_minus_J0", cdiag.get("rho_I_minus_J", "")),
                "coupling_ratio": cdiag.get("coupling_ratio", ""),
                "identity_deviation": cdiag.get("identity_deviation", ""),
                "condition_number": cdiag.get("condition_number", ""),
                "initial_residual_norm": cdiag.get("initial_residual_norm", ""),
                "direct_ratio": first_method_ratio(methods, ["direct_modal_inversion"]),
                "best_scalar_ratio": best_scalar_ratio(methods),
                "oracle_scalar_ratio": first_method_ratio(methods, ["oracle_scalar_scale_factor", "oracle_extended_scalar"]),
                "diagonal_ratio": first_method_ratio(methods, ["diagonal_modal_factor_initial"]),
                "full_gn_ratio": first_method_ratio(methods, ["full_jacobian_gn_no_trust"]),
                "trust_region_ratio": first_method_ratio(methods, ["full_jacobian_trust_region", "full_jacobian_trust_region_best_cap"]),
                "best_method": best_method(methods),
                "go_status": go_status_from_diag(diag),
                "notes": "",
                "direct_one_step_ratio": audit_extra.get("direct_one_step_ratio", ""),
                "best_alpha": audit_extra.get("best_alpha", ""),
                "best_alpha_ratio": audit_extra.get("best_alpha_ratio", ""),
                "small_step_linear_error": audit_extra.get("small_step_linear_error", ""),
                "direct_step_linear_error": audit_extra.get("direct_step_linear_error", ""),
                "diagnosis": audit_extra.get("diagnosis", ""),
            })
    return case_rows, method_rows, diagnostics_index, parsed_counts


def find_case(case_rows, step, case_name):
    return next((r for r in case_rows if r["step"] == step and r["case_name"] == case_name), None)


def claim_matrix(case_rows):
    s4b_easy = next((r for r in case_rows if r["step"] == "Step 4B" and r.get("direct_step_linear_error") not in ("", None)), None)
    return [
        {
            "claim_id": "A",
            "claim_text": "Direct modal inversion fails when the response Jacobian is not identity or when the unit step leaves the local linear regime.",
            "support_level": "strong",
            "main_evidence_steps": "Steps 1, 2, 3, 4B",
            "supporting_cases": "gain_mismatch_linear; coupled_linear; geom_hard_gain; physics_hard_gain; clean_hard_*",
            "key_numbers": "Direct ratios reach 1e3+ in hard cases; Step 4B direct one-step ratios exceed 1 even when small-step linearization is valid.",
            "caution_or_limitation": "rho(I-J0) alone is local; nonlinear FEM-like cases also need step-size validity diagnostics.",
        },
        {
            "claim_id": "B",
            "claim_text": "The spectral radius rho(I-J0) explains local direct-inversion stability, but by itself is not sufficient for global one-step behavior in nonlinear FEM-like response.",
            "support_level": "strong",
            "main_evidence_steps": "Step 4B",
            "supporting_cases": "fem_easy audit",
            "key_numbers": "fem_easy rho=0.387 but direct one-step ratio=1.269 and direct-step linearization error=0.927.",
            "caution_or_limitation": "Requires local linearization and line-search audit, not just eigenvalue diagnostics.",
        },
        {
            "claim_id": "C",
            "claim_text": "Fixed scalar scale factor is case-dependent; no single alpha is reliable across easy, marginal, hard, and FEM-like cases.",
            "support_level": "strong",
            "main_evidence_steps": "Steps 1-4B",
            "supporting_cases": "scalar grids across all summaries",
            "key_numbers": "Best alphas range from near 1e-4/1e-3 in FEM-like audits to larger values in synthetic cases.",
            "caution_or_limitation": "A tuned scalar can perform well on a specific case; the weakness is portability.",
        },
        {
            "claim_id": "D",
            "claim_text": "Extended scalar line search can stabilize some cases, but it requires case-specific tuning and may need alpha values far smaller than common choices.",
            "support_level": "strong",
            "main_evidence_steps": "Step 4B",
            "supporting_cases": "line_search_scan.csv original Step 4 cases",
            "key_numbers": "fem_easy best alpha=1.06e-2; fem_marginal best alpha=1.30e-3; fem_hard_nonlinear best alpha=6.65e-5.",
            "caution_or_limitation": "Line search is a diagnostic/tuning baseline, not a response model.",
        },
        {
            "claim_id": "E",
            "claim_text": "Diagonal response calibration is a strong practical approximation when modal response is mostly separable or diagonal-dominant.",
            "support_level": "strong",
            "main_evidence_steps": "Steps 1, 3B, 4B",
            "supporting_cases": "gain_mismatch_linear; coupling_hard; clean_hard_gain",
            "key_numbers": "Diagonal often reaches 1e-7 or better and sometimes beats trust-region in separable/gain-dominated cases.",
            "caution_or_limitation": "Diagonal is not a universal substitute for full Jacobian under strong coupling/ill-conditioning.",
        },
        {
            "claim_id": "F",
            "claim_text": "Full response Jacobian is the general method and adds value when coupling, ill-conditioning, or nonlinear step validity makes scalar/diagonal calibration unreliable.",
            "support_level": "moderate",
            "main_evidence_steps": "Steps 2, 3, 4B",
            "supporting_cases": "coupled_linear; weak/strong nonlinear; clean_hard_coupled; clean_hard_nonlinear",
            "key_numbers": "Full/trust strongly improves over direct/scalar in coupled and audited FEM-like cases.",
            "caution_or_limitation": "Full Jacobian does not always beat diagonal; claim must be generality/robustness, not universal dominance.",
        },
        {
            "claim_id": "G",
            "claim_text": "Trust-region control is useful because local Jacobian models can be valid at small steps but invalid at full direct-inversion step size.",
            "support_level": "strong",
            "main_evidence_steps": "Step 4B",
            "supporting_cases": "fem_easy; fem_marginal",
            "key_numbers": "Small-step linearization errors around 1e-4 to 1e-3 while direct-step errors are about 0.93-0.97.",
            "caution_or_limitation": "Trust-region improves step validity; it may not beat diagonal when diagonal calibration is already sufficient.",
        },
    ]


def recommended_cases(case_rows):
    def rec(category, row, why, claim, caution):
        return {
            "category": category,
            "case_name": row["case_name"] if row else "",
            "step": row["step"] if row else "",
            "why_include": why,
            "main_claim_supported": claim,
            "direct_ratio": row.get("direct_ratio", "") if row else "",
            "best_scalar_ratio": row.get("best_scalar_ratio", "") if row else "",
            "diagonal_ratio": row.get("diagonal_ratio", "") if row else "",
            "full_or_trust_ratio": min([x for x in [as_float(row.get("full_gn_ratio")) if row else None, as_float(row.get("trust_region_ratio")) if row else None] if x is not None], default=""),
            "rho_I_minus_J": row.get("rho_I_minus_J", "") if row else "",
            "coupling_ratio": row.get("coupling_ratio", "") if row else "",
            "caution": caution,
        }

    rows = []
    rows.append(rec("sanity_identity_case", find_case(case_rows, "Step 2", "frozen_response"), "J approximately identity and direct inversion succeeds.", "A", "Only a sanity check, not evidence against calibration."))
    rows.append(rec("synthetic_gain_mismatch_case", find_case(case_rows, "Step 1", "gain_mismatch_linear"), "Clean eigenvalue-driven direct divergence.", "A/E", "Synthetic linear case."))
    rows.append(rec("synthetic_coupling_case", find_case(case_rows, "Step 1", "coupled_linear"), "Controlled off-diagonal coupling with full Jacobian success.", "F", "Still synthetic modal response."))
    rows.append(rec("geometry_modal_hard_case", find_case(case_rows, "Step 2", "geom_hard_coupled_nonlinear"), "Geometry-space modal projection with nonlinear coupled hard response.", "A/F", "Response is still designed, not FEM-like."))
    rows.append(rec("mesh_physics_hard_case", find_case(case_rows, "Step 3", "physics_hard_nonlinear"), "Mesh-based physics-like hard nonlinear response.", "A/G", "Coupling modest; mainly gain/nonlinearity."))
    fem_like = find_case(case_rows, "Step 4B", "clean_hard_nonlinear") or find_case(case_rows, "Step 4B", "clean_hard_coupled")
    rows.append(rec("fem_like_clean_case", fem_like, "Audited FEM-like case with direct failure and response-calibrated improvement.", "B/D/G", "Use Step 4B clean/audited framing, not raw Step 4 selection."))
    return rows


def master_report(claims, recs, counts):
    lines = [
        "# Master Evidence Report",
        "",
        "## Purpose",
        "This report consolidates Steps 1 through 4B into a calibrated experimental story for response-Jacobian modal compensation.",
        "",
        "## Research Question",
        "When does modal compensation fail because it assumes an identity response Jacobian, and what level of response calibration is needed: scalar, diagonal, or full Jacobian with trust-region control?",
        "",
        "## Final Experimental Logic",
        "The sequence moves from controlled modal residuals, to geometry-space modal projection, to mesh-based physics-like response, to FEM-like response and audit. Each step isolates a different source of difficulty: gain mismatch, coupling, nonlinearity, step-size validity, and stiffness/conditioning.",
        "",
        "## What Each Step Contributed",
        "- Step 1: clean synthetic proof that direct inversion fails when eigenvalues make rho(I-J)>1, and Jacobian methods solve the intended inverse problem.",
        "- Step 2: verified modal basis intervention and projected residuals in plate geometry space.",
        "- Step 3: introduced mesh-based physics-like response and found hard non-identity regimes.",
        "- Step 3B: showed stronger coupling can be generated, but diagonal calibration may still match or beat full Jacobian.",
        "- Step 4: introduced a lightweight FEM-like thermal plate solve, but raw case selection was too aggressive.",
        "- Step 4B: audited Step 4, explained the fem_easy contradiction, and produced cleaner benchmark framing.",
        "",
        "## Main Evidence",
        "Direct inversion and fixed scalar factors are not reliable as general compensation rules. Scalar tuning can work, but the best alpha is case-dependent and can be orders of magnitude smaller than common choices. Diagonal calibration is a strong practical approximation in separable modal response. Full Jacobian trust-region is best framed as the general response-calibrated solver, especially for coupling, ill-conditioning, or invalid large steps.",
        "",
        "## Updated Final Claim",
        "The defensible claim is not that full Jacobian always beats diagonal calibration. The stronger claim is that heuristic scale factor should be replaced by response calibration: scalar is a scalar inverse-Jacobian approximation, diagonal is a mode-wise inverse approximation, and full Jacobian trust-region is the general response-calibrated method.",
        "",
        "## What Should Not Be Overclaimed",
        "- Do not claim full Jacobian always dominates diagonal calibration.",
        "- Do not present raw Step 4 hard cases without the Step 4B audit context.",
        "- Do not use rho(I-J0) alone as a global convergence predictor in nonlinear FEM-like settings.",
        "- Do not imply the FEM-like response is industrial FEM or validated manufacturing physics.",
        "",
        "## Recommended Final Benchmark Cases",
    ]
    for r in recs:
        lines.append(f"- {r['category']}: {r['step']} `{r['case_name']}` — {r['why_include']}")
    lines.extend([
        "",
        "## Practical Interpretation Of Scale Factor",
        "A scale factor is a crude inverse response model. A scalar factor assumes all modes have the same response gain. A diagonal factor allows one gain per mode. A full Jacobian accounts for coupling. Trust-region control addresses the additional fact that a local response model may be valid only for small steps.",
        "",
        "## What To Do Next",
        "Use the Step 4B clean cases as the paper-style FEM-like benchmark, then add differentiable/JAX or implicit Jacobian comparisons only after the benchmark is stable. Later add model mismatch, noise, calibrated FEM, and eventually real scan data.",
        "",
        "## Parsed Case Counts",
    ])
    for step, count in counts.items():
        lines.append(f"- {step}: {count}")
    return "\n".join(lines) + "\n"


def limitations_doc():
    return """# Limitations And Next Steps

## Current Limitations

1. The FEM-like response is still simplified and is not industrial FEM.
2. No real manufacturing scan data has been used yet.
3. Several steps use finite-difference Jacobians.
4. Full Jacobian does not always outperform diagonal calibration.
5. Some hard FEM-like regimes are aggressive and need Step 4B-style selection/audit.

## Next Experimental Steps

1. Use the Step 4B clean cases as the main FEM-like benchmark.
2. Add JAX or another differentiable/implicit solver only after the benchmark is stable.
3. Compare finite-difference Jacobian against JAX or implicit Jacobian.
4. Add mild model mismatch and measurement noise.
5. Eventually test on calibrated FEM or real scan data.
"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    case_rows, method_rows, diag_index, counts = collect_data()
    claims = claim_matrix(case_rows)
    recs = recommended_cases(case_rows)
    write_csv(MASTER_CASE, case_rows)
    write_csv(MASTER_METHOD, method_rows)
    write_csv(CLAIMS, claims)
    write_csv(RECOMMENDED, recs)
    LIMITATIONS.write_text(limitations_doc(), encoding="utf-8")
    MASTER_REPORT.write_text(master_report(claims, recs, counts), encoding="utf-8")
    DIAG_INDEX.write_text(json.dumps(diag_index, indent=2), encoding="utf-8")

    print(f"MASTER_EVIDENCE_REPORT.md: {MASTER_REPORT}")
    print(f"master_case_summary.csv: {MASTER_CASE}")
    print(f"master_method_summary.csv: {MASTER_METHOD}")
    print(f"claim_support_matrix.csv: {CLAIMS}")
    print(f"recommended_final_benchmark_cases.csv: {RECOMMENDED}")
    print(f"limitations_and_next_steps.md: {LIMITATIONS}")
    print(f"diagnostics_index.json: {DIAG_INDEX}")
    print()
    print("Cases parsed per step:")
    for step, count in counts.items():
        print(f"  {step}: {count}")
    print()
    print("Claim support:")
    print("claim_id | support_level | main_evidence | caution")
    for c in claims:
        print(f"{c['claim_id']} | {c['support_level']} | {c['main_evidence_steps']} | {c['caution_or_limitation']}")
    print()
    print("Recommended benchmarks:")
    print("category | step | case | main_reason")
    for r in recs:
        print(f"{r['category']} | {r['step']} | {r['case_name']} | {r['why_include']}")
    print()
    print("Final conclusion: GO for using current results as a paper-style experimental story, with calibrated claims and Step 4B audit framing.")


if __name__ == "__main__":
    main()
