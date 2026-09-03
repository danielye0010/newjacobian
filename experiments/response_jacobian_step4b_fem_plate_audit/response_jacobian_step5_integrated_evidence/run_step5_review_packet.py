"""Create a compact review packet from existing response-Jacobian outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
OUT = RESULTS / "response_jacobian_review_packet"

CLAIMS_IN = RESULTS / "response_jacobian_step5_integrated_evidence" / "claim_support_matrix.csv"
RECS_IN = RESULTS / "response_jacobian_step5_integrated_evidence" / "recommended_final_benchmark_cases.csv"
STEP4B_DIAG = RESULTS / "response_jacobian_step4b_fem_plate_audit" / "diagnostics.json"

PACKET = OUT / "CHATGPT_REVIEW_PACKET.md"
CLAIMS_OUT = OUT / "key_claims_table.csv"
RECS_OUT = OUT / "recommended_cases_table.csv"
METHODS_OUT = OUT / "key_method_comparison_table.csv"
AUDIT_OUT = OUT / "step4b_audit_key_table.csv"
CAUTIONS_OUT = OUT / "anomalies_and_cautions.md"

SUMMARY_FILES = {
    "Step 1": RESULTS / "response_jacobian_step1_synthetic" / "summary.csv",
    "Step 2": RESULTS / "response_jacobian_step2_plate_modal" / "summary.csv",
    "Step 3": RESULTS / "response_jacobian_step3_physics_plate" / "summary.csv",
    "Step 3B": RESULTS / "response_jacobian_step3b_coupling_stress" / "summary.csv",
    "Step 4": RESULTS / "response_jacobian_step4_fem_plate" / "summary.csv",
    "Step 4B": RESULTS / "response_jacobian_step4b_fem_plate_audit" / "audited_summary.csv",
}


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


def fnum(v, default=None):
    if v in (None, ""):
        return default
    try:
        return float(v)
    except Exception:
        return default


def fmt(v) -> str:
    x = fnum(v)
    return "" if x is None else f"{x:.3e}"


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def normalize_claims() -> List[Dict[str, str]]:
    wanted = ["claim_id", "claim_text", "support_level", "main_evidence_steps", "supporting_cases", "key_numbers", "caution_or_limitation"]
    rows = read_csv(CLAIMS_IN)
    return [{k: r.get(k, "") for k in wanted} for r in rows]


def normalize_recs() -> List[Dict[str, str]]:
    rows = read_csv(RECS_IN)
    out = []
    for r in rows:
        out.append({
            "category": r.get("category", ""),
            "step": r.get("step", ""),
            "case_name": r.get("case_name", ""),
            "why_include": r.get("why_include", ""),
            "main_claim_supported": r.get("main_claim_supported", ""),
            "rho_I_minus_J": r.get("rho_I_minus_J", ""),
            "coupling_ratio": r.get("coupling_ratio", ""),
            "direct_ratio": r.get("direct_ratio", ""),
            "best_scalar_ratio": r.get("best_scalar_ratio", ""),
            "oracle_scalar_ratio": r.get("oracle_scalar_ratio", ""),
            "diagonal_ratio": r.get("diagonal_ratio", ""),
            "full_or_trust_ratio": r.get("full_or_trust_ratio", ""),
            "caution": r.get("caution", ""),
        })
    return out


def method_rows_for_recs(recs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    all_methods = {step: read_csv(path) for step, path in SUMMARY_FILES.items()}
    selected_names = {
        "direct_modal_inversion",
        "oracle_scalar_scale_factor",
        "oracle_extended_scalar",
        "diagonal_modal_factor_initial",
        "full_jacobian_gn_no_trust",
        "full_jacobian_trust_region",
        "full_jacobian_trust_region_best_cap",
    }
    out = []
    for rec in recs:
        step = rec["step"]
        case = rec["case_name"]
        rows = [r for r in all_methods.get(step, []) if r.get("case_name") == case]
        for r in rows:
            name = r.get("method_name", "")
            if name in selected_names:
                out.append({
                    "step": step,
                    "case_name": case,
                    "method_name": name,
                    "final_residual_ratio": r.get("final_residual_ratio", ""),
                    "best_residual_ratio_seen": r.get("best_residual_ratio_seen", ""),
                    "converged": r.get("converged", ""),
                    "diverged": r.get("diverged", ""),
                    "iterations_run": r.get("iterations_run", ""),
                    "selected_alpha": r.get("selected_alpha", ""),
                    "accepted_steps": r.get("accepted_steps", ""),
                    "rejected_steps": r.get("rejected_steps", ""),
                    "final_lambda": r.get("final_lambda", ""),
                })
    return out


def best_scalar(methods: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    candidates = [r for r in methods if r.get("method_name") in ("oracle_scalar_scale_factor", "oracle_extended_scalar")]
    if not candidates:
        return None
    return min(candidates, key=lambda r: fnum(r.get("final_residual_ratio"), 1e99))


def method_pivot(recs: List[Dict[str, str]], method_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    pivots = []
    for rec in recs:
        case_methods = [r for r in method_rows if r["step"] == rec["step"] and r["case_name"] == rec["case_name"]]
        by_name = {r["method_name"]: r for r in case_methods}
        scalar = best_scalar(case_methods)
        trust = by_name.get("full_jacobian_trust_region_best_cap") or by_name.get("full_jacobian_trust_region")
        direct = fnum(by_name.get("direct_modal_inversion", {}).get("final_residual_ratio"))
        diag = fnum(by_name.get("diagonal_modal_factor_initial", {}).get("final_residual_ratio"))
        full = fnum(by_name.get("full_jacobian_gn_no_trust", {}).get("final_residual_ratio"))
        tr = fnum(trust.get("final_residual_ratio") if trust else None)
        if direct is not None and direct > 1:
            interp = "direct fails; response calibration needed"
        elif rec["category"] == "sanity_identity_case":
            interp = "sanity case where direct is valid"
        elif diag is not None and tr is not None and diag <= tr:
            interp = "diagonal calibration is competitive"
        else:
            interp = "full/trust adds robustness"
        pivots.append({
            "case": rec["case_name"],
            "category": rec["category"],
            "rho_I_minus_J": rec.get("rho_I_minus_J", ""),
            "coupling": rec.get("coupling_ratio", ""),
            "direct": direct,
            "best_scalar": fnum(scalar.get("final_residual_ratio")) if scalar else None,
            "diagonal": diag,
            "full_GN": full,
            "trust_region": tr,
            "interpretation": interp,
        })
    return pivots


def step4b_audit_table() -> List[Dict[str, object]]:
    diag = load_json(STEP4B_DIAG)
    rows = []
    for r in diag.get("original_audit_compact", []):
        rows.append({
            "case_name": r.get("original_case", ""),
            "rho_I_minus_J": r.get("rho", ""),
            "coupling_ratio": "",
            "condition_number": "",
            "direct_one_step_ratio": r.get("direct_one_step_ratio", ""),
            "best_alpha": r.get("best_alpha", ""),
            "best_alpha_ratio": r.get("best_alpha_ratio", ""),
            "small_step_linear_error": r.get("small_step_linear_error", ""),
            "direct_step_linear_error": r.get("direct_step_linear_error", ""),
            "diagnosis": r.get("diagnosis", ""),
            "clean_case_status": "original_step4_case",
        })
    for name, c in diag.get("clean_selected_cases", {}).items():
        rows.append({
            "case_name": name,
            "rho_I_minus_J": c.get("rho_I_minus_J0", ""),
            "coupling_ratio": c.get("coupling_ratio", ""),
            "condition_number": c.get("condition_number", ""),
            "direct_one_step_ratio": "",
            "best_alpha": "",
            "best_alpha_ratio": "",
            "small_step_linear_error": "",
            "direct_step_linear_error": "",
            "diagnosis": "clean selected Step 4B benchmark case",
            "clean_case_status": "clean_selected_case",
        })
    return rows


def markdown_table(rows: List[Dict[str, object]], cols: List[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.3e}")
            else:
                vals.append(str(v).replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def cautions_doc() -> str:
    return """# Anomalies And Cautions

1. Full Jacobian/trust-region does not always beat diagonal calibration.
2. Step 3B found strong coupling, but diagonal performed as well as or better than full Jacobian in selected cases.
3. Step 4 had aggressive FEM-like hard cases; Step 4B audit is needed to frame them correctly.
4. rho(I-J0) explains local stability but not full-step nonlinear behavior.
5. Scalar scale factor can work when tuned, but the best alpha may be case-dependent and very small.
6. Current FEM is FEM-like/simplified, not industrial calibrated FEM.
7. Finite-difference Jacobian was used in several steps.
8. No real manufacturing validation has been done yet.

## Recommended wording to avoid overclaiming

- Do not claim full Jacobian always outperforms diagonal response calibration.
- Do not claim industrial FEM validation has been completed.
- Do not claim direct inversion always fails.
- Do claim direct inversion fails when its identity-response and unit-step assumptions are violated.
- Do claim response Jacobian calibration provides a principled explanation and generalization of scale factor.
"""


def packet_md(claims, recs, pivots, audit_rows) -> str:
    fem_easy = next((r for r in audit_rows if r["case_name"] == "fem_easy"), {})
    return "\n".join([
        "# ChatGPT Review Packet",
        "",
        "## 1. Purpose Of This Packet",
        "This packet gives the compact evidence needed to judge the final paper story for response-Jacobian modal compensation.",
        "",
        "## 2. Current Final Conclusion From Step 5",
        "GO for a paper-style experimental story, with calibrated claims. The central story should be response calibration replacing heuristic scale factor, not full Jacobian always beating diagonal calibration.",
        "",
        "## 3. Claim Support Table",
        markdown_table(claims, ["claim_id", "support_level", "main_evidence_steps", "supporting_cases", "key_numbers", "caution_or_limitation"]),
        "",
        "## 4. Recommended Benchmark Cases",
        markdown_table(recs, ["category", "step", "case_name", "why_include", "main_claim_supported", "rho_I_minus_J", "coupling_ratio", "direct_ratio", "best_scalar_ratio", "diagonal_ratio", "full_or_trust_ratio", "caution"]),
        "",
        "## 5. Key Method Comparison Pivot Table",
        markdown_table(pivots, ["case", "rho_I_minus_J", "coupling", "direct", "best_scalar", "diagonal", "full_GN", "trust_region", "interpretation"]),
        "",
        "## 6. Step 4B Audit Summary",
        f"fem_easy contradiction: rho(I-J0)={fem_easy.get('rho_I_minus_J', '')}, direct one-step ratio={fem_easy.get('direct_one_step_ratio', '')}, best alpha={fem_easy.get('best_alpha', '')}, small-step linear error={fem_easy.get('small_step_linear_error', '')}, direct-step linear error={fem_easy.get('direct_step_linear_error', '')}.",
        "",
        "Explanation: small-step linearization is valid, but the full direct step leaves the local region. This means traditional scale factor is not a universal constant; for stiff FEM-like response, safe alpha can be much smaller than 0.05.",
        "",
        markdown_table(audit_rows, ["case_name", "rho_I_minus_J", "coupling_ratio", "condition_number", "direct_one_step_ratio", "best_alpha", "best_alpha_ratio", "small_step_linear_error", "direct_step_linear_error", "diagnosis", "clean_case_status"]),
        "",
        "## 7. Main Anomalies And Cautions",
        "- Full Jacobian/trust-region does not always beat diagonal calibration.",
        "- Step 3B found strong coupling, but diagonal was often as good or better.",
        "- Raw Step 4 hard cases are aggressive and need Step 4B audit framing.",
        "- rho(I-J0) is local; it does not guarantee safe full-step behavior.",
        "- FEM-like response is simplified, finite-difference Jacobians are used, and no real manufacturing validation exists yet.",
        "",
        "## 8. Suggested Final Research Claim",
        "Response calibration provides a principled generalization of scale-factor compensation. Direct inversion assumes identity response. Scalar scale factor is a scalar inverse-Jacobian approximation. Diagonal modal factor is a mode-wise approximation. Full Jacobian trust-region is the general response-calibrated method, especially valuable when coupling, ill-conditioning, or nonlinear step validity matter.",
        "",
        "## 9. Open Questions For ChatGPT To Evaluate",
        "Q1. Are the claims calibrated correctly?",
        "Q2. Which cases should go in the main paper?",
        "Q3. Which cases should go to supplement?",
        "Q4. Is the Step 4B audit enough to justify trust-region?",
        "Q5. Should the final contribution emphasize full Jacobian, diagonal calibration, or the unified response-calibration framework?",
        "Q6. What additional experiment, if any, is truly necessary before writing?",
        "",
    ]) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    claims = normalize_claims()
    recs = normalize_recs()
    method_rows = method_rows_for_recs(recs)
    pivots = method_pivot(recs, method_rows)
    audit_rows = step4b_audit_table()

    write_csv(CLAIMS_OUT, claims)
    write_csv(RECS_OUT, recs)
    write_csv(METHODS_OUT, method_rows)
    write_csv(AUDIT_OUT, audit_rows)
    CAUTIONS_OUT.write_text(cautions_doc(), encoding="utf-8")
    PACKET.write_text(packet_md(claims, recs, pivots, audit_rows), encoding="utf-8")

    print(f"CHATGPT_REVIEW_PACKET.md: {PACKET}")
    print(f"key_claims_table.csv: {CLAIMS_OUT}")
    print(f"recommended_cases_table.csv: {RECS_OUT}")
    print(f"key_method_comparison_table.csv: {METHODS_OUT}")
    print(f"step4b_audit_key_table.csv: {AUDIT_OUT}")
    print(f"anomalies_and_cautions.md: {CAUTIONS_OUT}")
    print()
    print("A. claim table:")
    print("claim_id | support_level | main_evidence_steps | caution_or_limitation")
    for r in claims:
        print(f"{r['claim_id']} | {r['support_level']} | {r['main_evidence_steps']} | {r['caution_or_limitation']}")
    print()
    print("B. recommended cases:")
    print("category | step | case_name | why_include | caution")
    for r in recs:
        print(f"{r['category']} | {r['step']} | {r['case_name']} | {r['why_include']} | {r['caution']}")
    print()
    print("C. method comparison:")
    print("case | direct | best_scalar | diagonal | full_GN | trust_region | interpretation")
    for r in pivots:
        print(f"{r['case']} | {fmt(r['direct'])} | {fmt(r['best_scalar'])} | {fmt(r['diagonal'])} | {fmt(r['full_GN'])} | {fmt(r['trust_region'])} | {r['interpretation']}")
    print()
    print("D. Step 4B audit:")
    print("case | rho(I-J0) | direct_one_step_ratio | best_alpha | best_alpha_ratio | small_step_error | direct_step_error | diagnosis")
    for r in audit_rows:
        print(f"{r['case_name']} | {fmt(r['rho_I_minus_J'])} | {fmt(r['direct_one_step_ratio'])} | {fmt(r['best_alpha'])} | {fmt(r['best_alpha_ratio'])} | {fmt(r['small_step_linear_error'])} | {fmt(r['direct_step_linear_error'])} | {r['diagnosis']}")


if __name__ == "__main__":
    main()
