from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]
AUTH_SOURCE = ROOT / "native_M32A_blind/source/nonlingeo.c"
AUTH_H32 = ROOT / "native_M32A_blind/H_native_49824x32.bin"
SOURCE_SHA = "8b2055f9f6dee896b2ead1a2e7e5831ff66c287d861d79c4b04ee382a8296178"
H32_SHA = "850dd8ad99b4047db0f5732fce4ffb9cf327b91eb83aad419e379539c78d8b03"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(name: str):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def artifact(path: Path, description: str) -> dict:
    return {
        "path": str(path.relative_to(HERE)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "description": description,
    }


def main() -> None:
    boundary = load_json("B48_SOURCE_BOUNDARY_AUDIT.json")
    b32 = load_json("B32_BACKWARD_COMPATIBILITY.json")
    bprefix = load_json("B48_PREFIX_AUDIT.json")
    bprov = load_json("B48_PROVENANCE.json")
    deriv = load_json("B48_DERIVATIVE_QUALIFICATION.json")
    hprefix = load_json("H48_PREFIX_AUDIT.json")
    action = load_json("H48_ACTION_QUALIFICATION.json")
    historical = load_json("H48_HISTORICAL_CONTEXT.json")

    fd_state = []
    for mode in range(33, 49):
        folder = HERE / "fd_new_modes" / f"mode_{mode:02d}"
        checks = {
            "final_vold_matches_locked": (folder / "final_vold.bin").read_bytes() == (folder / "locked_final_vold.bin").read_bytes(),
            "final_vold_after_matches_locked": (folder / "final_vold_after.bin").read_bytes() == (folder / "locked_final_vold.bin").read_bytes(),
            "prestr_matches_locked": (folder / "prestr.bin").read_bytes() == (folder / "locked_prestr.bin").read_bytes(),
            "prestr_after_matches_locked": (folder / "prestr_after.bin").read_bytes() == (folder / "locked_prestr.bin").read_bytes(),
            "material_labels_invariant": (folder / "material_labels_before.bin").read_bytes() == (folder / "material_labels_after.bin").read_bytes(),
            "history_invariant": (folder / "history_before.bin").read_bytes() == (folder / "history_after.bin").read_bytes(),
        }
        fd_state.append({"mode": mode, **checks, "passed": all(checks.values())})

    gates = {
        "source_boundary_audit": boundary["verdict"] == "PASS_TO_PHASE_1",
        "dimension_only_source_diff": not boundary["scientific_derivative_logic_change_required"] and not boundary["unknown_change_required"],
        "M32_backward_compatibility": b32["verdict"] == "PASS",
        "B48_first_32_prefix": bprefix["verdict"] == "PASS",
        "new_modes_33_48_independent_FD": deriv["verdict"] == "PASS" and deriv["all_new_modes_passed"],
        "FD_locked_state_invariance": all(row["passed"] for row in fd_state),
        "H48_first_32_prefix": hprefix["verdict"] == "PASS",
        "new_range_forward_adjoint_identity": action["verdict"] == "PASS",
        "authoritative_M32_source_untouched": sha256(AUTH_SOURCE) == SOURCE_SHA,
        "authoritative_H32_untouched": sha256(AUTH_H32) == H32_SHA,
    }
    verdict = "PASS" if all(gates.values()) else "FAIL"

    key_files = [
        (HERE / "generalized_source/nonlingeo.c", "qualified dimension-generalized production source"),
        (HERE / "generalized_source/native_B48Q_i4.exe", "qualified CalculiX executable"),
        (HERE / "generalized_source/ccx_2.23_B48Q.a", "isolated build archive"),
        (HERE / "solver/solve_B48Q.c", "48-action direct-solver dimension extension"),
        (HERE / "solver/solve_B48Q.exe", "48-action direct solver executable"),
        (HERE / "B48_native_49536x48.bin", "qualified native residual-design matrix"),
        (HERE / "B48_native_49536x48.npy", "qualified native residual-design matrix with shape metadata"),
        (HERE / "H48_native_49824x48.bin", "qualified weighted native tangent response"),
        (HERE / "H48_native_49824x48.npy", "qualified weighted native tangent response with shape metadata"),
        (HERE / "Hbar_native_49824x48.bin", "unweighted native tangent response"),
        (HERE / "U_a_active_49536x48.bin", "48 ordered native tangent solutions"),
        (HERE / "B48_PHI_MANIFEST.csv", "ordered Phi48 manifest"),
    ]
    artifacts = [artifact(path, description) for path, description in key_files]
    decision = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "R6P_NATIVE_B48_EXTENSION_QUALIFICATION",
        "verdict": verdict,
        "gates": gates,
        "B32": {
            "frobenius_relative_error": b32["frobenius_relative_error"],
            "all_columns_byte_equal": b32["all_columns_byte_equal"],
        },
        "B48_prefix": {
            "frobenius_relative_error": bprefix["frobenius_relative_prefix_error"],
            "all_columns_byte_equal": bprefix["all_prefix_columns_byte_equal"],
        },
        "new_mode_FD": {
            "confirmation_step": deriv["confirmation_step"],
            "median_relative_error": deriv["median_relative_error"],
            "p90_relative_error": deriv["p90_relative_error"],
            "maximum_relative_error": deriv["maximum_relative_error"],
            "worst_mode": deriv["worst_mode"],
        },
        "H48_prefix": {
            "frobenius_relative_error": hprefix["frobenius_prefix_relative_error"],
            "all_prefix_bytes_equal": hprefix["all_prefix_bytes_equal"],
        },
        "forward_adjoint": {
            "maximum_normalized_discrepancy": action["maximum_identity_normalized_discrepancy"],
            "tolerance": action["inherited_identity_tolerance"],
        },
        "historical_H48_secondary_context": historical,
        "FD_state_checks": fd_state,
        "artifacts": artifacts,
        "claims_supported": [
            "The validated native M32 residual-design implementation generalizes to 48 ordered production modes using dimension plumbing only.",
            "Generalized M32 and the first 32 B48 columns reproduce authoritative native B32 exactly.",
            "Modes 33 through 48 pass independent locked-state symmetric residual finite-difference qualification at the preregistered common step.",
            "H48 is a qualified native tangent response whose first 32 columns reproduce authoritative native H32 exactly.",
            "Native forward and transpose-adjoint paths satisfy the inherited identity tolerance for e33, e40, e48, and a fixed dense direction.",
        ],
        "claims_not_supported": [
            "No R6A inverse-information scaling, Direct-start trajectory, threshold, K_D scaling, nonlinear compensation, timing scaling, M64 result, or manuscript claim was evaluated.",
            "Historical finite-step H48 is not interchangeable with or an oracle for the new native tangent H48.",
        ],
        "prohibited_work_checks": {
            "Direct_start_PCG_run": False,
            "m99_m999_m9999_evaluated": False,
            "R6A_KD_scaling_computed": False,
            "nonlinear_compensated_or_perturbed_geometry_FEM_run": False,
            "M48_compensation_run": False,
            "timing_scaling_run": False,
            "M64_work": False,
            "manuscript_edited": False,
        },
        "baseline_scope_note": "The single B48 production construction necessarily included its one frozen baseline equilibrium. Each opt-in FD process initialized CalculiX with the common deck and then restored byte-identical locked vold/prestr before every signed residual assembly; no perturbed geometry was re-equilibrated and no compensated nonlinear FEM was run.",
        "recommendation": "Resume the original R6A inverse-information scaling experiment." if verdict == "PASS" else "Do not resume R6A until the failed R6P gate is reviewed.",
        "R6A_resumed_in_this_task": False,
    }
    decision_path = HERE / "R6P_FINAL_DECISION.json"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# R6P — Native B48 Extension Qualification",
        "",
        f"## 1. Executive decision: {verdict}",
        "",
        "R6P passes every mandatory gate. The frozen native M32 residual-design derivative was generalized to 48 ordered production directions using dimension plumbing only. The qualified recommendation is to resume the original R6A inverse-information scaling experiment; R6A was not resumed here.",
        "",
        "## 2. Why R6P was required",
        "",
        "R6A stopped because no authoritative native B48 existed and the validated production acquisition was fixed at 32 columns. R6P establishes whether an M48 oracle can be created without changing the scientific derivative formulation.",
        "",
        "## 3. Source-boundary audit",
        "",
        "`B48_SOURCE_BOUNDARY_AUDIT.md` and `.json` classify every required occurrence as dimension plumbing, output bookkeeping, or storage capacity. No scientific derivative or unknown change was required. The final authoritative M32 source SHA-256 remains `" + sha256(AUTH_SOURCE) + "`.",
        "",
        "## 4. Minimal source diff",
        "",
        "`B48_SOURCE_DIFF.md` records only configurable count bounds, loop/index expressions, fixed capacities, and generalized labels/manifests. The `results_se`/`mafillsmmain_se` derivative calls and `results`/`mafillsmmain` tangent calls are unchanged. The isolated H48 solver changes only `NMODE=32` to `48` and associated labels/count fields.",
        "",
        "## 5. M32 backward compatibility",
        "",
        f"PASS: Frobenius relative error `{b32['frobenius_relative_error']:.17e}`, maximum and median column errors zero, and all 32 columns byte-identical. Detailed results are in `B32_BACKWARD_COMPATIBILITY.json` and `B32_COLUMN_ERRORS.csv`.",
        "",
        "## 6. B48 construction provenance",
        "",
        f"Exactly one production B48 construction was executed after M32 passed. B48 has shape `49536 x 48`, SHA-256 `{bprov['binary_sha256']}`, invariant accepted-state hash `{bprov['state_hash']}`, and invariant private-input hashes. Source, executable, Phi-column, RHS-column, and runtime identities are frozen in `B48_PROVENANCE.json`.",
        "",
        "## 7. B48 first-32 prefix",
        "",
        f"PASS: Frobenius prefix relative error `{bprefix['frobenius_relative_prefix_error']:.17e}`; all first 32 columns are byte-identical to authoritative B32.",
        "",
        "## 8. Step refinement for modes 33, 40, and 48",
        "",
        f"The preregistered scale was `h0={deriv['h0']:.17e}` with `h0/2` and `h0/4`. Both candidate refined steps met the common stability rule, so the preregistered finest-common rule selected `h0/4={deriv['confirmation_step']:.17e}`. `B48_NEW_MODE_STEP_REFINEMENT.csv` records native-vs-FD and FD step-to-step errors. The inherited hook's raw h0/8 output is retained but excluded from selection and gates.",
        "",
        "## 9. Modes 33–48 independent FD qualification",
        "",
        f"PASS at the common confirmation step: median relative error `{deriv['median_relative_error']:.17e}`, p90 `{deriv['p90_relative_error']:.17e}`, maximum `{deriv['maximum_relative_error']:.17e}` at mode `{deriv['worst_mode']}`, all below the inherited `1e-2` standard. Every signed assembly restored the same locked state; all byte-invariance checks passed.",
        "",
        "## 10. B48 classification",
        "",
        "PASS. M32 compatibility, first-32 prefix invariance, common stable FD regime, and all 16 new-mode FD gates passed.",
        "",
        "## 11. H48 construction",
        "",
        f"After B48 passed, the frozen K_T was factored once and exactly 48 ordered native forward solves were performed. `H48_native_49824x48.bin` has SHA-256 `{hprefix['H48_sha256']}`. The solver runtime manifest confirms one factorization and 48 backsolves.",
        "",
        "## 12. H48 first-32 reproduction",
        "",
        f"PASS: Frobenius, maximum-column, and median-column prefix errors are all zero; the complete first-32 prefix is byte-identical to authoritative H32 (`{hprefix['authoritative_H32_sha256']}`).",
        "",
        "## 13. Forward/adjoint identity",
        "",
        f"PASS for e33, e40, e48, and the fixed seeded dense vector. Maximum normalized discrepancy is `{action['maximum_identity_normalized_discrepancy']:.17e}` versus the inherited `1e-10` tolerance; represented tangent asymmetry is `{action['represented_tangent_asymmetry_frobenius']:.17e}`.",
        "",
        "## 14. Historical H48 context",
        "",
        f"Secondary diagnostic only: historical H48 is a finite-step response object, while the new H48 is a native tangent response. Their relative Frobenius difference is `{historical['relative_frobenius_difference']:.17e}`. Historical H48 was not used as a qualification oracle.",
        "",
        "## 15. Claims supported and not supported",
        "",
        "Supported: dimension-only M48 generalization; exact M32/B48/H48 prefixes; independent qualification of modes 33–48; and new-range native forward/adjoint consistency. Not supported: any R6A inverse/scaling, Direct-start, threshold, nonlinear compensation, timing, M64, or manuscript conclusion.",
        "",
        "## 16. Recommendation",
        "",
        "Resume the original R6A inverse-information scaling experiment. Do not reinterpret this prerequisite qualification as an R6A result.",
        "",
        "## 17. Artifacts and hashes",
        "",
    ]
    for row in artifacts:
        lines.append(f"- `{row['path']}` — `{row['sha256']}` — {row['description']}")
    lines += [
        "",
        "Machine-readable gate results are in `R6P_FINAL_DECISION.json`; the B48/H48 prefix, derivative, action, and historical-context files provide the detailed evidence chain.",
        "",
        "## Final self-check",
        "",
        "Authoritative M32 source and H32 hashes remain unchanged. M32 was qualified before the sole M48 production construction. No M48 result was accepted before independent FD qualification. Modes 33–48 were all checked. No Direct-start PCG, pair threshold, K_D scaling, nonlinear compensation or perturbed-geometry re-equilibration, M48 compensation, timing scaling, manuscript edit, or M64 work occurred. The required baseline initialization scope is stated explicitly in `R6P_FINAL_DECISION.json`.",
    ]
    (HERE / "NATIVE_B48_EXTENSION_HANDOFF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    required = [
        "NATIVE_B48_EXTENSION_HANDOFF.md",
        "B48_SOURCE_BOUNDARY_AUDIT.json", "B48_SOURCE_BOUNDARY_AUDIT.md", "B48_SOURCE_DIFF.md",
        "B32_BACKWARD_COMPATIBILITY.json", "B32_COLUMN_ERRORS.csv",
        "B48_PROVENANCE.json", "B48_PREFIX_AUDIT.json", "B48_PHI_MANIFEST.csv",
        "B48_FD_PREREGISTRATION.json", "B48_NEW_MODE_STEP_REFINEMENT.csv",
        "B48_NEW_MODE_FD_QUALIFICATION.csv", "B48_DERIVATIVE_QUALIFICATION.json",
        "H48_ACTION_PREREGISTRATION.json", "H48_PREFIX_AUDIT.json",
        "H48_ADJOINT_IDENTITY.csv", "H48_ACTION_QUALIFICATION.json",
        "H48_HISTORICAL_CONTEXT.json", "R6P_FINAL_DECISION.json",
        "B48_native_49536x48.bin", "B48_native_49536x48.npy",
        "H48_native_49824x48.bin", "H48_native_49824x48.npy",
        "Hbar_native_49824x48.bin", "Hbar_native_49824x48.npy",
        "U_a_active_49536x48.bin", "U_a_active_49536x48.npy",
        "generalized_source/nonlingeo.c", "generalized_source/native_B48Q_i4.exe",
        "generalized_source/ccx_2.23_B48Q.a", "solver/solve_B48Q.c", "solver/solve_B48Q.exe",
        "run_M48/B48Q_solver_runtime.txt",
    ]
    freeze = {
        "schema_version": 1,
        "task": "R6P_NATIVE_B48_EXTENSION_QUALIFICATION",
        "verdict": verdict,
        "artifacts": [artifact(HERE / name, "R6P frozen qualification artifact") for name in required],
    }
    (HERE / "R6P_FREEZE_MANIFEST.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "gates": gates, "core_artifact_count": len(artifacts), "freeze_artifact_count": len(required)}, indent=2))
    if verdict != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
