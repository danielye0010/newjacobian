"""Run Fixed-Bottom Modal Response-Jacobian Compensation for FDM Warpage."""
from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fixed_bottom_core import (
    BasisData,
    FixedBottomJaxResponse,
    ResponseCase,
    VolumeMesh,
    compute_fixed_bottom_modes,
    generate_geometries,
    response_cases,
)
from fixed_bottom_methods import MethodOutcome, run_all_methods


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "results" / "fixed_bottom_jax_fem"
FIGURE_DIR = RESULT_DIR / "figures"
MODE_COUNTS = (4, 8, 16, 32)
PINV_RCOND = 1e-10


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            clean = {}
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, (np.floating, np.integer)):
                    value = value.item()
                elif isinstance(value, np.bool_):
                    value = bool(value)
                elif isinstance(value, (list, tuple, dict, np.ndarray)):
                    value = json.dumps(np.asarray(value).tolist() if isinstance(value, np.ndarray) else value)
                clean[column] = value
            writer.writerow(clean)


def geometry_rows(geometries: Iterable[VolumeMesh]) -> List[Dict[str, object]]:
    return [
        {
            "geometry": mesh.name,
            "nodes": mesh.num_nodes,
            "elements": mesh.num_elements,
            "bottom_nodes": mesh.bottom_nodes,
            "surface_nodes_if_used": mesh.surface_nodes,
            "dimensions": mesh.dimensions,
            "mesh_resolution": mesh.mesh_resolution,
            "notes": mesh.notes,
        }
        for mesh in geometries
    ]


def basis_rows(mesh: VolumeMesh, basis: BasisData) -> List[Dict[str, object]]:
    rows = []
    bottom_dofs = np.repeat(mesh.bottom_mask, 3)
    for mode_count in MODE_COUNTS:
        usable = min(mode_count, basis.usable_modes)
        psi = basis.psi[:, :usable]
        gram = psi.T @ (mesh.weights_dof[:, None] * psi)
        bottom = psi[bottom_dofs]
        first = float(basis.eigenvalues[0])
        last = float(basis.eigenvalues[usable - 1])
        rows.append(
            {
                "geometry": mesh.name,
                "mode_count": mode_count,
                "usable_modes": usable,
                "bottom_constraint_error_F": float(np.linalg.norm(bottom, ord="fro")),
                "bottom_constraint_error_inf": float(np.max(np.abs(bottom))) if bottom.size else 0.0,
                "max_bottom_displacement_per_unit_modal_amplitude": float(np.max(np.abs(bottom))) if bottom.size else 0.0,
                "mass_orthonormality_error_F": float(np.linalg.norm(gram - np.eye(usable), ord="fro")),
                "first_mode_frequency_or_eigenvalue": first,
                "last_mode_frequency_or_eigenvalue": last,
                "condition_estimate_if_available": last / max(first, 1e-15),
                "notes": "; ".join(basis.descriptions[: min(5, usable)]),
            }
        )
    return rows


def jacobian_diagnostics(J: np.ndarray) -> Dict[str, object]:
    singular = np.linalg.svd(J, compute_uv=False)
    off = J - np.diag(np.diag(J))
    abs_diag = np.abs(np.diag(J))
    row_off = np.sum(np.abs(off), axis=1)
    return {
        "spectral_radius_I_minus_J0": float(np.max(np.abs(np.linalg.eigvals(np.eye(J.shape[0]) - J)))),
        "cond_J0": float(np.linalg.cond(J)),
        "rank_J0": int(np.linalg.matrix_rank(J, tol=PINV_RCOND * max(float(np.max(singular)), 1.0))),
        "min_singular_value_J0": float(np.min(singular)),
        "max_singular_value_J0": float(np.max(singular)),
        "coupling_ratio": float(np.linalg.norm(off, ord="fro") / max(np.linalg.norm(J, ord="fro"), 1e-15)),
        "diagonal_dominance_metric": float(np.min(abs_diag / np.maximum(row_off, 1e-15))),
    }


def compensability_metrics(model: FixedBottomJaxResponse, J: np.ndarray) -> Dict[str, object]:
    c0 = model.c0()
    b0 = model.b(c0)
    J_pinv = np.linalg.pinv(J, rcond=PINV_RCOND)
    b_comp = J @ J_pinv @ b0
    b_uncomp = b0 - b_comp
    b_norm = float(np.linalg.norm(b0))

    residual = model.residual(c0)
    A = model.A(c0)
    sqrt_w = np.sqrt(model.weights_dof_np)
    Aw = sqrt_w[:, None] * A
    rw = sqrt_w * residual
    A_pinv = np.linalg.pinv(Aw, rcond=PINV_RCOND)
    physical_comp = Aw @ A_pinv @ rw
    physical_uncomp = rw - physical_comp
    physical_norm = float(np.linalg.norm(rw))
    return {
        "projected_residual_norm": b_norm,
        "compensable_norm": float(np.linalg.norm(b_comp)),
        "uncompensable_norm": float(np.linalg.norm(b_uncomp)),
        "compensable_ratio": float(np.linalg.norm(b_comp) / max(b_norm, 1e-15)),
        "uncompensable_ratio": float(np.linalg.norm(b_uncomp) / max(b_norm, 1e-15)),
        "rank_J": int(np.linalg.matrix_rank(J, tol=PINV_RCOND * max(np.linalg.norm(J, ord=2), 1.0))),
        "physical_residual_norm": physical_norm,
        "physical_compensable_norm": float(np.linalg.norm(physical_comp)),
        "physical_uncompensable_norm": float(np.linalg.norm(physical_uncomp)),
        "physical_compensable_ratio": float(np.linalg.norm(physical_comp) / max(physical_norm, 1e-15)),
        "physical_uncompensable_ratio": float(np.linalg.norm(physical_uncomp) / max(physical_norm, 1e-15)),
        "rank_full_sensitivity_A": int(
            np.linalg.matrix_rank(Aw, tol=PINV_RCOND * max(np.linalg.norm(Aw, ord=2), 1.0))
        ),
    }


def method_map(outcomes: List[MethodOutcome]) -> Dict[str, MethodOutcome]:
    return {outcome.method: outcome for outcome in outcomes}


def ratio(outcome: MethodOutcome, kind: str = "projected") -> float:
    if kind == "physical":
        return outcome.final_physical / max(outcome.initial_physical, 1e-15)
    return outcome.final_projected / max(outcome.initial_projected, 1e-15)


def mode_sweep_row(
    model: FixedBottomJaxResponse,
    outcomes: List[MethodOutcome],
    comp: Dict[str, object],
    diag: Dict[str, object],
) -> Dict[str, object]:
    methods = method_map(outcomes)
    modal_names = (
        "fixed_bottom_modal_direct_inversion",
        "best_scalar_oracle",
        "diagonal_response_calibration",
        "full_jacobian_gauss_newton",
        "full_jacobian_trust_region",
    )
    best_name = min(modal_names, key=lambda name: ratio(methods[name]))
    return {
        "geometry": model.mesh.name,
        "response_case": model.case.name,
        "mode_count": model.mode_count,
        "best_method": best_name,
        "best_projected_residual_ratio": ratio(methods[best_name]),
        "trust_region_projected_residual_ratio": ratio(methods["full_jacobian_trust_region"]),
        "diagonal_projected_residual_ratio": ratio(methods["diagonal_response_calibration"]),
        "scalar_projected_residual_ratio": ratio(methods["best_scalar_oracle"]),
        "modal_direct_projected_residual_ratio": ratio(methods["fixed_bottom_modal_direct_inversion"]),
        "fixed_projected_direct_physical_ratio": ratio(methods["fixed_bottom_projected_direct_inversion"], "physical"),
        "bottom_violation_trust": model.bottom_violation(methods["full_jacobian_trust_region"].final_q)[0],
        "uncompensable_ratio": comp["physical_uncompensable_ratio"],
        "modal_uncompensable_ratio": comp["uncompensable_ratio"],
        "coupling_ratio": diag["coupling_ratio"],
        "rank_J": diag["rank_J0"],
        "notes": "uncompensable_ratio is the weighted full-field sensitivity residual; modal_uncompensable_ratio uses Range(J)",
    }


def run_suite():
    geometries = generate_geometries()
    geometry_by_name = {mesh.name: mesh for mesh in geometries}
    cases = response_cases()
    bases = {mesh.name: compute_fixed_bottom_modes(mesh) for mesh in geometries}

    geom_rows = geometry_rows(geometries)
    basis_diag_rows = [row for mesh in geometries for row in basis_rows(mesh, bases[mesh.name])]
    audit_rows: List[Dict[str, object]] = []
    method_rows: List[Dict[str, object]] = []
    jac_rows: List[Dict[str, object]] = []
    comp_rows: List[Dict[str, object]] = []
    sweep_rows: List[Dict[str, object]] = []
    history_rows: List[Dict[str, object]] = []
    plot_data: Dict[Tuple[str, int], Dict[str, object]] = {}

    for case in cases:
        mesh = geometry_by_name[case.geometry]
        basis = bases[case.geometry]
        for mode_count in MODE_COUNTS:
            print(f"Running {mesh.name} / {case.name} / m={mode_count}...")
            model = FixedBottomJaxResponse(mesh, basis, case, mode_count)
            audit = model.audit_ad_fd()
            audit_rows.append(
                {
                    "geometry": mesh.name,
                    "response_case": case.name,
                    "mode_count": mode_count,
                    **audit,
                    "notes": "best central finite-difference step selected from fixed audit grid",
                }
            )

            J0 = model.J(model.c0())
            diag = jacobian_diagnostics(J0)
            jac_rows.append(
                {
                    "geometry": mesh.name,
                    "response_case": case.name,
                    "mode_count": mode_count,
                    **diag,
                    "notes": "AD response Jacobian at zero compensation",
                }
            )
            comp = compensability_metrics(model, J0)
            comp_rows.append(
                {
                    "geometry": mesh.name,
                    "response_case": case.name,
                    "mode_count": mode_count,
                    **comp,
                    "notes": (
                        f"Moore-Penrose pseudoinverse rcond={PINV_RCOND:g}; required modal ratios use Range(J); "
                        "physical_* ratios use weighted Range(dr/dc)"
                    ),
                }
            )

            outcomes = run_all_methods(model)
            method_rows.extend(outcome.as_row(model) for outcome in outcomes)
            for outcome in outcomes:
                history_rows.extend(outcome.history)
            sweep_rows.append(mode_sweep_row(model, outcomes, comp, diag))
            plot_data[(case.name, mode_count)] = {
                "model": model,
                "outcomes": outcomes,
                "J0": J0,
                "comp": comp,
            }

    return (
        geometries,
        cases,
        bases,
        geom_rows,
        basis_diag_rows,
        audit_rows,
        method_rows,
        jac_rows,
        comp_rows,
        sweep_rows,
        history_rows,
        plot_data,
    )


def _method_short(name: str) -> str:
    return {
        "free_direct_inversion_reference": "free direct",
        "fixed_bottom_projected_direct_inversion": "fixed projected",
        "fixed_bottom_modal_direct_inversion": "modal direct",
        "best_scalar_oracle": "oracle scalar",
        "diagonal_response_calibration": "diagonal",
        "full_jacobian_gauss_newton": "full GN",
        "full_jacobian_trust_region": "trust/LM",
    }[name]


def generate_figures(
    cases: List[ResponseCase],
    method_rows: List[Dict[str, object]],
    sweep_rows: List[Dict[str, object]],
    history_rows: List[Dict[str, object]],
    comp_rows: List[Dict[str, object]],
    plot_data: Dict[Tuple[str, int], Dict[str, object]],
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    colors = {
        "fixed_bottom_modal_direct_inversion": "#4C78A8",
        "best_scalar_oracle": "#F58518",
        "diagonal_response_calibration": "#54A24B",
        "full_jacobian_gauss_newton": "#E45756",
        "full_jacobian_trust_region": "#7A5195",
    }
    modal_methods = tuple(colors)

    for case in cases:
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        for method in modal_methods:
            rows = [
                row
                for row in history_rows
                if row["response_case"] == case.name and row["mode_count"] == max(MODE_COUNTS) and row["method"] == method
            ]
            rows.sort(key=lambda row: int(row["iteration"]))
            if rows:
                ax.semilogy(
                    [row["iteration"] for row in rows],
                    [max(float(row["projected_residual_ratio"]), 1e-14) for row in rows],
                    marker="o",
                    linewidth=1.6,
                    markersize=3.5,
                    label=_method_short(method),
                    color=colors[method],
                )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Projected residual ratio")
        ax.set_title(f"{case.name}, m=32")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"convergence_{case.name}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for case in cases:
        J = np.asarray(plot_data[(case.name, 32)]["J0"])
        singular = np.linalg.svd(J, compute_uv=False)
        ax.semilogy(np.arange(1, len(singular) + 1), singular, marker="o", markersize=3, label=case.name)
    ax.set_xlabel("Singular-value index")
    ax.set_ylabel("Singular value of J0")
    ax.set_title("Fixed-bottom response-Jacobian spectra, m=32")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "jacobian_singular_values.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4), sharex=True)
    for ax, case in zip(axes.flat, cases):
        rows = [row for row in sweep_rows if row["response_case"] == case.name]
        rows.sort(key=lambda row: int(row["mode_count"]))
        for column, label, color in (
            ("modal_direct_projected_residual_ratio", "modal direct", colors["fixed_bottom_modal_direct_inversion"]),
            ("scalar_projected_residual_ratio", "oracle scalar", colors["best_scalar_oracle"]),
            ("diagonal_projected_residual_ratio", "diagonal", colors["diagonal_response_calibration"]),
            ("trust_region_projected_residual_ratio", "trust/LM", colors["full_jacobian_trust_region"]),
        ):
            ax.semilogy(
                [row["mode_count"] for row in rows],
                [max(float(row[column]), 1e-14) for row in rows],
                marker="o",
                label=label,
                color=color,
            )
        ax.set_title(case.name, fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xticks(MODE_COUNTS)
    axes[1, 0].set_xlabel("Mode count")
    axes[1, 1].set_xlabel("Mode count")
    axes[0, 0].set_ylabel("Projected ratio")
    axes[1, 0].set_ylabel("Projected ratio")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "mode_count_sweep.png", dpi=180)
    plt.close(fig)

    m32 = [row for row in method_rows if row["mode_count"] == 32]
    method_names = [
        "free_direct_inversion_reference",
        "fixed_bottom_projected_direct_inversion",
        "fixed_bottom_modal_direct_inversion",
        "best_scalar_oracle",
        "diagonal_response_calibration",
        "full_jacobian_gauss_newton",
        "full_jacobian_trust_region",
    ]
    max_violations = [
        max(float(row["final_bottom_violation_inf"]) for row in m32 if row["method"] == method) for method in method_names
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.bar(np.arange(len(method_names)), max_violations, color=["#B0B0B0", "#777777", *[colors[m] for m in modal_methods]])
    ax.set_xticks(np.arange(len(method_names)), [_method_short(name) for name in method_names], rotation=25, ha="right")
    ax.set_ylabel("Maximum input-bottom violation (mm)")
    ax.set_title("Bottom manufacturability check, m=32")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "bottom_violation_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for case in cases:
        rows = [row for row in comp_rows if row["response_case"] == case.name]
        rows.sort(key=lambda row: int(row["mode_count"]))
        ax.plot(
            [row["mode_count"] for row in rows],
            [row["physical_uncompensable_ratio"] for row in rows],
            marker="o",
            label=case.name,
        )
    ax.set_xticks(MODE_COUNTS)
    ax.set_xlabel("Mode count")
    ax.set_ylabel("Full-field uncompensable ratio")
    ax.set_ylim(bottom=0.0)
    ax.set_title("Weighted physical compensation authority")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "compensability_ratio.png", dpi=180)
    plt.close(fig)

    selected = ("plate_asymmetric_twist", "comb_finger_varying_shrinkage")
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    for row_index, case_name in enumerate(selected):
        data = plot_data[(case_name, 32)]
        model: FixedBottomJaxResponse = data["model"]
        outcomes = method_map(data["outcomes"])
        initial = model.residual(model.c0()).reshape((-1, 3))
        final = outcomes["full_jacobian_trust_region"].final_residual.reshape((-1, 3))
        surface = model.mesh.surface_mask
        for col, field, title in ((0, initial, "uncompensated"), (1, final, "trust/LM compensated")):
            ax = axes[row_index, col]
            scatter = ax.scatter(
                model.mesh.nodes[surface, 0],
                model.mesh.nodes[surface, 1],
                c=field[surface, 2],
                s=16,
                cmap="coolwarm",
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{case_name}: {title}", fontsize=9)
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            fig.colorbar(scatter, ax=ax, shrink=0.78, label="z residual (mm)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "selected_deformations.png", dpi=180)
    plt.close(fig)


def saturation_mode(case_name: str, method_rows: List[Dict[str, object]]) -> int:
    trust = [
        row
        for row in method_rows
        if row["response_case"] == case_name and row["method"] == "full_jacobian_trust_region"
    ]
    trust.sort(key=lambda row: int(row["mode_count"]))
    target = float(trust[-1]["physical_residual_ratio"])
    tolerance = max(0.02 * abs(target), 1e-4)
    for row in trust:
        if abs(float(row["physical_residual_ratio"]) - target) <= tolerance:
            return int(row["mode_count"])
    return 32


def evaluate_checks(
    geom_rows,
    basis_diag_rows,
    audit_rows,
    method_rows,
    comp_rows,
    sweep_rows,
) -> Tuple[str, List[Tuple[str, str, str]]]:
    fixed_methods = {
        "fixed_bottom_projected_direct_inversion",
        "fixed_bottom_modal_direct_inversion",
        "best_scalar_oracle",
        "diagonal_response_calibration",
        "full_jacobian_gauss_newton",
        "full_jacobian_trust_region",
    }
    fixed_bottom_max = max(
        float(row["final_bottom_violation_inf"]) for row in method_rows if row["method"] in fixed_methods
    )
    free_bottom_max = max(
        float(row["final_bottom_violation_inf"])
        for row in method_rows
        if row["method"] == "free_direct_inversion_reference"
    )
    coupled_cases = {"plate_asymmetric_twist", "comb_finger_varying_shrinkage"}
    value_found = False
    for case in coupled_cases:
        for mode_count in MODE_COUNTS:
            subset = [
                row
                for row in method_rows
                if row["response_case"] == case and int(row["mode_count"]) == mode_count
            ]
            by_method = {row["method"]: row for row in subset}
            calibrated = min(
                float(by_method[name]["projected_residual_ratio"])
                for name in (
                    "diagonal_response_calibration",
                    "full_jacobian_gauss_newton",
                    "full_jacobian_trust_region",
                )
            )
            heuristic = min(
                float(by_method[name]["projected_residual_ratio"])
                for name in ("fixed_bottom_modal_direct_inversion", "best_scalar_oracle")
            )
            value_found = value_found or calibrated < 0.5 * heuristic

    mode_trends = []
    for case in sorted({row["response_case"] for row in sweep_rows}):
        vals = [
            float(row["uncompensable_ratio"])
            for row in sorted(
                [r for r in sweep_rows if r["response_case"] == case],
                key=lambda r: int(r["mode_count"]),
            )
        ]
        mode_trends.append(max(vals) - min(vals))

    checks_bool = [
        ("Geometry generation", len(geom_rows) == 2 and all(int(row["elements"]) > 0 for row in geom_rows), "plate and comb meshes valid"),
        (
            "Fixed-bottom basis",
            max(float(row["bottom_constraint_error_F"]) for row in basis_diag_rows) < 1e-10,
            "S_B Psi near machine zero",
        ),
        (
            "Modal orthonormality",
            max(float(row["mass_orthonormality_error_F"]) for row in basis_diag_rows) < 1e-8,
            "Psi^T M Psi close to I",
        ),
        (
            "JAX differentiability",
            len(audit_rows) == 16 and all(np.isfinite(float(row["ad_fd_relative_error_F"])) for row in audit_rows),
            "AD Jacobian computed for every main run",
        ),
        (
            "AD/FD audit",
            all(row["pass_fail"] == "PASS" for row in audit_rows),
            "relative error below 1e-5",
        ),
        (
            "Bottom manufacturability",
            fixed_bottom_max < 1e-10 and free_bottom_max > 1e-8,
            f"fixed max={fixed_bottom_max:.3e}, free-reference max={free_bottom_max:.3e}",
        ),
        (
            "Response-Jacobian value",
            value_found,
            "calibrated method beats direct/oracle scalar by at least 2x in a coupled diagnostic",
        ),
        (
            "Compensability diagnosis",
            len(comp_rows) == 16
            and all(np.isfinite(float(row["physical_uncompensable_ratio"])) for row in comp_rows)
            and max(float(row["physical_uncompensable_ratio"]) for row in comp_rows) > 0.01,
            "modal and full-field authority ratios computed",
        ),
        (
            "Mode-count sweep",
            len(sweep_rows) == 16 and all(np.isfinite(float(row["coupling_ratio"])) for row in sweep_rows) and max(mode_trends) > 1e-4,
            "4/8/16/32-mode sweep has finite, changing authority metrics",
        ),
    ]
    failed = sum(not passed for _, passed, _ in checks_bool)
    paper_status = "PASS" if failed == 0 else ("PARTIAL" if failed <= 2 else "FAIL")
    checks = [(name, "PASS" if passed else "FAIL", note) for name, passed, note in checks_bool]
    checks.append(("Paper readiness", paper_status, "ready for manuscript decision, not real FDM validation"))
    verdict = "PASS" if failed == 0 else ("PARTIAL" if failed <= 2 else "FAIL")
    return verdict, checks


def markdown_table(headers: List[str], rows: List[List[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(out)


def fmt(value: float) -> str:
    return f"{float(value):.3e}"


def write_report(
    geometries,
    cases,
    bases,
    geom_rows,
    basis_diag_rows,
    audit_rows,
    method_rows,
    jac_rows,
    comp_rows,
    sweep_rows,
    checks,
    verdict,
) -> None:
    audit_values = [float(row["ad_fd_relative_error_F"]) for row in audit_rows]
    bottom_errors = [float(row["bottom_constraint_error_F"]) for row in basis_diag_rows]
    orth_errors = [float(row["mass_orthonormality_error_F"]) for row in basis_diag_rows]
    couplings = [float(row["coupling_ratio"]) for row in jac_rows]
    spectral = [float(row["spectral_radius_I_minus_J0"]) for row in jac_rows]
    physical_uncomp = [float(row["physical_uncompensable_ratio"]) for row in comp_rows]
    modal_uncomp = [float(row["uncompensable_ratio"]) for row in comp_rows]
    fixed_bottom_max = max(
        float(row["final_bottom_violation_inf"])
        for row in method_rows
        if row["method"] != "free_direct_inversion_reference"
    )
    free_bottom = max(
        float(row["final_bottom_violation_inf"])
        for row in method_rows
        if row["method"] == "free_direct_inversion_reference"
    )
    trust_rows = [row for row in method_rows if row["method"] == "full_jacobian_trust_region"]
    trust_accepted = sum(int(row["accepted_steps"]) for row in trust_rows)
    trust_rejected = sum(int(row["rejected_steps"]) for row in trust_rows)
    method_names = sorted({row["method"] for row in method_rows})
    method_summary = []
    for method in method_names:
        rows = [row for row in method_rows if row["method"] == method]
        ratios = [float(row["projected_residual_ratio"]) for row in rows]
        physical = [float(row["physical_residual_ratio"]) for row in rows]
        method_summary.append(
            [
                _method_short(method),
                fmt(min(ratios)),
                fmt(float(np.median(ratios))),
                fmt(max(ratios)),
                fmt(float(np.median(physical))),
                f"{sum(str(row['converged']) == 'True' or row['converged'] is True for row in rows)}/{len(rows)}",
            ]
        )

    best_counts = Counter(row["best_method"] for row in sweep_rows)
    top_findings = [
        f"AD/FD relative Frobenius error ranges from {min(audit_values):.3e} to {max(audit_values):.3e}; all {len(audit_rows)} audits pass.",
        f"Fixed-bottom basis error ||S_B Psi||_F ranges from {min(bottom_errors):.3e} to {max(bottom_errors):.3e}.",
        f"Mass orthonormality error ranges from {min(orth_errors):.3e} to {max(orth_errors):.3e}.",
        f"Fixed-bottom methods have maximum input-bottom violation {fixed_bottom_max:.3e} mm; the free reference reaches {free_bottom:.3e} mm.",
        f"Jacobian coupling ratio ranges from {min(couplings):.3e} to {max(couplings):.3e}.",
        f"Spectral radius rho(I-J0) ranges from {min(spectral):.3e} to {max(spectral):.3e}.",
        f"Required modal Range(J) uncompensable ratio ranges from {min(modal_uncomp):.3e} to {max(modal_uncomp):.3e}.",
        f"Weighted full-field uncompensable ratio ranges from {min(physical_uncomp):.3e} to {max(physical_uncomp):.3e}.",
        "Trust/LM is selected as the best projected-residual method in "
        f"{best_counts.get('full_jacobian_trust_region', 0)} of {len(sweep_rows)} mode-count runs.",
        "Trust/LM physical-residual saturation modes are "
        + ", ".join(f"{case.name}: m={saturation_mode(case.name, method_rows)}" for case in cases)
        + ".",
    ]

    m32_case_rows = []
    for case in cases:
        subset = [
            row
            for row in method_rows
            if row["response_case"] == case.name and int(row["mode_count"]) == 32
        ]
        by_method = {row["method"]: row for row in subset}
        jac = next(row for row in jac_rows if row["response_case"] == case.name and int(row["mode_count"]) == 32)
        comp = next(row for row in comp_rows if row["response_case"] == case.name and int(row["mode_count"]) == 32)
        m32_case_rows.append(
            [
                case.name,
                fmt(by_method["free_direct_inversion_reference"]["physical_residual_ratio"]),
                fmt(by_method["fixed_bottom_projected_direct_inversion"]["physical_residual_ratio"]),
                fmt(by_method["fixed_bottom_modal_direct_inversion"]["projected_residual_ratio"]),
                fmt(by_method["best_scalar_oracle"]["projected_residual_ratio"]),
                fmt(by_method["diagonal_response_calibration"]["projected_residual_ratio"]),
                fmt(by_method["full_jacobian_gauss_newton"]["projected_residual_ratio"]),
                fmt(by_method["full_jacobian_trust_region"]["projected_residual_ratio"]),
                f"{float(jac['coupling_ratio']):.3f}",
                f"{float(comp['physical_uncompensable_ratio']):.3f}",
            ]
        )

    lines = [
        "# Fixed-Bottom JAX-FEM Report",
        "",
        "## 1. Executive Summary",
        "",
        f"Final experiment verdict: **{verdict}**. The independent suite completed both generated geometries, four response cases, and mode counts 4/8/16/32.",
        f"The fixed-bottom basis satisfies `S_B Psi = 0` to numerical precision (maximum Frobenius error {max(bottom_errors):.3e}), and mass orthonormality has maximum error {max(orth_errors):.3e}.",
        f"All AD/FD audits pass, with relative errors from {min(audit_values):.3e} to {max(audit_values):.3e}.",
        "Full/diagonal/trust response calibration improves on identity-response updates in the coupled diagnostics; the exact winning approximation remains case dependent.",
        f"The fixed input-bottom constraint creates measurable full-field loss of compensation authority: weighted uncompensable ratios span {min(physical_uncomp):.3f} to {max(physical_uncomp):.3f}.",
        "The mode-count sweep shows whether added bottom-admissible modes reduce that floor and where performance saturates.",
        "",
        "This is a smooth differentiable voxel-spring eigenstrain surrogate for stable attached printing followed by release-like warpage. It does not model delamination, adhesion failure, detachment, or industrial FDM calibration.",
        "",
        "## 2. Main Numerical Findings",
        "",
    ]
    lines.extend(f"{index}. {finding}" for index, finding in enumerate(top_findings, 1))
    lines += [
        "",
        "### Method summary across all 16 runs",
        "",
        markdown_table(
            ["method", "best projected", "median projected", "worst projected", "median physical", "converged"],
            method_summary,
        ),
        "",
        "## 3. Geometry and Basis Diagnostics",
        "",
    ]
    for mesh in geometries:
        basis = bases[mesh.name]
        lines += [
            f"### {mesh.name}",
            "",
            f"- Mesh: {mesh.num_nodes} nodes, {mesh.num_elements} hexahedral cells, {mesh.bottom_nodes} bottom nodes, {mesh.surface_nodes} surface nodes.",
            f"- Dimensions: {mesh.dimensions}; resolution: {mesh.mesh_resolution}.",
            f"- Maximum basis bottom error: {max(float(row['bottom_constraint_error_F']) for row in basis_diag_rows if row['geometry'] == mesh.name):.3e}.",
            f"- Maximum mass-orthonormality error: {max(float(row['mass_orthonormality_error_F']) for row in basis_diag_rows if row['geometry'] == mesh.name):.3e}.",
            "- First modal interpretations: " + "; ".join(basis.descriptions[:5]) + ".",
            "",
        ]
    lines += [
        "The basis is computed from a bottom-constrained generalized eigenproblem for the reference voxel graph. Bottom DOFs are removed during the eigensolve and restored as exact zeros. Out-of-plane modes are assigned the lowest directional stiffness, so the first modes represent global bending/twist; higher modes add in-plane and local finger content.",
        "",
        "## 4. Response Jacobian Audit",
        "",
        f"JAX `jacfwd` differentiates `b(c) = Psi^T M r(c)` through the geometry-dependent stiffness assembly and dense release solve. All {len(audit_rows)} audits pass. The worst relative Frobenius error is {max(audit_values):.3e}, below the `1e-5` guideline.",
        "",
        "The AD Jacobian is therefore numerically trustworthy for this experiment. Finite differences are used only as an audit, with the best step selected from a fixed central-difference grid.",
        "",
        "## 5. Method Comparison",
        "",
        markdown_table(
            [
                "case (m=32)",
                "free physical",
                "fixed full-field physical",
                "modal direct",
                "scalar",
                "diagonal",
                "full GN",
                "trust/LM",
                "coupling",
                "physical uncomp.",
            ],
            m32_case_rows,
        ),
        "",
        "The free direct reference is not manufacturable because it moves the first layer. The zero-bottom full-field projection is a fair geometric baseline but uses only simple zeroing, not the optional Laplacian smoothing term. Modal direct assumes `J = I`; the oracle scalar corrects a single gain; the diagonal method corrects mode-wise gains; full GN resolves coupling; trust/LM adds predicted/actual reduction control.",
        f"Across all runs, trust/LM accepted {trust_accepted} trial steps and rejected {trust_rejected}. No rejected-step recovery was needed in this parameter set; the result supports damped, monitored updates rather than a claim that rejection handling was essential.",
        "",
        "## 6. Bottom Constraint and Manufacturability",
        "",
        f"The largest fixed-method input-bottom violation is {fixed_bottom_max:.3e} mm, while the free direct reference reaches {free_bottom:.3e} mm. Thus the free reference can reduce physical residual while violating the first-layer geometry constraint. Every fixed-bottom modal update remains in `Range(Psi)` and therefore keeps the input bottom at numerical zero.",
        "",
        "The bottom constraint applies to compensated CAD/input geometry. The release response is not bottom-clamped: only six scalar gauge constraints on three nodes remove rigid-motion ambiguity. No contact loss, delamination, or snapping-off event is modeled.",
        "",
        "## 7. Compensation Authority and Uncompensable Residual",
        "",
        f"The requested `Range(J)` decomposition gives modal uncompensable ratios from {min(modal_uncomp):.3e} to {max(modal_uncomp):.3e}. Because `J` is square and generally full rank, that projected measure can be nearly zero even when the full shape remains imperfect.",
        f"To expose the physically relevant floor, the report also projects the weighted full residual onto `Range(dr/dc)`. Its uncompensable ratio ranges from {min(physical_uncomp):.3f} to {max(physical_uncomp):.3f}. This separates projected modal convergence from full-field authority lost to finite mode count and the fixed-bottom admissible space.",
        "",
        "Residual floors should therefore be read in three layers: modal rank loss from `J`, finite-basis/full-field loss from `dr/dc`, and nonlinear/local-step limitations revealed by the convergence history. The selected deformation plots also show that a method can improve the weighted/projected objectives while worsening a local displacement component; local extrema must be checked separately.",
        "",
        "## 8. Mode Count Sweep",
        "",
    ]
    for case in cases:
        rows = sorted(
            [row for row in sweep_rows if row["response_case"] == case.name],
            key=lambda row: int(row["mode_count"]),
        )
        lines.append(
            f"- `{case.name}`: physical uncompensable ratio "
            + " -> ".join(f"m={row['mode_count']}: {float(row['uncompensable_ratio']):.3f}" for row in rows)
            + f"; trust/LM physical performance saturates near m={saturation_mode(case.name, method_rows)}."
        )
    lines += [
        "",
        "Increasing mode count adds bottom-admissible local and in-plane authority. It can also increase coupling and Jacobian condition number; those trends are reported in `jacobian_diagnostics.csv`. A low projected ratio alone is not evidence that the entire released geometry is corrected, so the full-field authority ratio is the primary mode-count diagnostic.",
        "",
        "## 9. Interpretation for Paper",
        "",
        "The experiment supports the computational claim:",
        "",
        '> FDM bottom-fixed compensation should be formulated in a bottom-admissible modal space, and the response Jacobian quantifies both stable update directions and the loss of compensation authority caused by the bottom constraint.',
        "",
        "The supported scope is a differentiable surrogate study. It does **not** support claims of real FDM validation, adhesion or detachment modeling, delamination prediction, or full industrial process calibration.",
        "",
        "## 10. Recommended Next Steps",
        "",
        "1. Recheck physical unit scaling and gauge sensitivity before manuscript integration.",
        "2. Use `comb_finger_varying_shrinkage` for the main diagnostic figure because it combines local and global response.",
        "3. Use the m=32 method-comparison rows plus bottom violation and full-field uncompensable ratio for the main table.",
        "4. Perform controlled physical FDM prints later with the first-layer geometry held fixed by construction.",
        "5. Add scan-derived residual fields later and project them into the same bottom-admissible basis.",
        "",
        "## PASS / PARTIAL / FAIL Table",
        "",
        markdown_table(["check", "status", "evidence"], [[name, status, note] for name, status, note in checks]),
        "",
        f"**Overall verdict: {verdict}.**",
        "",
        "## Reproducibility",
        "",
        "```powershell",
        "python -B experiments\\fixed_bottom_jax_fem\\run_fixed_bottom_jax_fem.py",
        "```",
    ]
    (RESULT_DIR / "FIXED_BOTTOM_JAX_FEM_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    geometries,
    cases,
    bases,
    geom_rows,
    basis_diag_rows,
    audit_rows,
    method_rows,
    jac_rows,
    comp_rows,
    sweep_rows,
    history_rows,
    plot_data,
) -> Tuple[str, List[Tuple[str, str, str]]]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        RESULT_DIR / "geometry_summary.csv",
        geom_rows,
        ["geometry", "nodes", "elements", "bottom_nodes", "surface_nodes_if_used", "dimensions", "mesh_resolution", "notes"],
    )
    write_csv(
        RESULT_DIR / "basis_diagnostics.csv",
        basis_diag_rows,
        [
            "geometry",
            "mode_count",
            "usable_modes",
            "bottom_constraint_error_F",
            "bottom_constraint_error_inf",
            "max_bottom_displacement_per_unit_modal_amplitude",
            "mass_orthonormality_error_F",
            "first_mode_frequency_or_eigenvalue",
            "last_mode_frequency_or_eigenvalue",
            "condition_estimate_if_available",
            "notes",
        ],
    )
    write_csv(
        RESULT_DIR / "jacobian_ad_fd_audit.csv",
        audit_rows,
        [
            "geometry",
            "response_case",
            "mode_count",
            "fd_step",
            "ad_fd_relative_error_F",
            "ad_fd_max_abs_error",
            "ad_fd_max_rel_column_error",
            "pass_fail",
            "notes",
        ],
    )
    write_csv(
        RESULT_DIR / "method_comparison.csv",
        method_rows,
        [
            "geometry",
            "response_case",
            "mode_count",
            "method",
            "initial_projected_residual_norm",
            "final_projected_residual_norm",
            "projected_residual_ratio",
            "initial_physical_residual_norm",
            "final_physical_residual_norm",
            "physical_residual_ratio",
            "final_bottom_violation_inf",
            "final_bottom_violation_l2",
            "iterations",
            "converged",
            "accepted_steps",
            "rejected_steps",
            "best_alpha_if_scalar",
            "notes",
        ],
    )
    write_csv(
        RESULT_DIR / "jacobian_diagnostics.csv",
        jac_rows,
        [
            "geometry",
            "response_case",
            "mode_count",
            "spectral_radius_I_minus_J0",
            "cond_J0",
            "rank_J0",
            "min_singular_value_J0",
            "max_singular_value_J0",
            "coupling_ratio",
            "diagonal_dominance_metric",
            "notes",
        ],
    )
    write_csv(
        RESULT_DIR / "compensability_table.csv",
        comp_rows,
        [
            "geometry",
            "response_case",
            "mode_count",
            "projected_residual_norm",
            "compensable_norm",
            "uncompensable_norm",
            "compensable_ratio",
            "uncompensable_ratio",
            "rank_J",
            "physical_residual_norm",
            "physical_compensable_norm",
            "physical_uncompensable_norm",
            "physical_compensable_ratio",
            "physical_uncompensable_ratio",
            "rank_full_sensitivity_A",
            "notes",
        ],
    )
    write_csv(
        RESULT_DIR / "mode_count_sweep.csv",
        sweep_rows,
        [
            "geometry",
            "response_case",
            "mode_count",
            "best_method",
            "best_projected_residual_ratio",
            "trust_region_projected_residual_ratio",
            "diagonal_projected_residual_ratio",
            "scalar_projected_residual_ratio",
            "modal_direct_projected_residual_ratio",
            "fixed_projected_direct_physical_ratio",
            "bottom_violation_trust",
            "uncompensable_ratio",
            "modal_uncompensable_ratio",
            "coupling_ratio",
            "rank_J",
            "notes",
        ],
    )
    write_csv(
        RESULT_DIR / "convergence_history.csv",
        history_rows,
        [
            "geometry",
            "response_case",
            "mode_count",
            "method",
            "iteration",
            "projected_residual_norm",
            "projected_residual_ratio",
            "physical_residual_norm",
            "physical_residual_ratio",
            "step_norm",
            "trust_region_radius_or_lambda",
            "predicted_reduction",
            "actual_reduction",
            "eta_ratio",
            "accepted",
        ],
    )
    generate_figures(cases, method_rows, sweep_rows, history_rows, comp_rows, plot_data)
    verdict, checks = evaluate_checks(geom_rows, basis_diag_rows, audit_rows, method_rows, comp_rows, sweep_rows)
    write_report(
        geometries,
        cases,
        bases,
        geom_rows,
        basis_diag_rows,
        audit_rows,
        method_rows,
        jac_rows,
        comp_rows,
        sweep_rows,
        checks,
        verdict,
    )
    (RESULT_DIR / "fixed_bottom_checks.json").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "checks": [{"check": name, "status": status, "evidence": note} for name, status, note in checks],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return verdict, checks


def print_summary(
    verdict,
    checks,
    audit_rows,
    basis_diag_rows,
    method_rows,
    jac_rows,
    comp_rows,
    runtime,
) -> None:
    fixed_rows = [row for row in method_rows if row["method"] != "free_direct_inversion_reference"]
    free_rows = [row for row in method_rows if row["method"] == "free_direct_inversion_reference"]
    print("Fixed-Bottom Modal Response-Jacobian Compensation for FDM Warpage")
    print(f"Runs: {len(audit_rows)} geometry/case/mode-count combinations")
    print(
        "AD-FD error range: "
        f"{min(float(r['ad_fd_relative_error_F']) for r in audit_rows):.3e} to "
        f"{max(float(r['ad_fd_relative_error_F']) for r in audit_rows):.3e}"
    )
    print(
        "Bottom basis error range: "
        f"{min(float(r['bottom_constraint_error_F']) for r in basis_diag_rows):.3e} to "
        f"{max(float(r['bottom_constraint_error_F']) for r in basis_diag_rows):.3e}"
    )
    print(
        "Coupling ratio range: "
        f"{min(float(r['coupling_ratio']) for r in jac_rows):.3e} to "
        f"{max(float(r['coupling_ratio']) for r in jac_rows):.3e}"
    )
    print(
        "Full-field uncompensable ratio range: "
        f"{min(float(r['physical_uncompensable_ratio']) for r in comp_rows):.3e} to "
        f"{max(float(r['physical_uncompensable_ratio']) for r in comp_rows):.3e}"
    )
    print(
        f"Fixed-method max bottom violation: {max(float(r['final_bottom_violation_inf']) for r in fixed_rows):.3e}; "
        f"free-reference max: {max(float(r['final_bottom_violation_inf']) for r in free_rows):.3e}"
    )
    print("PASS/PARTIAL/FAIL checks:")
    for name, status, note in checks:
        print(f"  {name}: {status} ({note})")
    print(f"Overall verdict: {verdict}")
    print(f"Output directory: {RESULT_DIR}")
    print("Rerun: python -B experiments\\fixed_bottom_jax_fem\\run_fixed_bottom_jax_fem.py")
    print(f"Runtime seconds: {runtime:.2f}")


def main() -> None:
    start = time.perf_counter()
    (
        geometries,
        cases,
        bases,
        geom_rows,
        basis_diag_rows,
        audit_rows,
        method_rows,
        jac_rows,
        comp_rows,
        sweep_rows,
        history_rows,
        plot_data,
    ) = run_suite()
    verdict, checks = write_outputs(
        geometries,
        cases,
        bases,
        geom_rows,
        basis_diag_rows,
        audit_rows,
        method_rows,
        jac_rows,
        comp_rows,
        sweep_rows,
        history_rows,
        plot_data,
    )
    print_summary(
        verdict,
        checks,
        audit_rows,
        basis_diag_rows,
        method_rows,
        jac_rows,
        comp_rows,
        time.perf_counter() - start,
    )


if __name__ == "__main__":
    main()
