from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]
RUN = HERE / "run_M48"
AUTH = ROOT / "native_M32A_blind"
BASIS = ROOT / "framework_validity_audit/08_final_continuum_compensation_experiment/parsed/m_c8i_basis.npz"
HIST = ROOT / "response_jacobian_spectral_convergence_audit/N_M48_A_H_J_r0.npz"
NEQ, NZS, NK, NPHYS, NOUT, M = 49536, 1408923, 16557, 5805, 49824, 48
PREFIX_TOL = 1.0e-10
IDENTITY_TOL = 1.0e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def output_operator():
    sys.path[:0] = [
        str(ROOT / "framework_validity_audit/08_final_continuum_compensation_experiment"),
        str(ROOT / "framework_validity_audit/07_external_element_formulation_closure"),
        str(ROOT / "framework_validity_audit/06_energy_locking_and_coupled_extension"),
        str(ROOT / "framework_validity_audit/04_nonlinear_continuum_validation"),
    ]
    from continuum_lbracket_problem import build_common_surface_operator
    from run_energy_locking_audit import custom_mesh
    mesh = custom_mesh(40, 2, axial_h=40 / 128)
    surface = build_common_surface_operator({"nominal": mesh, "fine": mesh})
    return surface.interpolation["nominal"], np.repeat(np.sqrt(surface.weights / surface.total_area), 3)


def main() -> None:
    solver_meta = dict(
        line.split("=", 1)
        for line in (RUN / "B48Q_solver_runtime.txt").read_text().splitlines()
        if "=" in line
    )
    assert solver_meta["factorization_count"] == "1"
    assert solver_meta["backsolve_count"] == "48"
    assert solver_meta["solve_order"] == "1_to_48"

    bmat = np.column_stack([np.fromfile(RUN / f"Ra{mode}_native.bin", "<f8") for mode in range(1, M + 1)])
    umat = np.column_stack([np.fromfile(RUN / f"ua{mode}_active_49536.bin", "<f8") for mode in range(1, M + 1)])
    amap = np.fromfile(RUN / "active_dof_map.bin", "<i4").reshape(NK, 4)
    physical_map = amap[:NPHYS, 1:4].astype(np.int64) - 1
    storage = np.zeros((NPHYS, 3, M), dtype=np.float64)
    mask = physical_map >= 0
    storage[mask, :] = umat[physical_map[mask], :]
    with np.load(BASIS, allow_pickle=False) as source:
        modes = np.asarray(source["modes"], dtype=np.float64)[:, :M]
    sbar, weights = output_operator()
    hbar = np.ascontiguousarray(sbar @ (modes + storage.reshape(3 * NPHYS, M)), dtype="<f8")
    h48 = np.ascontiguousarray(weights[:, None] * hbar, dtype="<f8")
    assert h48.shape == (NOUT, M) and np.isfinite(h48).all()
    for stem, value in (("Hbar_native_49824x48", hbar), ("H48_native_49824x48", h48), ("U_a_active_49536x48", umat)):
        value.tofile(HERE / f"{stem}.bin")
        np.save(HERE / f"{stem}.npy", value, allow_pickle=False)

    auth_h32 = np.fromfile(AUTH / "H_native_49824x32.bin", "<f8").reshape(NOUT, 32)
    prefix_errors = np.asarray([rel(h48[:, index], auth_h32[:, index]) for index in range(32)])
    prefix = {
        "verdict": "PASS" if rel(h48[:, :32], auth_h32) <= PREFIX_TOL and np.all(prefix_errors <= PREFIX_TOL) else "FAIL",
        "preregistered_relative_tolerance": PREFIX_TOL,
        "frobenius_prefix_relative_error": rel(h48[:, :32], auth_h32),
        "maximum_prefix_column_relative_error": float(prefix_errors.max()),
        "median_prefix_column_relative_error": float(np.median(prefix_errors)),
        "all_prefix_bytes_equal": h48[:, :32].tobytes() == auth_h32.tobytes(),
        "per_column_relative_errors": {str(index + 1): float(value) for index, value in enumerate(prefix_errors)},
        "H48_sha256": sha256(HERE / "H48_native_49824x48.bin"),
        "H48_npy_sha256": sha256(HERE / "H48_native_49824x48.npy"),
        "authoritative_H32_sha256": sha256(AUTH / "H_native_49824x32.bin"),
        "solver_source_sha256": sha256(HERE / "solver" / "solve_B48Q.c"),
        "solver_executable_sha256": sha256(HERE / "solver" / "solve_B48Q.exe"),
        "factorization_count": int(solver_meta["factorization_count"]),
        "backsolve_count": int(solver_meta["backsolve_count"]),
    }
    (HERE / "H48_PREFIX_AUDIT.json").write_text(json.dumps(prefix, indent=2) + "\n", encoding="utf-8")
    if prefix["verdict"] != "PASS":
        raise SystemExit(2)

    diagonal = np.fromfile(RUN / "K_M32A_diagonal.bin", "<f8")
    lower = np.fromfile(RUN / "K_M32A_offdiagonal.bin", "<f8")
    jq = np.fromfile(RUN / "K_M32A_jq.bin", "<i4")
    irow = np.fromfile(RUN / "K_M32A_irow.bin", "<i4")
    columns = np.repeat(np.arange(NEQ), np.diff(jq))
    rows = irow.astype(np.int64) - 1
    k = csc_matrix((np.concatenate((diagonal, lower, lower)),
                    (np.concatenate((np.arange(NEQ), rows, columns)),
                     np.concatenate((np.arange(NEQ), columns, rows)))), shape=(NEQ, NEQ))
    asym = k - k.T
    asymmetry = float(np.sqrt(asym.multiply(asym).sum()) / np.sqrt(k.multiply(k).sum()))
    factor = splu(k, permc_spec="COLAMD")

    def active_to_physical(active: np.ndarray) -> np.ndarray:
        physical = np.zeros((NPHYS, 3), dtype=np.float64)
        physical[mask] = active[physical_map[mask]]
        return physical.reshape(-1)

    def physical_to_active(physical: np.ndarray) -> np.ndarray:
        active = np.zeros(NEQ, dtype=np.float64)
        active[physical_map[mask]] = np.asarray(physical).reshape(NPHYS, 3)[mask]
        return active

    def forward(vector: np.ndarray) -> np.ndarray:
        solution = factor.solve(-(bmat @ vector))
        return weights * (sbar @ (modes @ vector + active_to_physical(solution)))

    def adjoint(vector: np.ndarray) -> np.ndarray:
        pullback = sbar.T @ (weights * vector)
        lam = factor.solve(physical_to_active(pullback), trans="T")
        return modes.T @ pullback - bmat.T @ lam

    modal_vectors = []
    for index in (32, 39, 47):
        value = np.zeros(M); value[index] = 1.0; modal_vectors.append(value)
    rng_v = np.random.default_rng(20260909)
    dense = rng_v.standard_normal(M); dense /= np.linalg.norm(dense); modal_vectors.append(dense)
    rng_w = np.random.default_rng(20260910)
    output_vectors = []
    for _ in range(4):
        value = rng_w.standard_normal(NOUT); value /= np.linalg.norm(value); output_vectors.append(value)
    labels = ["e33", "e40", "e48", "dense_seed_20260909"]
    identity_rows = []
    for label, vector, outvec in zip(labels, modal_vectors, output_vectors):
        fwd = forward(vector)
        adj = adjoint(outvec)
        lhs = float(outvec @ fwd)
        rhs = float(vector @ adj)
        scale = max(abs(lhs), abs(rhs), np.finfo(float).tiny)
        discrepancy = abs(lhs - rhs) / scale
        identity_rows.append({
            "label": label,
            "modal_vector_sha256": sha_array(vector),
            "output_vector_sha256": sha_array(outvec),
            "forward_inner_product": lhs,
            "adjoint_inner_product": rhs,
            "absolute_discrepancy": abs(lhs - rhs),
            "normalized_discrepancy": discrepancy,
            "tolerance": IDENTITY_TOL,
            "passed": discrepancy <= IDENTITY_TOL,
        })
    write_csv(HERE / "H48_ADJOINT_IDENTITY.csv", identity_rows)

    historical = {"performed": False}
    if HIST.is_file():
        with np.load(HIST, allow_pickle=False) as source:
            old = np.asarray(source["H"], dtype=np.float64)
        historical = {
            "performed": True,
            "historical_definition": "finite-step response object",
            "new_definition": "qualified native tangent response",
            "qualification_oracle": False,
            "historical_shape": list(old.shape),
            "relative_frobenius_difference": rel(h48, old),
            "columnwise_median_relative_difference": float(np.median([rel(h48[:, j], old[:, j]) for j in range(M)])),
            "columnwise_maximum_relative_difference": float(max(rel(h48[:, j], old[:, j]) for j in range(M))),
        }
    (HERE / "H48_HISTORICAL_CONTEXT.json").write_text(json.dumps(historical, indent=2) + "\n", encoding="utf-8")
    action = {
        "verdict": "PASS" if all(row["passed"] for row in identity_rows) and asymmetry <= 1e-14 else "FAIL",
        "inherited_identity_tolerance": IDENTITY_TOL,
        "represented_tangent_asymmetry_tolerance": 1e-14,
        "represented_tangent_asymmetry_frobenius": asymmetry,
        "maximum_identity_normalized_discrepancy": max(row["normalized_discrepancy"] for row in identity_rows),
        "tests": identity_rows,
        "native_paths": "forward: -Bv -> K solve -> W S(Phi v + u); adjoint: S^T Ww -> K^T solve -> Phi^T pullback - B^T lambda",
    }
    (HERE / "H48_ACTION_QUALIFICATION.json").write_text(json.dumps(action, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prefix": prefix, "action": action, "historical": historical}, indent=2))
    if action["verdict"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
