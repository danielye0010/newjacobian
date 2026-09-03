"""Synthetic response models for Step 1 response-Jacobian diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

K = 5
C_STAR = np.array([1.0, -0.8, 0.6, -0.4, 0.3], dtype=float)
C0 = np.zeros(K, dtype=float)


@dataclass(frozen=True)
class ResponseModel:
    name: str
    A: np.ndarray
    beta: float
    c_star: np.ndarray = field(default_factory=lambda: C_STAR.copy())

    def error(self, c: np.ndarray) -> np.ndarray:
        return np.asarray(c, dtype=float) - self.c_star

    def h(self, e: np.ndarray) -> np.ndarray:
        e = np.asarray(e, dtype=float)
        return nonlinear_h(e)

    def b(self, c: np.ndarray) -> np.ndarray:
        e = self.error(c)
        return self.A @ e + self.beta * nonlinear_h(e)

    def jacobian(self, c: np.ndarray) -> np.ndarray:
        e = self.error(c)
        return self.A + self.beta * nonlinear_dh_de(e)


def nonlinear_h(e: np.ndarray) -> np.ndarray:
    e = np.asarray(e, dtype=float)
    h = np.zeros(K, dtype=float)
    h[0] = e[0] ** 2
    h[1] = e[0] * e[1]
    h[2] = e[1] ** 2
    h[3] = e[2] * e[3]
    h[4] = e[4] ** 2 + 0.5 * e[0] * e[4]
    return h


def nonlinear_dh_de(e: np.ndarray) -> np.ndarray:
    e = np.asarray(e, dtype=float)
    J = np.zeros((K, K), dtype=float)
    J[0, 0] = 2.0 * e[0]
    J[1, 0] = e[1]
    J[1, 1] = e[0]
    J[2, 1] = 2.0 * e[1]
    J[3, 2] = e[3]
    J[3, 3] = e[2]
    J[4, 0] = 0.5 * e[4]
    J[4, 4] = 2.0 * e[4] + 0.5 * e[0]
    return J


def finite_difference_jacobian(model: ResponseModel, c: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    J = np.zeros((K, K), dtype=float)
    for i in range(K):
        step = np.zeros(K, dtype=float)
        step[i] = eps
        J[:, i] = (model.b(c + step) - model.b(c - step)) / (2.0 * eps)
    return J


def jacobian_check(model: ResponseModel, c: np.ndarray) -> Dict[str, float]:
    Ja = model.jacobian(c)
    Jfd = finite_difference_jacobian(model, c)
    diff = Ja - Jfd
    fd_norm = np.linalg.norm(Jfd, ord="fro")
    return {
        "max_abs_error": float(np.max(np.abs(diff))),
        "relative_frobenius_error": float(np.linalg.norm(diff, ord="fro") / max(fd_norm, 1e-15)),
    }


def coupled_A(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(K, K))
    Q, R = np.linalg.qr(G)
    signs = np.sign(np.diag(R))
    signs[signs == 0.0] = 1.0
    Q = Q * signs
    Lambda = np.diag([0.4, 0.9, 1.5, 2.6, 3.2])
    return Q @ Lambda @ Q.T


def make_cases() -> Dict[str, ResponseModel]:
    A_gain = np.diag([0.3, 0.8, 1.2, 2.5, 3.0])
    A_coupled = coupled_A(42)
    return {
        "gain_mismatch_linear": ResponseModel("gain_mismatch_linear", A_gain, 0.0),
        "coupled_linear": ResponseModel("coupled_linear", A_coupled, 0.0),
        "weak_nonlinear_coupled": ResponseModel("weak_nonlinear_coupled", A_coupled, 0.15),
        "strong_nonlinear_coupled": ResponseModel("strong_nonlinear_coupled", A_coupled, 0.50),
    }


def case_diagnostics(model: ResponseModel, c0: np.ndarray = C0) -> Dict[str, object]:
    J0 = model.jacobian(c0)
    eigvals = np.linalg.eigvals(J0)
    I = np.eye(K)
    offdiag = J0 - np.diag(np.diag(J0))
    rng = np.random.default_rng(123)
    c_rand = rng.normal(size=K)
    return {
        "J0": J0.tolist(),
        "eigenvalues": [[float(np.real(v)), float(np.imag(v))] for v in eigvals],
        "rho_I_minus_J0": float(np.max(np.abs(np.linalg.eigvals(I - J0)))),
        "identity_deviation": float(np.linalg.norm(J0 - I, ord="fro") / np.linalg.norm(I, ord="fro")),
        "coupling_ratio": float(np.linalg.norm(offdiag, ord="fro") / max(np.linalg.norm(J0, ord="fro"), 1e-15)),
        "condition_number": float(np.linalg.cond(J0)),
        "jacobian_check_c0": jacobian_check(model, c0),
        "jacobian_check_random_seed_123": jacobian_check(model, c_rand),
    }


def format_matrix(M: np.ndarray, precision: int = 6) -> str:
    return np.array2string(np.asarray(M), precision=precision, suppress_small=False)
