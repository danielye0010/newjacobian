from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]
RUN48 = HERE / "run_M48"
PHI = HERE / "phi48"
FDROOT = HERE / "fd_new_modes"
NK = 16557
NPHYS = 5805
H0 = 0.08957732492949613
EPS0 = 0.0004


def main() -> None:
    FDROOT.mkdir(exist_ok=True)
    zeros = np.zeros((NK, 6), dtype=np.float64)
    for mode in range(33, 49):
        out = FDROOT / f"mode_{mode:02d}"
        out.mkdir(exist_ok=True)
        phi = np.zeros((NK, 3), dtype=np.float64)
        phi[:NPHYS] = np.fromfile(PHI / f"modal_phi{mode}.bin", "<f8").reshape(NPHYS, 3)
        table = np.column_stack((np.arange(1, NK + 1), phi, zeros))
        with (out / "proof_vectors.txt").open("w", encoding="ascii", newline="\n") as stream:
            stream.write(f"{NK} {NPHYS} {H0:.17e} {EPS0:.17e}\n")
            np.savetxt(stream, table, fmt=["%d"] + ["%.17e"] * 9)
        for name, source in (
            ("locked_final_vold.bin", RUN48 / "forward_final_vold.bin"),
            ("locked_prestr.bin", RUN48 / "forward_prestr.bin"),
            ("native_M32A.inp", RUN48 / "native_M32A.inp"),
        ):
            shutil.copyfile(source, out / name)
    print("prepared fixed-state FD inputs for modes 33 through 48")


if __name__ == "__main__":
    main()
