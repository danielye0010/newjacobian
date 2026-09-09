from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]
BASIS = ROOT / "framework_validity_audit/08_final_continuum_compensation_experiment/parsed/m_c8i_basis.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    with np.load(BASIS, allow_pickle=False) as source:
        modes = np.asarray(source["modes"], dtype="<f8")
    assert modes.shape == (17415, 64)
    assert np.isfinite(modes[:, :48]).all()
    out = HERE / "phi48"
    rows = []
    for mode in range(1, 49):
        values = np.ascontiguousarray(modes[:, mode - 1], dtype="<f8")
        path = out / f"modal_phi{mode}.bin"
        values.tofile(path)
        rows.append({
            "mode": mode,
            "entries": values.size,
            "dtype": "little-endian float64",
            "order": "physical-node-major XYZ",
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    with (HERE / "B48_PHI_MANIFEST.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"prepared {len(rows)} exact ordered parent-basis columns")


if __name__ == "__main__":
    main()
