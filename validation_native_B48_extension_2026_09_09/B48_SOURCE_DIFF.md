# R6P Phase 1 — Minimal Generalized-Source Diff

Baseline: `native_M32A_blind/source/nonlingeo.c` (`8b2055f9f6dee896b2ead1a2e7e5831ff66c287d861d79c4b04ee382a8296178`).

Generalized source: `generalized_source/nonlingeo.c`.

The complete diff is restricted to the Phase-0 inventory:

- capacities `33 -> 49` and `35 -> 51` for 48 production evaluations, one post-loop control, and two state checkpoints;
- configuration gate `xcount == 32 -> 1 <= xcount <= 48`;
- production/control loop bounds and checkpoint indices expressed through `xcount`;
- runtime counts, order fields, per-mode loops, and messages expressed through `xcount`.

The `results_se` and `mafillsmmain_se` residual-derivative calls and argument lists are byte-identical to the frozen source. The `results` and `mafillsmmain` tangent calls and argument lists are byte-identical. No material, eigenstrain, accepted-state copy, perturbation, assembly-order, solver-control, precision, or cleanup logic changed.

Build flags preserved: `-Wall -O2 -fcommon -DARCH=Linux -DSPOOLES -DARPACK -DMATRIXSTORAGE -DNETWORKOUT -fopenmp`. The isolated build uses GCC/GFortran 16.2.0 and the same SPOOLES/ARPACK/PaStiX/SPM/OpenBLAS dependency families. The exact executable and source hashes are frozen in `B48_PROVENANCE.json` after qualification.

An exact live diff can be reproduced with:

```powershell
git diff --no-index -- native_M32A_blind/source/nonlingeo.c validation_native_B48_extension_2026_09_09/generalized_source/nonlingeo.c
```

The official CalculiX 2.23 `mortar.h` was recovered from `https://www.dhondt.de/ccx_2.23.src.tar.bz2` because the local working-tree copy was an unreadable OneDrive placeholder. Its SHA-256 is `1a7e2389d4dfad2774272ac202122aa0095ac3460a0c291cff62f43e3ce66f36` and its byte count (22,011) matches the placeholder metadata.
