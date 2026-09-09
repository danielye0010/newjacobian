# R6P Phase 0 — Native B48 Source-Boundary Audit

Decision: **PASS TO PHASE 1**. Every required 32-to-48 source modification is dimension plumbing, output/bookkeeping, or fixed-capacity storage. The residual-design calls and their argument lists at lines 4486–4520, tangent calls and argument lists at lines 4585–4617, accepted-state copying, perturbation definition, assembly order, precision, and solver controls require no change. No category D or unresolved category E occurrence was found.

Authoritative source: `native_M32A_blind/source/nonlingeo.c`, SHA-256 `8b2055f9f6dee896b2ead1a2e7e5831ff66c287d861d79c4b04ee382a8296178`.

| Lines | Current behavior | Proposed minimal isolated-source change | Class |
|---:|---|---|:---:|
| 4358–4362 | Describes fixed M32 acquisition | Describe configured M<=48 orchestration | B |
| 4367 | Holds configured count | Add a fixed maximum-capacity constant; retain runtime `xcount` | A |
| 4372–4374 | Per-evaluation arrays have capacity 33 | Increase capacity to 49 (48 production plus one control) | C |
| 4381 | State/private-hash arrays have capacities 35/33 | Increase capacities to 51/49 | C |
| 4392–4398 | Reads configuration but requires exactly 32; M32 error text | Accept only 1<=count<=48; generalize error text | A/B |
| 4443–4446 | Fixed 33-evaluation loop, production/control split at 32 | Loop through `xcount+1`; split at `xcount` | A |
| 4478 | M32-specific input error text | Generalize label only | B |
| 4523–4527 | Control filenames selected at fixed index 32 and labeled M32A | Select at `xcount`; use qualification-area control name | A/B |
| 4531, 4540, 4544–4548, 4572 | M32 labels/comments and control filename | Generalize labels/comments/name only | B |
| 4619–4623 | Tangent artifact filenames contain M32A | Use stable generalized qualification names | B |
| 4624–4626 | Post-tangent state checkpoint fixed at index 34 | Use `xcount+2` | A |
| 4635–4656 | Runtime filename, counts, order, checkpoint and timing indices fixed to 32/33/34 | Emit configured values and index by `xcount` | A/B |
| 4659–4666 | State-hash filename and loops/indices fixed to 32/33/34 | Loop/index by `xcount` | A/B |
| 4669–4686 | Private-input/lifecycle filenames, loops, roles, and control index fixed to 32/33 | Loop/index by `xcount` | A/B |
| 4695 | M32 completion label | Generalize label only | B |

Classification key: A = pure dimension plumbing; B = output/manifest bookkeeping; C = memory/storage size; D = scientific derivative logic; E = unknown.

Phase-0 preservation checks:

- Frozen final-paper package verification: PASS, 66/66.
- R6A JSON and CSV parse: PASS.
- Git object check: PASS.
- Authoritative M32 source is unchanged at the hash above.

The isolated generalized source may therefore be created. This audit does not qualify B32, B48, or H48.
