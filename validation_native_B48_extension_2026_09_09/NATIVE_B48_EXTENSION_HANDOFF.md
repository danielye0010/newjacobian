# R6P — Native B48 Extension Qualification

## 1. Executive decision: PASS

R6P passes every mandatory gate. The frozen native M32 residual-design derivative was generalized to 48 ordered production directions using dimension plumbing only. The qualified recommendation is to resume the original R6A inverse-information scaling experiment; R6A was not resumed here.

## 2. Why R6P was required

R6A stopped because no authoritative native B48 existed and the validated production acquisition was fixed at 32 columns. R6P establishes whether an M48 oracle can be created without changing the scientific derivative formulation.

## 3. Source-boundary audit

`B48_SOURCE_BOUNDARY_AUDIT.md` and `.json` classify every required occurrence as dimension plumbing, output bookkeeping, or storage capacity. No scientific derivative or unknown change was required. The final authoritative M32 source SHA-256 remains `8b2055f9f6dee896b2ead1a2e7e5831ff66c287d861d79c4b04ee382a8296178`.

## 4. Minimal source diff

`B48_SOURCE_DIFF.md` records only configurable count bounds, loop/index expressions, fixed capacities, and generalized labels/manifests. The `results_se`/`mafillsmmain_se` derivative calls and `results`/`mafillsmmain` tangent calls are unchanged. The isolated H48 solver changes only `NMODE=32` to `48` and associated labels/count fields.

## 5. M32 backward compatibility

PASS: Frobenius relative error `0.00000000000000000e+00`, maximum and median column errors zero, and all 32 columns byte-identical. Detailed results are in `B32_BACKWARD_COMPATIBILITY.json` and `B32_COLUMN_ERRORS.csv`.

## 6. B48 construction provenance

Exactly one production B48 construction was executed after M32 passed. B48 has shape `49536 x 48`, SHA-256 `90c66d7e244d0e03a5692e25bd6fd18a7a060738762e5e3230b5f78f799e79ab`, invariant accepted-state hash `24eb730d35c0744b`, and invariant private-input hashes. Source, executable, Phi-column, RHS-column, and runtime identities are frozen in `B48_PROVENANCE.json`.

## 7. B48 first-32 prefix

PASS: Frobenius prefix relative error `0.00000000000000000e+00`; all first 32 columns are byte-identical to authoritative B32.

## 8. Step refinement for modes 33, 40, and 48

The preregistered scale was `h0=8.95773249294961327e-02` with `h0/2` and `h0/4`. Both candidate refined steps met the common stability rule, so the preregistered finest-common rule selected `h0/4=2.23943312323740332e-02`. `B48_NEW_MODE_STEP_REFINEMENT.csv` records native-vs-FD and FD step-to-step errors. The inherited hook's raw h0/8 output is retained but excluded from selection and gates.

## 9. Modes 33–48 independent FD qualification

PASS at the common confirmation step: median relative error `4.04052088665606880e-06`, p90 `6.26747229856166058e-06`, maximum `7.82233990128457369e-06` at mode `33`, all below the inherited `1e-2` standard. Every signed assembly restored the same locked state; all byte-invariance checks passed.

## 10. B48 classification

PASS. M32 compatibility, first-32 prefix invariance, common stable FD regime, and all 16 new-mode FD gates passed.

## 11. H48 construction

After B48 passed, the frozen K_T was factored once and exactly 48 ordered native forward solves were performed. `H48_native_49824x48.bin` has SHA-256 `fca64c0628acf3196d138ccbb46bd1a4b62beac0ae6306d144189a4f60108ae3`. The solver runtime manifest confirms one factorization and 48 backsolves.

## 12. H48 first-32 reproduction

PASS: Frobenius, maximum-column, and median-column prefix errors are all zero; the complete first-32 prefix is byte-identical to authoritative H32 (`850dd8ad99b4047db0f5732fce4ffb9cf327b91eb83aad419e379539c78d8b03`).

## 13. Forward/adjoint identity

PASS for e33, e40, e48, and the fixed seeded dense vector. Maximum normalized discrepancy is `1.42249405792319574e-13` versus the inherited `1e-10` tolerance; represented tangent asymmetry is `0.00000000000000000e+00`.

## 14. Historical H48 context

Secondary diagnostic only: historical H48 is a finite-step response object, while the new H48 is a native tangent response. Their relative Frobenius difference is `3.95888803318472182e-04`. Historical H48 was not used as a qualification oracle.

## 15. Claims supported and not supported

Supported: dimension-only M48 generalization; exact M32/B48/H48 prefixes; independent qualification of modes 33–48; and new-range native forward/adjoint consistency. Not supported: any R6A inverse/scaling, Direct-start, threshold, nonlinear compensation, timing, M64, or manuscript conclusion.

## 16. Recommendation

Resume the original R6A inverse-information scaling experiment. Do not reinterpret this prerequisite qualification as an R6A result.

## 17. Artifacts and hashes

- `generalized_source/nonlingeo.c` — `34dfba3100fd58f63313702e3f994b8bdef0b981f72531de878f95bceb75c4b2` — qualified dimension-generalized production source
- `generalized_source/native_B48Q_i4.exe` — `4af4cbd8b07e196e918093dca2665b2bcc3d64a828536d292255285d0ac64291` — qualified CalculiX executable
- `generalized_source/ccx_2.23_B48Q.a` — `3797f22ff07be1326bbdc84ef96120e42ac6e8161f1eb7ee44ecdd3aced59488` — isolated build archive
- `solver/solve_B48Q.c` — `a9d42838d61f1e5df253dbead69802b301ceb4967e7e14298a141f439a4302e2` — 48-action direct-solver dimension extension
- `solver/solve_B48Q.exe` — `ba790e9a85c831a2b64b3c96078b2a104c540b48129e76650fa19043b2bfa594` — 48-action direct solver executable
- `B48_native_49536x48.bin` — `90c66d7e244d0e03a5692e25bd6fd18a7a060738762e5e3230b5f78f799e79ab` — qualified native residual-design matrix
- `B48_native_49536x48.npy` — `41678f39c68d48febff73dc25292181ddb625c59c450c46844410056a6f60dc6` — qualified native residual-design matrix with shape metadata
- `H48_native_49824x48.bin` — `fca64c0628acf3196d138ccbb46bd1a4b62beac0ae6306d144189a4f60108ae3` — qualified weighted native tangent response
- `H48_native_49824x48.npy` — `88960cdc43aa4f9e4805cc95586191441a2b0e59d71eacbaf9010608629f1e74` — qualified weighted native tangent response with shape metadata
- `Hbar_native_49824x48.bin` — `7f228faff59c5f0c8599441d8a46ee9b89a0e12a863c2e9db60e82280e1c5cf2` — unweighted native tangent response
- `U_a_active_49536x48.bin` — `d6173fe006577cb7818c3bdf547a09c129b9b4a504957f119a2c45eeda8f6a1c` — 48 ordered native tangent solutions
- `B48_PHI_MANIFEST.csv` — `8434422d5c77c437b66dbc934d8d8198d67635c821b2c7ac306afdefc7545b42` — ordered Phi48 manifest

Machine-readable gate results are in `R6P_FINAL_DECISION.json`; the B48/H48 prefix, derivative, action, and historical-context files provide the detailed evidence chain.

## Final self-check

Authoritative M32 source and H32 hashes remain unchanged. M32 was qualified before the sole M48 production construction. No M48 result was accepted before independent FD qualification. Modes 33–48 were all checked. No Direct-start PCG, pair threshold, K_D scaling, nonlinear compensation or perturbed-geometry re-equilibration, M48 compensation, timing scaling, manuscript edit, or M64 work occurred. The required baseline initialization scope is stated explicitly in `R6P_FINAL_DECISION.json`.
