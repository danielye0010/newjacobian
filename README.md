# Response-Jacobian Compensation Experiments

Research code for testing response-aware geometric compensation in synthetic, plate, FEM-like, and differentiable JAX models. The experiments compare geometry-only updates with scalar, diagonal, full-Jacobian, and trust-region response corrections, with particular attention to fixed-boundary admissibility and whether numerical evidence supports each method claim.

## Scope

This repository contains controlled numerical experiments and lightweight surrogate models. The fixed-bottom voxel-spring and plate models are designed for response-Jacobian diagnostics; they are not calibrated additive-manufacturing process models.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Run

Run commands from the repository root. A small synthetic diagnostic is the quickest starting point:

```bash
python experiments/response_jacobian_step1_synthetic/run_step1_synthetic.py
```

The differentiable fixed-bottom benchmark is available through:

```bash
python experiments/fixed_bottom_jax_fem/run_fixed_bottom_jax_fem.py
```

Generated CSV, JSON, Markdown, and figure artifacts are written under `results/` and are intentionally excluded from version control.

## Experiment layout

| Module | Purpose |
|---|---|
| `response_jacobian_step1_synthetic` | Controlled low-dimensional response and solver checks |
| `response_jacobian_step2_plate_modal` | Fixed-connectivity modal geometry experiment |
| `response_jacobian_step3_physics_plate` | Geometry-dependent plate response |
| `response_jacobian_step3b_coupling_stress` | Stronger response-coupling stress tests |
| `response_jacobian_step4_fem_plate` | Lightweight thermal FEM-like plate model |
| `response_jacobian_step4b_fem_plate_audit` | Linearization, step-size, and case-selection checks |
| `step6_jax_differentiable_fem_response_jacobian` | JAX-differentiable FEM response benchmark |
| `fixed_bottom_jax_fem` | Fixed-boundary admissibility and response-inversion studies |

Each module includes its own README with its entry point and modeling boundary.

## Reproducibility boundary

The repository tracks source code, dependency versions, and concise run instructions. Solver outputs, caches, paper source, manuscript drafts, and large generated evidence packages remain local.

## License

No reuse license has been declared yet. Contact the repository owner before redistributing or incorporating the code elsewhere.
