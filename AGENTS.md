# Research project workflow

- Keep this existing project on Windows unless a separate, validated Linux workflow is requested.
- Store project files, downloads, environments, caches, and generated outputs on D: whenever possible. Do not relocate this active project merely to satisfy that preference.
- Before a major change, check whether Git is healthy. Use Git for meaningful checkpoints rather than every small edit.
- Once a usable baseline exists, initialize Git if it is absent. At major checkpoints, review, commit, and push automatically; pause only when licensing, sensitive content, repository ownership, or unusually large files need a user decision.
- This repository uses an opt-in baseline for the large legacy tree. When a previously untracked scientific module receives a major change, review that module and deliberately add its relevant source, documentation, and compact evidence while continuing to exclude generated outputs and bundled dependencies.
- Create checkpoints at a verified baseline, before a risky refactor, after a major validated result, and at manuscript or release milestones.
- Before committing, review the diff and exclude secrets, virtual environments, caches, build products, third-party toolchains, raw datasets, and large generated solver outputs unless they are deliberately tracked with Git LFS.
- Push milestone commits to the user's GitHub account (`danielye0010`). Public repositories are acceptable after checking licenses, large files, and sensitive content.
- Never force-push or rewrite published history unless the user explicitly requests it.
- For a new project, discuss and recommend Windows or WSL2/Linux before implementation. Prefer WSL2 for Python/JAX, GCC/gfortran, CMake, CalculiX source builds, batch computation, and reproducible scientific environments. Prefer Windows for Unity, Windows-only executables, and GUI-heavy Office workflows.
