# VDT dynamic-span research workspace

This branch keeps the upstream SimCT implementation pinned at
`cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e` and adds a framework-independent
reference layer in `vdt_span/`.

The separation is deliberate:

- `kdflow/` is the upstream PyTorch/SGLang implementation.
- `vdt_span/` defines small byte-alignment, scoring, and segmentation contracts.
- a Torch or Tunix adapter must prove parity against those contracts before a
  training result is interpreted scientifically.

## CPU setup

```bash
make study-env
make test-core
make reference-canary
```

The study environment contains only `pytest` and NumPy. It does not install the
heavy GPU stack from the upstream `requirements.txt`. Because the WSL root was
nearly full during setup, `.venv-study` is a repository-local symlink to the
actual environment under `/mnt/d/dev/codex/vdt-dynamic-span/.venv-study`.
The canary writes a small, deterministic JSON result to
`artifacts/reference_canary.json`.

## Current claim boundary

Passing the core tests proves only that the deterministic reference primitives
obey their stated contracts. It does not prove parity with all upstream SimCT
training behavior, TPU model support, a successful accelerator run, or a
downstream accuracy improvement.

## First experimental ladder

1. Paper-faithful static SimCT on fixed logits and fixed rollouts.
2. Current-public-code SimCT at the pinned commit.
3. Fixed coarsening at matched compute.
4. Transparent adaptive boundary policy.
5. Learned semi-Markov energy only if an oracle/dynamic gap is measurable.
