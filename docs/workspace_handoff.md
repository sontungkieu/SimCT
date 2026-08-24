# Workspace handoff

## Checkouts

| Purpose | Path | Branch | Starting revision |
|---|---|---|---|
| Reference implementation and TPU port | `/home/tung/vdt-dynamic-span` | `vdt/implementation` | `cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e` |
| Student learning lab | `/home/tung/vdt-dynamic-span-learning` | `vdt/learning` | `cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e` |

Both worktrees track the same pinned upstream SimCT repository but intentionally
contain different uncommitted changes. No commit or push has been made.

## Lightweight local checks

```bash
cd /home/tung/vdt-dynamic-span
make study-env
make test-all
make reference-canary
```

The learning lab is independent of the production checkout:

```bash
cd /home/tung/vdt-dynamic-span-learning
python3 scripts/grade_learning.py --smoke
python3 scripts/grade_learning.py
```

The full grader starts at `0/100` by design because the six exercise functions
are genuine TODOs. The smoke command must remain green throughout the exercise.

## Storage boundary

The WSL root filesystem had less than 100 MB free during setup. Keep model
weights, converted checkpoints, compiler caches, Kaggle packages, and run
outputs under `/mnt/d/dev/codex/vdt-dynamic-span/` or on Kaggle storage. The
small source checkout remains under `/home/tung`; `.venv-study` in that checkout
is a symlink to `/mnt/d/dev/codex/vdt-dynamic-span/.venv-study` so NumPy does not
consume the remaining WSL-root space.

Do not point recursive cleanup commands at either worktree. `artifacts/` and
`runs/` are ignored local outputs, not evidence that a remote job succeeded.

## User-visible Codex tasks

- `01a03288-5193-7c90-8c2b-152b0cf54028`: SimCT baseline parity audit.
- `01a03288-5dbd-7b93-b6bd-c5398b82a7ee`: Tunix/MaxText TPU scaffold.
- `01a03288-6a41-73f3-a952-193d53f09442`: dynamic-span learning lab.
- `01a03297-4a1b-7291-87e3-3714f66475cf`: real Tunix backend and JAX parity.
- `01a032a7-2283-72a3-b5e5-8d834be74171`: fail-closed Kaggle v5e-8 dry-run package.

Operational states such as scaffolded, queued, or hardware-detected must not be
reported as a reproduced scientific baseline. A baseline result additionally
requires model provenance, actual optimizer updates, metrics, and audited run
artifacts.
