# Kaggle TPU locked environment

This uv project locks the Python userspace used by the SimCT Kaggle notebooks.
It is intentionally separate from the repository's developer environment.

- `uv.lock` is the executable userspace source of truth.
- `provider-constraints.json` records the exact Kaggle-owned Python, JAX,
  JAXLIB, and TPU topology contract.
- JAX, JAXLIB, and libtpu remain in the uv resolution graph for compatibility,
  but the notebook export excludes them and inherits them from Kaggle through a
  `--system-site-packages` virtual environment.
- Every other resolved dependency is installed from the lock without a second
  dependency solve.

Check the lock without changing it:

```bash
uv lock --check --project environments/kaggle-tpu
```

After an intentional dependency edit, regenerate it with:

```bash
uv lock --project environments/kaggle-tpu --python 3.12
```

Do not edit the provider constraints merely to make a new Kaggle image pass.
First verify that the new Python/JAX/JAXLIB combination still provides eight
TPU devices and passes the real training canary.
