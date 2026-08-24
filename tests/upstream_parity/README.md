# Upstream SimCT parity fixtures

These tests describe the pinned public SimCT snapshot at
`cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e` and contrast it with small
paper-reference calculations.

They are intentionally CPU-light and do not import KDFlow, PyTorch, model
weights, tokenizers, or GPU runtimes. A passing test means the audited source
still matches the documented contract; it does **not** mean the paper's results
have been reproduced.

See `docs/reproduction/simct_paper_code_contract.md` for evidence, caveats, and
the distinction among paper-math, pre-safeguard public code, and current public
defaults.
