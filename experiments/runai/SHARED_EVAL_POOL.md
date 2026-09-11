# Shared evaluation pool

A pool is one immutable `plan.json` plus its shared directory of state, cells
and journals. Workers on any authorized node mount the same directory and use
the exact source hashes, model/data identities and compatible grading runtime.
Node-local job managers own local GPUs/processes; they must not share SQLite
databases across nodes. Host resource deadlines remain enforced locally.

Generation workers claim checkpoints via a shared exclusive checkpoint lock.
Each GPU defaults to 256 concurrent generation requests and writes durable
responses plus a hash-checked `generation-complete.json` marker per cell.
They never wait for scoring or acquire a GPU from another node.

CPU scoring workers may run on any node, including nodes with no GPU. Each
claims a ready `(checkpoint, benchmark, eval seed)` cell with `score.lock`,
rechecks completion inside the lock, validates the spool contract and scores
only that cell. Multiple scoring workers can process different cells in
parallel. `--score-workers` controls CPU concurrency inside each worker;
provision CPU slots for the sum across workers. Scaling is limited by the
number of ready cells, not individual examples within one cell.

Examples with the same PLAN and pinned source on each node:

```bash
# Generation on an explicitly allocated local GPU; default concurrency is 256.
python3 scripts/evaluation/eval_queue.py worker --plan "$PLAN" --gpu 2 --internal-code-execution
# Scoring elsewhere; no GPU required.
python3 scripts/evaluation/eval_queue.py score-spool --plan "$PLAN" --score-workers 4 --internal-code-execution
```

The shared filesystem must honor cross-node `flock`, already tested on the
current allocation. A shared `scoring.lock` lets new workers coexist while
excluding the historical single exclusive scorer. Worker exit releases claims;
after a crash another worker resumes durable response/score journals. A
scoring error is persisted per cell as `score-error.json`, never silently
retried by every node. After diagnosing it, `retry-scoring --plan "$PLAN"`
archives error markers under cell locks without removing scored examples.

Do not mix combined workers with separated scorers. For existing runs, stop
and verify writers, preserve their artifacts, and use an audited migration
to update the plan/source contract before launching this version. Changing
defaults does not change existing processes or pinned plans. This protocol
does not claim multi-node GPU runtime verification from local tests.
