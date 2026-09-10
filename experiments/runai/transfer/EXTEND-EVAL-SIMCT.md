# Extend an active company eval with SimCT

Run extend-eval-simct.py using system Python 3.12. Pass --source pointing
to the unchanged checkout that generated --plan, --out a new sibling
directory, --simct the completed SimCT checkpoint root, and --worker-pids
the currently observed two Python worker PIDs.

The tool validates the 17-job plan, fixed new MP run names, source and data
hashes, SimCT completion/SFT initialization, and hashes eight SimCT checkpoints
before stopping the exact workers via Linux pidfds. It sends TERM to their
snapshotted descendants and workers, waits up to 30 seconds, then requires
GPU and job locks to be free. It never kills unrelated processes or launches GPUs.
An interrupted request may be regenerated. Original journals remain untouched.

A new 25-job plan copies journals; validates request, score and completed-metric
hashes; changes only their plan hash bindings; retains an audit receipt and
original deadline. Partial trailing writes can be repaired in the copy only.
Any validation failure leaves the output marked incomplete; do not launch it.
Only status=verified in migration.json permits launching worker --concurrency 16.
Use the ORIGINAL source checkout for new workers, because source hashes and
server identities must remain unchanged. Do not use recover-startup.

Tests: synthetic one-cell migration verifies 25 jobs, retained deadline, exact
response bytes and unchanged original cells; unrelated PID is rejected.
These tests do not simulate production SGLang teardown or establish numerical
identity across concurrency settings. Per-item seeds/decoding parameters stay
fixed, but batch scheduling can affect floating-point execution. Record the
concurrency switch in the migration receipt. No guarantee that all jobs fit
the remaining original deadline.

## Separate generation and grading bounds

The updated worker has --score-buffer (default 64, range concurrency through
256). Generation futures are capped independently at --concurrency; waiting
items plus in-flight generation are capped at score-buffer. Active scorers
are capped at --score-workers (default 2). Waiting entries hold item references;
LCB test decoding happens inside active scorer threads, not in queued payloads.
Completed responses continue to be journaled before grading. A sustained grading
bottleneck eventually fills the buffer and applies backpressure; the buffer
cannot eliminate a CPU throughput bottleneck. The loaded dataset and response
dictionaries remain in memory as before, so this is not a total RAM cap.

Deployment requires an audited source/plan migration: existing plan source hashes
include eval_queue.py. Do not overwrite a running checkout or bypass source hash
checks. Use the updated helper with --queue-update for a 25-job plan. This mode
requires exactly the audited old and new queue SHA256 values, copies the source
into the new output, updates only eval_queue.py, and binds copied cell contracts
to the new plan. The default mode still extends 17 jobs to 25.

Regression: a blocked two-scorer test allows eight buffered/in-flight items plus
two active scorer items, limits decoded payloads to two, and completes all items
after release. Existing resume tests verify completed cells are not regenerated.

## Pipeline update deployment

For the 25-job queue, pass --queue-update with the verified new eval_queue.py
and omit --worker-pids to discover workers by exact --plan argument. The helper
revalidates executable script/plan identity before signaling. It records new
source, concurrency 16, score_workers 8, score_buffer 64 in migration.json.
Only launch after status=verified, using that receipt's source and new plan.
The source copy keeps runtime launch scripts; no Python dependencies are installed.
Old plan/source/journals remain unchanged. Per-cell timing may omit the last
interrupted segment when TERM bypasses Python cleanup; do not use migrated timing
as an exact end-to-end throughput measurement. Scorer timeouts/limits and decoding
parameters remain unchanged; execution ordering can still affect floating point.
