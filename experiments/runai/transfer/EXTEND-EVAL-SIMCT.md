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
