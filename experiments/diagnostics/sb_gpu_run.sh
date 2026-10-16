#!/usr/bin/env bash
# sb_gpu_run.sh - FIXED study launcher. Single entry for every GPU job.
# Guards (refuse with reason, never launch blind):
#   known node + gpu index, UUID match, GPU free, arm assigned this phase.
# Usage: SB_ROOT=... SB_NEW=... bash sb_gpu_run.sh <stage> --arm S --dtype bfloat16 --gpu 0
set -u
set -o pipefail
STAGE="${1:?usage: sb_gpu_run.sh <stage> --arm A ...}"; shift
ARM=""; DTYPE="bfloat16"; GPU=""; SHARD=""; RUN=""; LR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --arm) ARM="${2:?}"; shift 2;;
    --dtype) DTYPE="${2:?}"; shift 2;;
    --gpu) GPU="${2:?}"; shift 2;;
    --shard) SHARD="${2:?}"; shift 2;;
    --run) RUN="${2:?}"; shift 2;;
    --lr) LR="${2:?}"; shift 2;;
    *) echo "REFUSE: unknown arg $1"; exit 2;;
  esac
done
[ -n "$GPU" ] || { echo "REFUSE: need --gpu N"; exit 2; }
SB_ROOT="${SB_ROOT:?REFUSE: need SB_ROOT env}"
SB_NEW="${SB_NEW:?REFUSE: need SB_NEW env (study checkout)}"
HOST="$(cat /etc/hostname)"

uuid_for() {
  case "$1:$2" in
    huyhq21-membedding-v3-0-0:0) echo "GPU-84ea467a-795a-a39b-55a5-c8e8536c005a";;
    huyhq21-membedding-v3-0-0:1) echo "GPU-b9dbf2f0-b394-314b-9e5f-f21294afc5ce";;
    huyhq21-membedding-v3-0-0:2) echo "GPU-32563997-e5e4-0387-7824-ff37d3298382";;
    huyhq21-membedding-v3-0-0:3) echo "GPU-eaed83d1-a3de-c74d-4080-cb822cc8c5e4";;
    convert-nemo-rl-4gpus-v100-2608-0-3:0) echo "GPU-ac08a6f4-a8ac-c5da-1a07-5aa9de571897";;
    convert-nemo-rl-4gpus-v100-2608-0-3:1) echo "GPU-d54a9471-92a7-f478-2a3e-c08dd1ce4bff";;
    convert-nemo-rl-4gpus-v100-2608-0-3:2) echo "GPU-0edc5198-6a29-8c63-d702-ca8912c8e16a";;
    convert-nemo-rl-4gpus-v100-2608-0-3:3) echo "GPU-755a3227-7d88-a068-9552-6061514b40e7";;
    huyhq21-membedding-v1-0-0:0) echo "GPU-9679efe6-954f-ae1d-ed1f-1544cc67114e";;
    huyhq21-membedding-v1-0-0:1) echo "GPU-30b4be09-0d9f-2649-9cfa-6a356742a2fe";;
    huyhq21-membedding-v1-0-0:2) echo "GPU-505ae3dd-e8f2-814e-4402-81835980c007";;
    huyhq21-membedding-v1-0-0:3) echo "GPU-52333868-e7f4-2ca4-0737-63189b81ebad";;
    *) echo "";;
  esac
}

# Assigned arm per node:gpu for each stage. Unlisted = refuse.
role_for() {
  case "$1:$2:$3" in
    qualify:huyhq21-membedding-v3-0-0:0) echo "S";;
    qualify:huyhq21-membedding-v3-0-0:3) echo "B";;
    qualify:huyhq21-membedding-v3-0-0:2) echo "Q";;
    qualify:huyhq21-membedding-v1-0-0:0) echo "S";;
    qualify:huyhq21-membedding-v1-0-0:1) echo "B";;
    qualify:huyhq21-membedding-v1-0-0:2) echo "Q";;
    bench:huyhq21-membedding-v3-0-0:1) echo "S";;
    bench:huyhq21-membedding-v3-0-0:3) echo "B";;
    bench:convert-nemo-rl-4gpus-v100-2608-0-3:0) echo "Q";;
    bench:huyhq21-membedding-v1-0-0:3) echo "S";;
    pool:convert-nemo-rl-4gpus-v100-2608-0-3:*) echo "S";;
    audit:convert-nemo-rl-4gpus-v100-2608-0-3:*) echo "S";;
    fixtures:convert-nemo-rl-4gpus-v100-2608-0-3:*) echo "S";;
    propose:convert-nemo-rl-4gpus-v100-2608-0-3:*) echo "S";;
    propose:huyhq21-membedding-v3-0-0:*) echo "B";;
    propose:huyhq21-membedding-v1-0-0:*) echo "Q";;
    score:convert-nemo-rl-4gpus-v100-2608-0-3:*) echo "S";;
    score:huyhq21-membedding-v3-0-0:*) echo "B";;
    score:huyhq21-membedding-v1-0-0:*) echo "Q";;
    bridge:huyhq21-membedding-v3-0-0:3) echo "B";;
    train:huyhq21-membedding-v3-0-0:*) echo "S";;
    *) echo "";;
  esac
}

# Pool shards per gpu (10 shards over 4 cards; same split on all nodes).
pool_shards_for() {
  case "$1:$2" in
    convert-nemo-rl-4gpus-v100-2608-0-3:0|huyhq21-membedding-v3-0-0:0|huyhq21-membedding-v1-0-0:0) echo "00 01";;
    convert-nemo-rl-4gpus-v100-2608-0-3:1|huyhq21-membedding-v3-0-0:1|huyhq21-membedding-v1-0-0:1) echo "02 03";;
    convert-nemo-rl-4gpus-v100-2608-0-3:2|huyhq21-membedding-v3-0-0:2|huyhq21-membedding-v1-0-0:2) echo "04 05 06";;
    convert-nemo-rl-4gpus-v100-2608-0-3:3|huyhq21-membedding-v3-0-0:3|huyhq21-membedding-v1-0-0:3) echo "07 08 09";;
    *) echo "";;
  esac
}

EXP="$(uuid_for "$HOST" "$GPU")"
[ -n "$EXP" ] || { echo "REFUSE: unknown node/gpu $HOST:$GPU (wrong terminal?)"; exit 3; }
GOT="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")"
[ "$GOT" = "$EXP" ] || { echo "REFUSE: uuid mismatch got=$GOT exp=$EXP"; exit 3; }
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$EXP"; then
  echo "REFUSE: gpu $GPU ($HOST) busy"
  exit 4
fi
WANT="$(role_for "$STAGE" "$HOST" "$GPU")"
[ -n "$WANT" ] || { echo "REFUSE: $HOST gpu$GPU not assigned for stage $STAGE"; exit 5; }
[ "$ARM" = "$WANT" ] || { echo "REFUSE: $HOST gpu$GPU runs $WANT this phase, not $ARM"; exit 5; }
if [ "$STAGE" = "pool" ] || [ "$STAGE" = "audit" ] || [ "$STAGE" = "fixtures" ] \
    || [ "$STAGE" = "propose" ] || [ "$STAGE" = "score" ]; then
  [ -n "$SHARD" ] || { echo "REFUSE: $STAGE needs --shard NN"; exit 2; }
  case " $(pool_shards_for "$HOST" "$GPU") " in
    *" $SHARD "*) ;;
    *) echo "REFUSE: shard $SHARD not assigned to $HOST gpu$GPU"; exit 5;;
  esac
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$SB_NEW/experiments/diagnostics:$SB_NEW/experiments/mp_opd"
case "$STAGE" in
  qualify)
    case "$DTYPE" in bfloat16|float32) ;;
      *) echo "REFUSE: bad dtype $DTYPE"; exit 2;;
    esac
    exec /opt/venv/bin/python "$SB_NEW/experiments/diagnostics/sb_qualify.py" \
      --arm "$ARM" --dtype "$DTYPE"
    ;;
  bench)
    case "$DTYPE" in bfloat16|float32) ;;
      *) echo "REFUSE: bad dtype $DTYPE"; exit 2;;
    esac
    exec /opt/venv/bin/python "$SB_NEW/experiments/diagnostics/sb_bench.py" \
      --arm "$ARM" --dtype "$DTYPE"
    ;;
  pool)
    [ "$ARM" = "S" ] || { echo "REFUSE: pool is student-only"; exit 2; }
    exec /opt/venv/bin/python "$SB_NEW/experiments/diagnostics/sb_pool.py" \
      --shard "$SHARD"
    ;;
  audit)
    # CPU-only (tokenizers); GPU guard still applies for placement discipline.
    exec /opt/venv/bin/python "$SB_NEW/experiments/diagnostics/sb_audit.py" \
      --shard "$SHARD"
    ;;
  fixtures)
    # CPU-only; reads pool+audit of the same shard.
    exec /opt/venv/bin/python "$SB_NEW/experiments/diagnostics/sb_fixtures.py" \
      --shard "$SHARD"
    ;;
  propose)
    POS="$SB_ROOT/work/positions.json"
    [ -f "$POS" ] || { echo "REFUSE: $POS missing; run P-positions first"; exit 6; }
    exec /opt/venv/bin/python "$SB_NEW/experiments/diagnostics/sb_propose.py" \
      --shard "$SHARD" --arm "$ARM" --positions-file "$POS"
    ;;
  score)
    POS="$SB_ROOT/work/positions.json"
    [ -f "$POS" ] || { echo "REFUSE: $POS missing; run P-positions first"; exit 6; }
    exec /opt/venv/bin/python "$SB_NEW/experiments/diagnostics/sb_score.py" \
      --shard "$SHARD" --arm "$ARM" --positions-file "$POS"
    ;;
  bridge)
    [ "$ARM" = "B" ] || { echo "REFUSE: bridge is official-B"; exit 2; }
    exec /opt/venv/bin/python "$SB_NEW/experiments/diagnostics/sb_bridge.py" \
      --n 200
    ;;
  train)
    [ "$ARM" = "S" ] || { echo "REFUSE: train student is S"; exit 2; }
    [ -n "$RUN" ] || { echo "REFUSE: train needs --run ID"; exit 2; }
    if [ -n "$LR" ]; then
      exec /opt/venv/bin/python "$SB_NEW/experiments/diagnostics/sb_train.py" \
        --run "$RUN" --lr "$LR"
    else
      exec /opt/venv/bin/python "$SB_NEW/experiments/diagnostics/sb_train.py" \
        --run "$RUN"
    fi
    ;;
  *) echo "REFUSE: unknown stage $STAGE"; exit 2;;
esac
