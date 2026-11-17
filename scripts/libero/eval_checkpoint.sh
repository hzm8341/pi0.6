#!/usr/bin/env bash
# Evaluate one checkpoint on LIBERO-10 Task 0: start the policy server (repo .venv), run the Docker client, stop the server.
# Usage: scripts/libero/eval_checkpoint.sh <config> <checkpoint_dir> <tag> [extra serve_policy args, e.g. --cfg-beta 1.5]
# Env: NUM_TRIALS (default 50), EPISODE_START (0), RECORD_DIR (unset = no recording), PORT (8000).
set -euo pipefail
cd "$(dirname "$0")/../.."
CONFIG=$1; CKPT=$2; TAG=$3; shift 3
PORT=${PORT:-8000}; NUM_TRIALS=${NUM_TRIALS:-50}; EPISODE_START=${EPISODE_START:-0}
mkdir -p logs data/libero/eval
.venv/bin/python scripts/serve_policy.py --port "$PORT" "$@" policy:checkpoint \
    --policy.config "$CONFIG" --policy.dir "$CKPT" > "logs/serve_${TAG}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; do
    kill -0 $SERVER 2>/dev/null || { echo "server died, see logs/serve_${TAG}.log"; exit 1; }
    sleep 5
done
CLIENT_ARGS="--args.port $PORT --args.task-suite-name libero_10 --args.task-ids 0 --args.num-trials-per-task $NUM_TRIALS \
 --args.episode-start $EPISODE_START --args.summary-path data/libero/eval/${TAG}.json --args.video-out-path data/libero/videos/${TAG}"
if [[ -n "${RECORD_DIR:-}" ]]; then CLIENT_ARGS="$CLIENT_ARGS --args.record-dir $RECORD_DIR --args.no-save-videos"; fi
docker run --rm --gpus all --network host -v "$PWD":/app -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
    -e MUJOCO_EGL_DEVICE_ID=0 -e NVIDIA_DRIVER_CAPABILITIES=all -e CLIENT_ARGS="$CLIENT_ARGS" libero \
    > "logs/client_${TAG}.log" 2>&1
python3 -c "import json;d=json.load(open('data/libero/eval/${TAG}.json'));print('${TAG}', d['total_successes'],'/',d['total_episodes'],'=',d['success_rate'])"
