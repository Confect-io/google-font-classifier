#!/usr/bin/env bash
# Status of the current Vast.ai training run.
#
# Finds the running instance itself, so there is no host/port to remember —
# they change on every launch.
#
#   ./check_training.sh          one-shot status
#   ./check_training.sh log      live tail of the remote training log (ctrl-C to stop)
#   ./check_training.sh ssh      open a shell on the box
#   ./check_training.sh hf       what has reached HuggingFace so far
#
# Safe to run from anywhere, any time, including after a reboot. Training does
# not depend on this laptop: it is nohup-ed on the box and the box destroys
# itself when finished.

set -uo pipefail
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
RESULTS_REPO="${RESULTS_REPO:-confect/google-font-classifier-v6}"

read -r ID HOST PORT RATE < <(vastai show instances --raw 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
r=[i for i in d if i.get('actual_status')=='running']
print(*( (r[0]['id'], r[0]['ssh_host'], r[0]['ssh_port'], r[0].get('dph_total') or 0) if r else ('','','','') ))
")

if [ -z "${ID:-}" ]; then
    echo "No running instance."
    echo "If the run finished, results are at https://huggingface.co/$RESULTS_REPO"
    exit 0
fi

RSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=25 -p "$PORT" "root@$HOST")

case "${1:-status}" in
  ssh)  exec "${RSH[@]}" ;;
  log)  exec "${RSH[@]}" "tail -f /workspace/training.log" ;;
  hf)
    uv run --quiet --with huggingface_hub python3 - "$RESULTS_REPO" <<'PY'
import sys
from huggingface_hub import HfApi
api = HfApi(); repo = sys.argv[1]
try:
    fs = api.list_repo_files(repo, repo_type="model")
except Exception as e:
    print("repo not created yet:", str(e)[:80]); raise SystemExit
ck = sorted({int(f.split("/")[1].split("-")[1]) for f in fs if "/checkpoint-" in f})
print("checkpoints synced:", ck[-5:] if ck else "none yet")
print("result_model present:", any("result_model" in f for f in fs))
print("logs:", [f for f in fs if f.startswith("logs/")])
PY
    ;;
  *)
    echo "instance $ID  @ $HOST:$PORT  \$$RATE/hr"
    "${RSH[@]}" bash -s <<'REMOTE' 2>/dev/null | tr -d '\r'
pgrep -f run_training.sh >/dev/null && echo "job:  RUNNING" || echo "job:  NOT RUNNING"
echo "step: $(tail -c 4000 /workspace/training.log | tr '\r' '\n' | grep -aoE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+' | tail -1)"
echo "loss: $(grep -ao "{'loss'[^}]*}" /workspace/training.log | tail -1)"
echo "eval: $(grep -ao "{'eval_loss'[^}]*}" /workspace/training.log | tail -1)"
echo "ckpt: $(ls -d /workspace/output/*/checkpoint-* 2>/dev/null | tail -1)"
echo "gpu:  $(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader)"
grep -aE 'Traceback|FAILED:' /workspace/training.log | tail -2
REMOTE
    ;;
esac
