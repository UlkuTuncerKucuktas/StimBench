#!/usr/bin/env bash
# Detached StimBench-Syn generation. Env: CONFIG, OUT, N, PY. Extra args pass through.
#   bash run_synth.sh
#   N=1 OUT=/storage/pvc-trubaai-1tb/siu/bench CONFIG=configs/synth/wan22_a14b_480p_fast.yaml bash run_synth.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=${CONFIG:-configs/synth/wan22_a14b_480p.yaml}
[[ "$CONFIG" = /* ]] || CONFIG="$HERE/$CONFIG"
PY=${PY:-python3}

OUT=${OUT:-$($PY -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output']['root'])" "$CONFIG")}
EXTRA=()
[[ -n "${N:-}" ]] && EXTRA+=(--n-per-class "$N")

LOG="$OUT/gen.log"
ERR="$OUT/gen.stderr"
PIDFILE="$OUT/gen.pid"
mkdir -p "$OUT"

# a finished setsid child lingers as a zombie, which kill -0 would still count as running
if [[ -f "$PIDFILE" ]] && [[ "$(ps -o stat= -p "$(cat "$PIDFILE")" 2>/dev/null)" =~ ^[^Z]*$ ]] \
   && ps -o args= -p "$(cat "$PIDFILE")" 2>/dev/null | grep -q gen_synth; then
  echo "already running as PID $(cat "$PIDFILE"); tail -f $LOG"
  exit 1
fi

if ! "$PY" "$HERE/gen_synth.py" plan --config "$CONFIG" --out "$OUT" --show 0 --check-tokens "${EXTRA[@]}" "$@" >"$OUT/plan.txt" 2>&1; then
  echo "plan failed its audit or crashed; not starting. Output:"; cat "$OUT/plan.txt"; exit 1
fi

# -u: unbuffered log; setsid: SIGHUP cannot reach it
setsid nohup "$PY" -u "$HERE/gen_synth.py" generate --config "$CONFIG" \
  --out "$OUT" "${EXTRA[@]}" "$@" >>"$ERR" 2>&1 &
PID=$!
echo "$PID" >"$PIDFILE"

sleep 3
if ! kill -0 "$PID" 2>/dev/null; then
  echo "failed to start; last lines of $ERR:"; tail -20 "$ERR"; rm -f "$PIDFILE"; exit 1
fi

cat <<EOF
started  PID $PID
config   $CONFIG
output   $OUT

  tail -f $LOG                        # progress
  grep -E 'FAILED|retime' $LOG        # problems only
  kill $PID                           # stop; rerun to resume
EOF
