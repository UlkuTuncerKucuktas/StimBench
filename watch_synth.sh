#!/usr/bin/env bash
# Keep a detached generation run alive: every INTERVAL seconds append a status line to
# OUT/status.txt and, if the generator has died with clips still missing, relaunch it
# (run_synth.sh resumes). Stops when the manifest holds EXPECTED clips or after MAX_RESTARTS.
#   setsid nohup bash watch_synth.sh >/dev/null 2>&1 &
# Env: CONFIG, OUT (as for run_synth.sh), EXPECTED (default 520), INTERVAL (600), MAX_RESTARTS (5)

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=${CONFIG:-configs/synth/wan22_a14b_480p_fast.yaml}
[[ "$CONFIG" = /* ]] || CONFIG="$HERE/$CONFIG"
PY=${PY:-python3}
OUT=${OUT:-$($PY -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output']['root'])" "$CONFIG")}
EXPECTED=${EXPECTED:-520}
INTERVAL=${INTERVAL:-600}
MAX_RESTARTS=${MAX_RESTARTS:-5}
STATUS="$OUT/status.txt"
restarts=0

# unique files, as the manifest reader counts them: a resumed run appends a second
# line for every clip it regenerates
done_count() {
  [[ -f "$OUT/manifest.jsonl" ]] || { echo 0; return; }
  "$PY" -c 'import json,sys;print(len({json.loads(l)["file"] for l in open(sys.argv[1]) if l.strip()}))' \
    "$OUT/manifest.jsonl" 2>/dev/null || echo 0
}
alive() {
  local pid
  pid=$(cat "$OUT/gen.pid" 2>/dev/null) || return 1
  [[ "$(ps -o stat= -p "$pid" 2>/dev/null)" =~ ^[^Z]*$ ]] && ps -o args= -p "$pid" 2>/dev/null | grep -q gen_synth
}

while true; do
  n=$(done_count)
  failed=$(grep -c -E 'FAILED|retime failed' "$OUT/gen.log" 2>/dev/null); failed=${failed:-0}
  last=$(grep -E '^\S+ \S+ INFO +\[' "$OUT/gen.log" 2>/dev/null | tail -1 | sed 's/.*INFO *//')
  if alive; then state=running; else state=stopped; fi
  echo "$(date '+%F %T') $state clips=$n/$EXPECTED failed=$failed restarts=$restarts last: $last" >>"$STATUS"
  if (( n >= EXPECTED )); then
    echo "$(date '+%F %T') complete" >>"$STATUS"
    exit 0
  fi
  if [[ $state == stopped ]]; then
    if (( restarts >= MAX_RESTARTS )); then
      echo "$(date '+%F %T') giving up after $restarts restarts" >>"$STATUS"
      exit 1
    fi
    restarts=$((restarts + 1))
    echo "$(date '+%F %T') relaunching (restart $restarts)" >>"$STATUS"
    CONFIG="$CONFIG" OUT="$OUT" bash "$HERE/run_synth.sh" >>"$STATUS" 2>&1
  fi
  sleep "$INTERVAL"
done
