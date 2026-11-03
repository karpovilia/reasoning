#!/bin/bash
# Run a command so that it is GONE by a hard deadline (PTREV-005: STOP_AT is the latest release time of the GPU).
# Usage: deadline_run.sh DEADLINE_EPOCH TERM_LEAD_SECONDS KILL_MARGIN_SECONDS -- cmd args...
#   at DEADLINE - TERM_LEAD   : SIGTERM to the command's process group (the runner checkpoints and exits on TERM)
#   at DEADLINE - KILL_MARGIN : SIGKILL to the whole group, independent of the command's cooperation
# Polling every 0.2 s; KILL_MARGIN must cover the poll delay and process teardown (>= 2 s).
# Exit: the command's code; 124 if it ended after the deadline TERM; 137 if it had to be killed.
set -u
DEADLINE=$1; LEAD=$2; MARGIN=$3; shift 3; [ "$1" = "--" ] && shift
[ "$MARGIN" -ge 2 ] && [ "$LEAD" -gt "$MARGIN" ] || { echo "BAD_DEADLINE_ARGS lead=$LEAD margin=$MARGIN" >&2; exit 2; }
TERM_AT=$((DEADLINE - LEAD)); KILL_AT=$((DEADLINE - MARGIN))
setsid "$@" &
pid=$!
state=0
while kill -0 $pid 2>/dev/null; do
  now=$(date +%s)
  if [ $state -eq 0 ] && [ "$now" -ge "$TERM_AT" ]; then
    echo "DEADLINE_TERM $(date -Is) pid=$pid deadline=$DEADLINE" >&2; kill -TERM -- -$pid 2>/dev/null; state=1
  fi
  if [ $state -le 1 ] && [ "$now" -ge "$KILL_AT" ]; then
    echo "DEADLINE_KILL $(date -Is) pid=$pid deadline=$DEADLINE" >&2; kill -KILL -- -$pid 2>/dev/null; state=2
  fi
  sleep 0.2
done
wait $pid; rc=$?
kill -KILL -- -$pid 2>/dev/null   # any leftover member of the group
echo "DEADLINE_DONE $(date -Is) rc=$rc state=$state" >&2
[ $state -eq 1 ] && exit 124
[ $state -eq 2 ] && exit 137
exit $rc
