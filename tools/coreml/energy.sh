#!/usr/bin/env bash
# Energy of one workload on an Apple Silicon Mac: powermetrics' CPU, GPU and ANE rails while it runs.
#
#   sudo -v                                  # once, in the same terminal: the ticket belongs to its tty
#   tools/coreml/energy.sh NAME --ready FILE -- COMMAND [ARGS...]
#   tools/coreml/energy.sh NAME --skip SECONDS -- COMMAND [ARGS...]
#
# The workload starts first; powermetrics starts once it is warm: when COMMAND creates FILE (predict_bench does,
# with READY_FILE=FILE in its environment) or after SECONDS (a model load, for a command that cannot say so).
# powermetrics samples once a second until COMMAND exits. Prints the workload's own output (its throughput) and the
# mean power of each rail in mW over the samples, the first and last of which are dropped. millijoules per token is
# the rails together, in mW, divided by the tokens per second the workload reports: mW / (tokens/s) = mJ/token.
# NAME "idle" with `--skip 0 -- sleep 30` gives the baseline. The display, the SSD and the charger are not in these
# rails. Raw samples go to energy-NAME.txt, the workload's output to energy-NAME.out in the current directory.
set -euo pipefail
name=$1
shift
ready="" skip=0
while [[ ${1:-} != -- ]]; do
	case ${1:-} in
	--ready) ready=$2; shift 2 ;;
	--skip) skip=$2; shift 2 ;;
	*) echo "usage: energy.sh NAME [--ready FILE | --skip SECONDS] -- COMMAND [ARGS...]" >&2; exit 2 ;;
	esac
done
shift
sudo -n true 2>/dev/null || { echo "sudo has no valid ticket: run \`sudo -v\` in this terminal first" >&2; exit 1; }
[[ -n $ready ]] && rm -f "$ready"
out=energy-$name.out raw=energy-$name.txt
READY_FILE=$ready "$@" >"$out" 2>&1 &
wl=$!
if [[ -n $ready ]]; then
	until [[ -e $ready ]] || ! kill -0 "$wl" 2>/dev/null; do sleep 0.2; done
else
	sleep "$skip"
fi
kill -0 "$wl" 2>/dev/null || { echo "the workload ended before the measurement began:" >&2; cat "$out" >&2; exit 1; }
sudo -n powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 >"$raw" 2>/dev/null &
pm=$!
wait "$wl" || true
sleep 1
sudo -n pkill -x powermetrics 2>/dev/null || true
wait "$pm" 2>/dev/null || true
cat "$out"
awk -v name="$name" '
/^CPU Power:/ { cpu[nc++] = $3 }
/^GPU Power:/ { gpu[ng++] = $3 }
/^ANE Power:/ { ane[na++] = $3 }
function mean(a, n,   i, s, c) { for (i = 1; i < n - 1; i++) { s += a[i]; c++ } return c ? s / c : 0 }
END {
	c = mean(cpu, nc); g = mean(gpu, ng); a = mean(ane, na)
	printf "%s: CPU %.0f mW, GPU %.0f mW, ANE %.0f mW, together %.0f mW (%d samples)\n", name, c, g, a, c + g + a, nc - 2
}' "$raw"
