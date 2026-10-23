#!/usr/bin/env bash
# The checks of docs/OPTIMIZE.md section 7 that need a particular machine: a
# hard disk, several NUMA nodes, the installer on Linux. No root for the
# measurements; the installer part (-a) needs root or sudo.
#
#   scripts/check-optimize.sh -b ./nanollm -c ./nanollm.yaml            # measurements
#   scripts/check-optimize.sh -b ./nanollm -c ./nanollm.yaml -a nanollm-VERSION-linux-amd64.tar.gz
#
#   -b FILE   the nanollm binary built from the branch
#   -c FILE   a configuration that points at the libraries and the models
#             (backends.lib_dir, registry.models_dir, a data_dir you may write to)
#   -o DIR    where the results go (default ./check-optimize)
#   -a FILE   also run install.sh --archive FILE into /opt/nanollm-check as the
#             service nanollm-check on 127.0.0.1:18092, measure through it, and
#             remove it again
#   -m DIR    the models to copy for -a (default: registry.models_dir is not read;
#             /var/lib/nanollm/models if it exists)
#   -s        skip the dry runs and the measurements (with -a: only the installer)
#
# Send the output directory back. Nothing outside it changes, except what -a
# installs and removes.
set -u

BIN=./nanollm CFG=./nanollm.yaml OUT=./check-optimize ARCHIVE="" MODELS=/var/lib/nanollm/models SKIP=0
while getopts "b:c:o:a:m:sh" opt; do
  case "$opt" in
    b) BIN=$OPTARG ;; c) CFG=$OPTARG ;; o) OUT=$OPTARG ;; a) ARCHIVE=$OPTARG ;; m) MODELS=$OPTARG ;; s) SKIP=1 ;;
    *) sed -n '2,24p' "$0"; exit 2 ;;
  esac
done
HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ $SKIP -eq 0 ]]; then
  [[ -x "$BIN" ]] || { echo "check-optimize: $BIN is not an executable (-b)" >&2; exit 2; }
  [[ -f "$CFG" ]] || { echo "check-optimize: $CFG is not a file (-c)" >&2; exit 2; }
fi
mkdir -p "$OUT"
FAILED=0
say()  { printf '==> %s\n' "$*"; }
pass() { printf '    PASS  %s\n' "$*"; }
fail() { printf '    FAIL  %s\n' "$*"; FAILED=$((FAILED + 1)); }
info() { printf '    %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# jq-less access to the profile: python3 when there is one.
pyval() { # file, python expression over p
  have python3 || { echo ""; return; }
  python3 - "$1" "$2" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
p = d.get("profile", d)
print(eval(sys.argv[2]))
PY
}

say "the machine"
{
  uname -srm
  have lscpu && lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|NUMA"
  have numactl && numactl --hardware | head -12
  have lsblk && lsblk -d -o NAME,ROTA,TYPE,SIZE,MODEL
  have free && free -m | head -2
  echo "user: $(id -un) ($(id -u))"
} 2>&1 | tee "$OUT/machine.txt" | sed 's/^/    /'
NUMA_NODES=1
if have lscpu; then
  NUMA_NODES=$(lscpu | awk -F: '/NUMA node\(s\)/ {gsub(/ /, "", $2); print $2}')
  NUMA_NODES=${NUMA_NODES:-1}
fi

if [[ $SKIP -eq 0 ]]; then
say "dry runs (nothing is measured)"
for m in silent optimal powerful; do
  start=$(date +%s.%N)
  "$BIN" optimize -config "$CFG" --mode "$m" --dry-run >"$OUT/dryrun-$m.txt" 2>&1
  rc=$?
  end=$(date +%s.%N)
  secs=$(awk -v a="$start" -v b="$end" 'BEGIN { printf "%.1f", b - a }')
  if [[ $rc -eq 0 ]]; then pass "$m: $secs s"; else fail "$m exited with $rc (see $OUT/dryrun-$m.txt)"; fi
  awk -v s="$secs" 'BEGIN { exit !(s <= 10) }' || fail "$m took $secs s; the survey should fit 10 s"
done
if grep -qE "on hdd" "$OUT/dryrun-powerful.txt"; then
  info "the model is on a hard disk"
  grep -q "slow_disk" "$OUT/dryrun-powerful.txt" && pass "slow_disk is reported" || fail "a hard disk but no slow_disk problem"
else
  info "the model is not on a hard disk (the slow-disk checks do not apply)"
fi
if [[ "$NUMA_NODES" -gt 1 ]]; then
  grep -q "#interleave" "$OUT/dryrun-powerful.txt" && grep -q "#distribute" "$OUT/dryrun-powerful.txt" &&
    pass "both NUMA policies are candidates ($NUMA_NODES nodes)" || fail "$NUMA_NODES NUMA nodes but not both policies in the powerful dry run"
else
  info "one NUMA node: no policy to choose"
fi

say "measurements"
for m in silent optimal powerful; do
  "$BIN" optimize -config "$CFG" --mode "$m" --json >"$OUT/$m.json" 2>"$OUT/$m.err"
  rc=$?
  if [[ $rc -ne 0 ]]; then fail "$m exited with $rc (see $OUT/$m.err)"; continue; fi
  dur=$(pyval "$OUT/$m.json" 'round(p["duration_s"], 1)')
  dev=$(pyval "$OUT/$m.json" 'p["chosen"]["device"]["value"]')
  preset=$(pyval "$OUT/$m.json" 'p["chosen"]["preset"]["value"]')
  pmode=$(pyval "$OUT/$m.json" 'p["chosen"]["plan_mode"]["value"]')
  held=$(pyval "$OUT/$m.json" 'round(p["memory"]["measured_bytes"] / 2**30, 2)')
  numa=$(pyval "$OUT/$m.json" 'p["chosen"].get("numa", {}).get("value", "")')
  probs=$(pyval "$OUT/$m.json" '",".join(x["code"] for x in p.get("problems", []))')
  info "$m: ${dur} s, $dev $preset $pmode, held ${held} GiB, numa '${numa}', problems: ${probs}"
  awk -v s="${dur:-0}" 'BEGIN { exit !(s <= 120) }' && pass "$m is within the 2 minutes" || fail "$m took ${dur} s"
  case "$m" in
    silent)
      [[ "$dev" == *cpu* ]] && pass "the quiet mode runs on the CPU" || fail "the quiet mode chose $dev"
      if grep -qE "on hdd" "$OUT/dryrun-powerful.txt"; then
        case "$pmode" in
          *M8*|*M9*|*hybrid*|*mmap*) fail "on a hard disk the quiet mode chose a streaming mode: $preset $pmode" ;;
          *) pass "on a hard disk the quiet mode did not stream ($preset $pmode)" ;;
        esac
      fi ;;
    optimal)
      awk -v h="${held:-99}" 'BEGIN { exit !(h <= 4.2) }' && pass "optimal holds ${held} GiB (limit 4 GiB + 200 MiB)" || fail "optimal holds ${held} GiB" ;;
    powerful)
      [[ "$NUMA_NODES" -gt 1 ]] && { [[ -n "$numa" ]] && pass "powerful picked the NUMA policy '$numa'" || fail "no NUMA policy in the powerful profile"; } ;;
  esac
done

fi

if [[ -n "$ARCHIVE" ]]; then
  say "the installer: install.sh --archive into /opt/nanollm-check as nanollm-check"
  SUDO=""
  [[ $(id -u) -eq 0 ]] || { have sudo && SUDO=sudo || SUDO=""; }
  if [[ $(id -u) -ne 0 && -z "$SUDO" ]]; then
    fail "the installer needs root or sudo"
  else
    DATA=/var/lib/nanollm-check
    $SUDO mkdir -p "$DATA"
    if [[ -d "$MODELS" ]]; then info "copying the models from $MODELS (no download)"; $SUDO cp -a "$MODELS" "$DATA/models"; fi
    $SUDO bash "$HERE/install.sh" --archive "$ARCHIVE" --prefix /opt/nanollm-check --data-dir "$DATA" \
      --addr 127.0.0.1:18092 --service nanollm-check --mode optimal --no-cuda 2>&1 | tee "$OUT/install.log" | sed 's/^/    /'
    [[ ${PIPESTATUS[0]} -eq 0 ]] && pass "install.sh finished" || fail "install.sh failed (see $OUT/install.log)"
    if have systemctl; then
      systemctl is-active --quiet nanollm-check && pass "nanollm-check is active" || fail "nanollm-check is not active"
    fi
    curl -fsS -m 10 http://127.0.0.1:18092/healthz >"$OUT/install-healthz.json" 2>/dev/null && pass "/healthz answers" || fail "/healthz does not answer"
    $SUDO /opt/nanollm-check/nanollm optimize --status --json -config "$DATA/nanollm.yaml" >"$OUT/install-status.json" 2>&1
    grep -q '"stale": false' "$OUT/install-status.json" && grep -q '"missing": false' "$OUT/install-status.json" &&
      pass "the installer left a current profile" || fail "optimize --status after the install: $(tr -d '\n ' <"$OUT/install-status.json" | cut -c1-200)"
    $SUDO cp "$DATA"/tune/*.json "$OUT/" 2>/dev/null
    $SUDO bash "$HERE/install.sh" --uninstall --purge --prefix /opt/nanollm-check --data-dir "$DATA" --service nanollm-check 2>&1 |
      tee "$OUT/uninstall.log" | sed 's/^/    /'
    $SUDO rm -rf "$DATA" /opt/nanollm-check
  fi
fi

say "result"
if [[ $FAILED -eq 0 ]]; then echo "    all checks passed"; else echo "    $FAILED checks failed"; fi
echo "    send the directory $OUT back"
exit $((FAILED > 0))
