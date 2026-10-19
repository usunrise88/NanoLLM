#!/usr/bin/env bash
# Root-only host settings that stage 2 measurements need (docs/STAGE2.md 2.0).
#
#   sudo scripts/host-setup-stage2.sh            # apply
#   sudo scripts/host-setup-stage2.sh --revert   # undo
#
# What it changes, and nothing else:
#
# 1. memlock. RLIMIT_MEMLOCK is 8 MiB, so the memory modes that lock weights
#    in RAM (M7 ram-ple-disk, M8 ram-hybrid: range-lock) cannot load. It
#      - writes a drop-in for the user manager (user@UID.service) and a
#        limits.d file, so the limit is unlimited for the next login and the
#        next restart of the user manager (neither is restarted here);
#      - raises the limit of the running Claude Code process(es) of this user
#        with prlimit, so the shells it starts from now on have it. Only
#        processes named "claude" of the user are touched.
#    Locked memory is still capped by the MemoryMax of the cgroup a run is in.
# 2. perf. kernel.perf_event_paranoid is 4, which stops `perf` for ordinary
#    users; it is set to 1 until the next reboot (not persisted) so the token
#    profile (OPT-12) can be taken. --revert puts back what it found.
#
# Nothing here opens a port, installs a package or touches another user's
# processes.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "run it with sudo: sudo $0 $*" >&2
  exit 1
fi

TARGET_USER=${SUDO_USER:-ubuntu}
TARGET_UID=$(id -u "$TARGET_USER")
DROPIN_DIR=/etc/systemd/system/user@${TARGET_UID}.service.d
DROPIN=$DROPIN_DIR/nanollm-memlock.conf
LIMITS=/etc/security/limits.d/90-nanollm-memlock.conf
STATE=/var/lib/nanollm-host-setup
PERF=/proc/sys/kernel/perf_event_paranoid

if [[ ${1:-} == --revert ]]; then
  rm -f "$DROPIN" "$LIMITS"
  rmdir "$DROPIN_DIR" 2>/dev/null || true
  systemctl daemon-reload
  if [[ -f $STATE/perf_event_paranoid ]]; then
    cat "$STATE/perf_event_paranoid" >"$PERF"
    rm -f "$STATE/perf_event_paranoid"
  fi
  echo "reverted: the drop-in and limits.d file are gone (running processes keep the limit they have until they exit)"
  echo "perf_event_paranoid is $(cat $PERF)"
  exit 0
fi

# 1. memlock
mkdir -p "$DROPIN_DIR"
printf '[Service]\nLimitMEMLOCK=infinity\n' >"$DROPIN"
printf '%s soft memlock unlimited\n%s hard memlock unlimited\n' "$TARGET_USER" "$TARGET_USER" >"$LIMITS"
systemctl daemon-reload
echo "memlock: wrote $DROPIN and $LIMITS (effective at the next login / user-manager start)"
for pid in $(pgrep -u "$TARGET_UID" -x claude || true); do
  prlimit --pid "$pid" --memlock=unlimited:unlimited
  echo "memlock: raised the limit of claude (pid $pid): $(prlimit --pid "$pid" --memlock --noheadings --output SOFT,HARD)"
done

# 2. perf
mkdir -p "$STATE"
[[ -f $STATE/perf_event_paranoid ]] || cat "$PERF" >"$STATE/perf_event_paranoid"
echo 1 >"$PERF"
echo "perf: perf_event_paranoid $(cat "$STATE/perf_event_paranoid") -> $(cat "$PERF") until reboot or --revert"
