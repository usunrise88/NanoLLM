#!/usr/bin/env bash
#
# Install NanoLLM on Linux (a systemd service) or macOS (a launchd agent).
#
#   curl -fsSL https://github.com/usunrise88/NanoLLM/releases/latest/download/install.sh | bash
#
# The one-liner is the whole interface: it works out the latest release,
# checks that this machine can run it, unpacks the archive (SPEC §10:
# /opt/nanollm and /var/lib/nanollm on Linux, /usr/local/nanollm and
# ~/Library/Application Support/NanoLLM on macOS), writes a configuration
# with two API keys, downloads the default model and leaves a running service
# behind. Run it again to upgrade in place: the configuration, the keys and
# the weights are kept. Every decision it makes is printed, and every one can
# be overridden by a flag, or by the matching NANOLLM_* variable for the pipe
# where flags are awkward:
#
#   curl -fsSL .../install.sh | bash -s -- --addr 0.0.0.0:8080 --cuda
#   NANOLLM_ADDR=0.0.0.0:8080 curl -fsSL .../install.sh | bash
#
# The service itself is registered by the binary (`nanollm service install`),
# so the installer and `nanollm service` can never disagree about the unit.
# --help lists the rest.
set -euo pipefail

REPO="${NANOLLM_REPO:-usunrise88/NanoLLM}"
VERSION="${NANOLLM_VERSION:-}"
PREFIX="${NANOLLM_PREFIX:-}"
DATA_DIR="${NANOLLM_DATA_DIR:-}"
ADDR="${NANOLLM_ADDR:-127.0.0.1:8080}"
# Whether an address was asked for at all: an upgrade keeps the
# configuration, so a flag that was given and not applied has to be reported.
ADDR_SET=0; [[ -n "${NANOLLM_ADDR:-}" ]] && ADDR_SET=1
SVC_USER="${NANOLLM_USER:-nanollm}"
SERVICE="${NANOLLM_SERVICE:-}"
WITH_UI="${NANOLLM_UI:-1}"
WITH_MODEL="${NANOLLM_DOWNLOAD:-1}"
# Empty means "leave it as it is": none on a fresh install, and on an upgrade
# whatever the installation already has, refreshed to match the new binary.
CUDA="${NANOLLM_CUDA:-}"
START="${NANOLLM_START:-1}"
DAEMON="${NANOLLM_DAEMON:-0}"
ARCHIVE="${NANOLLM_ARCHIVE:-}"
CUDA_ARCHIVE="${NANOLLM_CUDA_ARCHIVE:-}"
# The autotuner mode to measure this machine for after the install (silent,
# optimal or powerful); empty asks, or takes optimal without a terminal.
MODE="${NANOLLM_MODE:-}"
OPTIMIZE="${NANOLLM_OPTIMIZE:-1}"
ACTION=install
PURGE=0
# The piped form cannot pass a flag, so these have variables as well.
if [[ "${NANOLLM_UNINSTALL:-}" == "1" ]]; then ACTION=uninstall; fi
if [[ "${NANOLLM_PURGE:-}" == "1" ]]; then PURGE=1; fi

OS="$(uname -s)"
SUDO=""
TMP=""
BASE=""
KEYS=""
DEPS_MISSING=""

say()  { printf '==> %s\n' "$*" >&2; }
warn() { printf '    warning: %s\n' "$*" >&2; }
die()  { printf 'install.sh: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

usage() {
  cat >&2 <<EOF
Install NanoLLM as a service: systemd on Linux, launchd on macOS.

  --version TAG       release to install                    (default: the latest)
  --prefix DIR        binary and libraries                  (Linux /opt/nanollm, macOS /usr/local/nanollm)
  --data-dir DIR      configuration, weights, job database  (Linux /var/lib/nanollm,
                                                             macOS ~/Library/Application Support/NanoLLM)
  --addr HOST:PORT    listen address                        ($ADDR)
  --user NAME         Linux: system account to run as       ($SVC_USER)
  --service NAME      unit name or launchd label            (nanollm, com.github.usunrise88.nanollm)
  --no-ui             install the build without the web interface
  --no-download       write the configuration but fetch no model
  --cuda              Linux x86-64: add the CUDA libraries (needs an NVIDIA GPU and driver)
  --no-cuda           remove them again
  --daemon            macOS: a LaunchDaemon for the whole machine instead of an agent
  --no-start          install everything, start nothing
  --mode MODE         silent, optimal or powerful: measure this machine for it after the install
                      (a new install asks; without a terminal it takes optimal; see: nanollm optimize)
  --no-optimize       do not measure; run `nanollm optimize` later
  --archive PATH      install this archive instead of downloading one (tests, offline)
  --cuda-archive PATH the CUDA add-on to go with --archive
  --uninstall         stop and remove the service and the installation
  --purge             with --uninstall: also delete the configuration, the keys and the weights

Each option has an environment variable: NANOLLM_VERSION, NANOLLM_PREFIX,
NANOLLM_DATA_DIR, NANOLLM_ADDR, NANOLLM_USER, NANOLLM_SERVICE, NANOLLM_UI=0,
NANOLLM_DOWNLOAD=0, NANOLLM_CUDA=1 (or 0), NANOLLM_DAEMON=1, NANOLLM_START=0,
NANOLLM_ARCHIVE, NANOLLM_CUDA_ARCHIVE, NANOLLM_MODE, NANOLLM_OPTIMIZE=0,
NANOLLM_UNINSTALL=1, NANOLLM_PURGE=1.
A flag wins over one.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)      VERSION="${2:?--version needs a tag}"; shift 2 ;;
    --prefix)       PREFIX="${2:?--prefix needs a directory}"; shift 2 ;;
    --data-dir)     DATA_DIR="${2:?--data-dir needs a directory}"; shift 2 ;;
    --addr)         ADDR="${2:?--addr needs host:port}"; ADDR_SET=1; shift 2 ;;
    --user)         SVC_USER="${2:?--user needs a name}"; shift 2 ;;
    --service)      SERVICE="${2:?--service needs a name}"; shift 2 ;;
    --no-ui)        WITH_UI=0; shift ;;
    --no-download)  WITH_MODEL=0; shift ;;
    --cuda)         CUDA=1; shift ;;
    --no-cuda)      CUDA=0; shift ;;
    --daemon)       DAEMON=1; shift ;;
    --no-start)     START=0; shift ;;
    --mode)         MODE="${2:?--mode needs silent, optimal or powerful}"; shift 2 ;;
    --no-optimize)  OPTIMIZE=0; shift ;;
    --archive)      ARCHIVE="${2:?--archive needs a path}"; shift 2 ;;
    --cuda-archive) CUDA_ARCHIVE="${2:?--cuda-archive needs a path}"; CUDA=1; shift 2 ;;
    --uninstall)    ACTION=uninstall; shift ;;
    --purge)        PURGE=1; shift ;;
    -h|--help)      usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done

cleanup() {
  [[ -n "$TMP" ]] && rm -rf "$TMP"
  return 0
}
trap cleanup EXIT

# Paths given relative to where the script was started from stay meaningful
# after it changes directory.
abspath() { [[ "$1" == /* ]] && printf '%s' "$1" || printf '%s/%s' "$(pwd)" "$1"; }
[[ -z "$ARCHIVE" ]] || ARCHIVE="$(abspath "$ARCHIVE")"
[[ -z "$CUDA_ARCHIVE" ]] || CUDA_ARCHIVE="$(abspath "$CUDA_ARCHIVE")"

# --- platform ---------------------------------------------------------------

# The account the service runs as and the account that owns the data. On
# Linux that is a system account; on macOS the agent belongs to the user who
# ran this, through sudo or not, and a daemon runs as root.
RUN_AS=""
RUN_HOME=""

case "$OS" in
  Linux)
    PREFIX="${PREFIX:-/opt/nanollm}"
    DATA_DIR="${DATA_DIR:-/var/lib/nanollm}"
    SERVICE="${SERVICE:-nanollm}"
    RUN_AS="$SVC_USER"
    ;;
  Darwin)
    PREFIX="${PREFIX:-/usr/local/nanollm}"
    SERVICE="${SERVICE:-com.github.usunrise88.nanollm}"
    if [[ "$DAEMON" == "1" ]]; then
      RUN_AS=root
      DATA_DIR="${DATA_DIR:-/Library/Application Support/NanoLLM}"
    else
      RUN_AS="${SUDO_USER:-$(id -un)}"
      [[ "$RUN_AS" != root ]] || die "a LaunchAgent belongs to a user, and this is root; run it as yourself, or pass --daemon"
      RUN_HOME="$(eval printf '%s' "~$RUN_AS")"
      DATA_DIR="${DATA_DIR:-$RUN_HOME/Library/Application Support/NanoLLM}"
    fi
    ;;
  *) die "this installs on Linux and macOS; on Windows use install.ps1" ;;
esac
CONFIG="$DATA_DIR/nanollm.yaml"
EXE="$PREFIX/nanollm"

systemd_running() { have systemctl && [[ -d /run/systemd/system ]]; }

# --- privileges ---------------------------------------------------------------

# escalate decides how the privileged steps run. Root is needed for the
# prefix, the system account and the unit, and for nothing else, so each
# privileged command is prefixed rather than the script re-executing itself:
# there is nothing to re-execute when it arrived through a pipe, and sudo
# reads the password from the terminal even when stdin is that pipe.
escalate() {
  if [[ "$(id -u)" -eq 0 ]]; then
    SUDO=""
    return
  fi
  # A macOS agent under a prefix this user can write needs no root at all.
  if [[ "$OS" == Darwin && "$DAEMON" != "1" ]] && can_write "$PREFIX"; then
    SUDO=""
    return
  fi
  have sudo || die "this needs root and sudo is not installed; run it as root"
  SUDO="sudo"
  say "not running as root; the privileged steps go through sudo"
  sudo -v || die "sudo refused; run this as root instead"
}

can_write() {
  local d="$1"
  while [[ ! -e "$d" ]]; do d="$(dirname "$d")"; done
  [[ -w "$d" ]]
}

# as_owner runs a command as the account that owns the data, so that what it
# writes (the configuration, the weights) already belongs to the account
# that has to read and write them later.
as_owner() {
  if [[ "$(id -un)" == "$RUN_AS" ]]; then
    "$@"
  elif [[ "$(id -u)" -eq 0 ]] && have runuser; then
    runuser -u "$RUN_AS" -- "$@"
  else
    sudo -u "$RUN_AS" -H -- "$@"
  fi
}

# --- preconditions --------------------------------------------------------------

TARGET=""
preflight() {
  case "$OS-$(uname -m)" in
    Linux-x86_64|Linux-amd64) TARGET=linux-amd64 ;;
    Linux-aarch64|Linux-arm64) TARGET=linux-arm64 ;;
    Darwin-*) TARGET=darwin-universal ;;
    *) die "there is no release for $OS $(uname -m)" ;;
  esac
  if [[ "$OS" == Linux ]] && compgen -G "/lib/ld-musl-*" >/dev/null 2>&1; then
    die "this is a musl system (Alpine); the llama.cpp libraries need glibc — use the Docker image"
  fi

  have tar || die "tar is required"
  [[ -n "$ARCHIVE" ]] || have curl || die "curl is required"
  have sha256sum || have shasum || die "sha256sum or shasum is required"
  if [[ "$OS" == Linux ]] && ! systemd_running; then
    warn "systemd is not running here (a container?); the files will be installed and no service will be"
  fi
}

# --- the release --------------------------------------------------------------

# latest_tag asks GitHub which release is current: the redirect that
# /releases/latest performs first, because the API is rate limited.
latest_tag() {
  local url tag
  url="$(curl -fsSL -o /dev/null -w '%{url_effective}' "https://github.com/$REPO/releases/latest" 2>/dev/null || true)"
  tag="${url##*/}"
  if [[ "$url" == *"/releases/tag/"* && -n "$tag" ]]; then
    printf '%s\n' "$tag"
    return
  fi
  tag="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null |
    sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)"
  [[ -n "$tag" ]] || die "cannot work out the latest release; pass --version TAG"
  printf '%s\n' "$tag"
}

fetch() { curl -fSL -# --retry 4 --retry-delay 2 --connect-timeout 20 -o "$2" "$1"; }

sha256_check() {
  # sha256sum -c reads "<hash>  <file>" lines; shasum -a 256 -c does too.
  if have sha256sum; then sha256sum -c --quiet -; else shasum -a 256 -c --quiet -; fi
}

# verify DIR FILE SUMS: FILE against the line for it in SUMS. HTTPS already
# says the bytes came from GitHub; this says that all of them arrived.
verify() {
  local dir="$1" file="$2" sums="$3" line pattern
  # The name is matched literally: it is full of dots.
  pattern="$(printf '%s' "$file" | sed 's/[^[:alnum:]_-]/\\&/g')"
  line="$(grep -E "[[:space:]][*]?${pattern}\$" "$sums" || true)"
  [[ -n "$line" ]] || die "$(basename "$sums") does not mention $file"
  (cd "$dir" && printf '%s\n' "$line" | sha256_check) || die "$file failed its checksum"
  say "checksum ok: $file"
}

# obtain NAME [LOCAL]: the archive NAME in $TMP, verified. A local archive is
# used as it is, checked against a sha256sums.txt beside it when there is one.
obtain() {
  local name="$1" local_path="${2:-}"
  if [[ -n "$local_path" ]]; then
    [[ -f "$local_path" ]] || die "no such archive: $local_path"
    if [[ -f "$(dirname "$local_path")/sha256sums.txt" ]]; then
      verify "$(dirname "$local_path")" "$(basename "$local_path")" "$(dirname "$local_path")/sha256sums.txt"
    else
      say "using $local_path (no sha256sums.txt beside it to check against)"
    fi
    printf '%s' "$local_path"
    return
  fi
  say "downloading $name"
  fetch "$BASE/$name" "$TMP/$name" || die "no such release asset: $BASE/$name"
  if [[ ! -f "$TMP/sha256sums.txt" ]]; then
    fetch "$BASE/sha256sums.txt" "$TMP/sha256sums.txt" 2>/dev/null ||
      die "$VERSION publishes no sha256sums.txt, so $name cannot be verified"
  fi
  verify "$TMP" "$name" "$TMP/sha256sums.txt"
  printf '%s' "$TMP/$name"
}

# --- dependencies -------------------------------------------------------------

# have_lib NAME: whether the dynamic loader can find a shared library.
have_lib() {
  if have ldconfig && ldconfig -p 2>/dev/null | grep "[[:space:]]$1 " >/dev/null; then
    return 0
  fi
  local d
  for d in /lib /lib64 /usr/lib /usr/lib64 /lib/*-linux-gnu /usr/lib/*-linux-gnu /usr/local/lib; do
    [[ -e "$d/$1" ]] && return 0
  done
  return 1
}

# check_deps: the llama.cpp Linux builds link libgomp (every CPU backend)
# and libvulkan (the Vulkan backend), and neither travels in the archive
# (STAGE1, facts of the pinned tag). Nothing is installed behind the
# operator's back; what is missing is named with the command that adds it.
check_deps() {
  [[ "$OS" == Linux ]] || return 0
  local missing=() pkgs
  have_lib libgomp.so.1 || missing+=(libgomp.so.1)
  have_lib libvulkan.so.1 || missing+=(libvulkan.so.1)
  (( ${#missing[@]} )) || return 0
  DEPS_MISSING="${missing[*]}"

  if have apt-get; then pkgs="sudo apt-get install"
    [[ " ${missing[*]} " == *libgomp* ]] && pkgs+=" libgomp1"
    [[ " ${missing[*]} " == *libvulkan* ]] && pkgs+=" libvulkan1 mesa-vulkan-drivers"
  elif have dnf; then pkgs="sudo dnf install"
    [[ " ${missing[*]} " == *libgomp* ]] && pkgs+=" libgomp"
    [[ " ${missing[*]} " == *libvulkan* ]] && pkgs+=" vulkan-loader mesa-vulkan-drivers"
  elif have zypper; then pkgs="sudo zypper install"
    [[ " ${missing[*]} " == *libgomp* ]] && pkgs+=" libgomp1"
    [[ " ${missing[*]} " == *libvulkan* ]] && pkgs+=" libvulkan1 Mesa-vulkan-device-select"
  elif have pacman; then pkgs="sudo pacman -S"
    [[ " ${missing[*]} " == *libgomp* ]] && pkgs+=" gcc-libs"
    [[ " ${missing[*]} " == *libvulkan* ]] && pkgs+=" vulkan-icd-loader"
  else
    pkgs="your package manager: the packages that provide ${missing[*]}"
  fi
  if [[ " ${missing[*]} " == *libgomp* ]]; then
    warn "libgomp.so.1 (GCC's OpenMP runtime) is not installed. Every llama.cpp CPU
    backend links it, so until it is there the server can load no model."
  fi
  if [[ " ${missing[*]} " == *libvulkan* ]]; then
    warn "libvulkan.so.1 (the Vulkan loader) is not installed, so the Vulkan backend
    is skipped and models run on the CPU. A GPU also needs its driver's Vulkan
    ICD: mesa for AMD and Intel, the proprietary driver for NVIDIA."
  fi
  warn "install with: $pkgs"
}

# --- CUDA ---------------------------------------------------------------------

has_cuda_libs() { compgen -G "$PREFIX/lib/*ggml-cuda*" >/dev/null 2>&1; }

nvidia_present() {
  [[ -e /proc/driver/nvidia/version ]] || { have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; }
}

# install_cuda: the add-on is ggml-cuda and the CUDA runtime only, laid out
# as lib/, and goes over the installation. It must come from the same
# release as the binary: the CUDA runtime is the one that ggml-cuda was built
# with. (Upstream's ggml-cuda on Linux; the Windows add-ons carry ours, which
# nothing checks against the pinned commit: it is ggml-base that does.)
install_cuda() {
  [[ "$TARGET" == linux-amd64 ]] || { warn "the CUDA add-on is for Linux x86-64 only; skipped"; return; }
  if [[ -z "$CUDA_ARCHIVE" ]] && ! nvidia_present; then
    warn "--cuda: no NVIDIA GPU or driver found (/proc/driver/nvidia, nvidia-smi); the add-on is not installed"
    return
  fi
  local name="nanollm-${VERSION}-linux-amd64-cuda.tar.gz" path
  path="$(obtain "$name" "$CUDA_ARCHIVE")"
  say "unpacking the CUDA add-on into $PREFIX/lib"
  $SUDO tar -xzf "$path" -C "$PREFIX" --no-same-owner
}

# --- configuration ------------------------------------------------------------

# config_has_keys reports whether the configuration is a working one. It is
# asked of the file, through $SUDO because it is 0600 to its owner, and a
# failed read is not taken to mean "no keys": that would overwrite a working
# configuration on every upgrade.
config_has_keys() {
  $SUDO test -f "$CONFIG" || return 1
  local out
  out="$($SUDO "$EXE" key list -config "$CONFIG" 2>/dev/null)" || return 1
  [[ "$out" != *"no keys in"* ]]
}

# config_addr is server.addr from the configuration.
config_addr() {
  local out
  # shellcheck disable=SC2016 # awk program, not shell
  out="$($SUDO awk 'index($0, "  addr:") == 1 { sub("^  addr:[ \t]*", ""); sub("#.*", ""); print; exit }' "$CONFIG" 2>/dev/null)" || return 1
  out="$(printf '%s' "$out" | tr -d "\"' ")"
  [[ -n "$out" ]] || return 1
  printf '%s' "$out"
}

effective_addr() { config_addr || printf '%s' "$ADDR"; }

probe_url() {
  local addr host port
  addr="$(effective_addr)"
  host="${addr%:*}" port="${addr##*:}"
  case "$host" in ""|"0.0.0.0"|"::"|"[::]"|"*") host=127.0.0.1 ;; esac
  printf 'http://%s:%s' "$host" "$port"
}

# --- the service --------------------------------------------------------------

service_installed() {
  case "$OS" in
    Linux) [[ -f "/etc/systemd/system/$SERVICE.service" ]] ;;
    Darwin)
      if [[ "$DAEMON" == "1" ]]; then [[ -f "/Library/LaunchDaemons/$SERVICE.plist" ]]
      else [[ -f "$RUN_HOME/Library/LaunchAgents/$SERVICE.plist" ]]; fi ;;
  esac
}

# svc CMD [ARGS]: `nanollm service CMD` as whoever it must run as: root
# for systemd and a LaunchDaemon, the user for an agent.
svc() {
  local verb="$1"; shift
  case "$OS" in
    Linux) $SUDO "$EXE" service "$verb" -name "$SERVICE" "$@" ;;
    Darwin)
      if [[ "$DAEMON" == "1" ]]; then $SUDO "$EXE" service "$verb" -system -name "$SERVICE" "$@"
      else as_owner "$EXE" service "$verb" -name "$SERVICE" "$@"; fi ;;
  esac
}

can_run_service() { [[ "$OS" == Darwin ]] || systemd_running; }

wait_for_health() {
  local url i
  url="$(probe_url)"
  for ((i = 0; i < 60; i++)); do
    if curl -fsS --max-time 2 "$url/healthz" >/dev/null 2>&1; then
      say "$SERVICE answers on $url"
      return 0
    fi
    sleep 1
  done
  warn "no answer from $url/healthz after a minute; see: $EXE service status, and $(log_hint)"
  return 1
}

log_hint() {
  case "$OS" in
    Linux) printf 'journalctl -u %s -n 50' "$SERVICE" ;;
    Darwin)
      if [[ "$DAEMON" == "1" ]]; then printf '/Library/Logs/NanoLLM/nanollm.log'
      else printf '%s/Library/Logs/NanoLLM/nanollm.log' "$RUN_HOME"; fi ;;
  esac
}

# --- install ------------------------------------------------------------------

ensure_user() {
  [[ "$OS" == Linux ]] || return 0
  if ! id -u "$SVC_USER" >/dev/null 2>&1; then
    local shell=/usr/sbin/nologin
    [[ -x "$shell" ]] || shell=/sbin/nologin
    [[ -x "$shell" ]] || shell=/bin/false
    say "creating the system account $SVC_USER"
    $SUDO useradd --system --home-dir "$DATA_DIR" --shell "$shell" "$SVC_USER" ||
      die "could not create the account $SVC_USER"
  fi
  # The GPU device nodes (/dev/dri/renderD*, /dev/nvidia*) belong to these
  # groups on most distributions; a service outside them runs on the CPU.
  local g
  for g in video render; do
    if getent group "$g" >/dev/null 2>&1 && ! id -nG "$SVC_USER" | tr ' ' '\n' | grep -qx "$g"; then
      $SUDO usermod -aG "$g" "$SVC_USER" || warn "could not add $SVC_USER to the $g group"
    fi
  done
}

install_nanollm() {
  escalate
  preflight
  TMP="$(mktemp -d)"

  local suffix="" archive name had_cuda=0
  [[ "$WITH_UI" == "1" ]] || suffix="-noui"
  if [[ -n "$ARCHIVE" ]]; then
    # The tag is in the name: nanollm-<tag>-<target>[-noui].<ext>.
    name="$(basename "$ARCHIVE")"
    if [[ -z "$VERSION" ]]; then
      VERSION="${name#nanollm-}"
      VERSION="${VERSION%-"$TARGET"*}"
    fi
  else
    [[ -n "$VERSION" ]] || { say "asking GitHub for the latest release"; VERSION="$(latest_tag)"; }
    BASE="https://github.com/$REPO/releases/download/$VERSION"
    name="nanollm-${VERSION}-${TARGET}${suffix}.tar.gz"
  fi
  archive="$(obtain "$name" "$ARCHIVE")"

  # Stopped before anything is overwritten: replacing the binary and the
  # libraries under a running process is how a restart mixes versions.
  if [[ -x "$EXE" ]] && can_run_service && service_installed; then
    say "stopping the running $SERVICE service"
    svc stop >/dev/null 2>&1 || true
  fi

  ensure_user
  $SUDO mkdir -p "$PREFIX"
  if [[ "$OS" == Linux ]]; then
    $SUDO install -d -o "$SVC_USER" -g "$(id -gn "$SVC_USER")" -m 0750 "$DATA_DIR"
  else
    as_owner mkdir -p "$DATA_DIR"
  fi

  # lib/ is replaced, not merged: the new binary checks the libraries'
  # commit, and a CUDA backend left from the last release would not match.
  has_cuda_libs && had_cuda=1
  say "unpacking $(basename "$archive") into $PREFIX"
  $SUDO rm -rf "$PREFIX/lib"
  $SUDO tar -xzf "$archive" -C "$PREFIX" --no-same-owner
  [[ -x "$EXE" ]] || die "$(basename "$archive") did not contain nanollm"
  # A browser's quarantine flag on a downloaded archive is inherited by what
  # it unpacks, and Gatekeeper refuses to load a quarantined unsigned dylib.
  if [[ "$OS" == Darwin ]] && have xattr; then
    $SUDO xattr -dr com.apple.quarantine "$PREFIX" 2>/dev/null || true
  fi

  case "$CUDA:$had_cuda" in
    1:*) install_cuda ;;
    :1) say "refreshing the CUDA add-on to match the new binary"; install_cuda ;;
    0:1) say "the CUDA add-on was removed (--no-cuda)" ;;
  esac

  check_deps

  if config_has_keys; then
    say "keeping the configuration in $CONFIG"
    if [[ "$ADDR_SET" == "1" && "$(effective_addr)" != "$ADDR" ]]; then
      warn "--addr was not applied: an upgrade keeps the configuration, and the server
    listens on $(effective_addr). Set server.addr in $CONFIG and restart:
      $EXE service restart"
    fi
  else
    say "writing the configuration with two fresh API keys"
    # Into a file rather than the terminal: the keys are printed once, in
    # the summary at the end, where they are not scrolled away.
    if ! as_owner "$EXE" init -config "$CONFIG" -data-dir "$DATA_DIR" -addr "$ADDR" -force -no-download >"$TMP/init.log" 2>&1; then
      cat "$TMP/init.log" >&2
      die "nanollm init did not finish; the output above says why. Run this again once it is fixed"
    fi
    KEYS="$(grep -E '^ *(admin|user) key' "$TMP/init.log" || true)"
  fi

  if [[ "$WITH_MODEL" == "1" ]]; then
    say "fetching the default model (gigabytes; --no-download skips it)"
    as_owner "$EXE" models pull -config "$CONFIG" ||
      warn "the model was not fetched; the server starts without it: $EXE models pull -config \"$CONFIG\""
  fi

  if ! can_run_service; then
    say "installed in $PREFIX; start it with: $EXE serve -config \"$CONFIG\""
    summary
    return
  fi
  local args=(-config "$CONFIG")
  [[ "$OS" == Linux ]] && args+=(-user "$SVC_USER")
  [[ "$START" == "1" ]] || args+=(-no-start)
  say "registering the $SERVICE service"
  svc install "${args[@]}" >&2 || die "nanollm service install failed; the output above says why"
  local healthy=0
  if [[ "$START" == "1" ]] && wait_for_health; then healthy=1; fi
  if [[ "$healthy" == "1" && "$WITH_MODEL" == "1" ]]; then optimize_machine "$([[ -n "$KEYS" ]] && echo 1 || echo 0)"; fi
  summary
  # /healthz does not wait for a model, so a service that does not answer it
  # within a minute is broken, and a deployment script has to be told.
  [[ "$START" != "1" || "$healthy" == "1" ]] || die "the service did not come up; see $(log_hint)"
}

# --- the autotuner ---------------------------------------------------------------

# json_field prints the value of a top-level key of the small JSON that
# `nanollm optimize --status --json` prints, without needing jq.
json_field() { sed -n "s/^ *\"$1\": *\"\{0,1\}\([^\",]*\)\"\{0,1\},\{0,1\} *$/\1/p" | head -1; }

# interactive says whether a person can be asked: a terminal on input and
# output, which a piped `curl | bash` is not.
interactive() { [[ -t 0 && -t 2 ]]; }

# optimize_machine measures this machine for a mode through the running
# service (the measurement happens inside it) and applies the result. A new
# install measures for the mode asked for, or asks, or takes optimal without a
# terminal; an upgrade offers a recheck when the libraries or the driver made
# the profile stale (docs/OPTIMIZE.md, 4.3). Whatever goes wrong is a warning
# with the command to repeat: the service works by the limits of the mode and
# the static rules without a profile.
optimize_machine() {
  local new_install="$1" status cur stale missing mode="$MODE" a
  if [[ "$OPTIMIZE" != "1" ]]; then
    say "not measured (--no-optimize); to tune this machine: $EXE optimize"
    return
  fi
  status="$(as_owner "$EXE" optimize --status --json -config "$CONFIG" 2>/dev/null || true)"
  cur="$(printf '%s\n' "$status" | json_field mode)"
  stale="$(printf '%s\n' "$status" | json_field stale)"
  missing="$(printf '%s\n' "$status" | json_field missing)"
  if [[ -n "$cur" ]]; then
    # A mode is set: an upgrade, or a configuration that was kept.
    if [[ -z "$mode" && "$stale" != "true" && "$missing" != "true" ]]; then
      say "the $cur profile is current"
      return
    fi
    if [[ -z "$mode" ]]; then
      if interactive; then
        say "the $cur profile needs a recheck (the libraries, the driver or the hardware changed)"
        read -r -p "Measure again in the $cur mode? [Y/n] " a </dev/tty || a=n
        case "$a" in [nN]*) say "later: $EXE optimize --mode $cur"; return ;; esac
        mode="$cur"
      else
        warn "the $cur profile needs a recheck. Run: $EXE optimize --mode $cur --yes"
        return
      fi
    fi
  elif [[ -z "$mode" ]]; then
    if interactive; then
      mode="$(ask_mode)"
    elif [[ "$new_install" == "1" ]]; then
      mode=optimal
    else
      # An upgrade of an installation that never chose a mode: its behaviour is
      # not changed behind the person's back.
      say "this installation has no autotuner mode yet. To tune this machine: $EXE optimize"
      return
    fi
  fi
  case "$mode" in silent|optimal|powerful) ;; *) warn "'$mode' is not a mode (silent, optimal, powerful); not measured"; return ;; esac
  say "measuring this machine for the $mode mode (about a minute; the model is unavailable meanwhile)"
  as_owner "$EXE" optimize -config "$CONFIG" --mode "$mode" --yes >&2 ||
    warn "the measurement failed. The service runs by the limits of the mode; to try again: $EXE optimize --mode $mode --yes"
}

# ask_mode shows the three modes and prints the one chosen; Enter is optimal.
ask_mode() {
  local a
  {
    echo
    echo "How should NanoLLM use this machine?"
    echo "  1. silent    the least load: CPU only, 2 threads, low priority; the slowest answers"
    echo "  2. optimal   up to 4 GiB of memory and half of the cores; the fastest answer within that"
    echo "  3. powerful  all the free memory and cores"
  } >&2
  read -r -p "Mode [2]: " a </dev/tty || a=2
  case "$(printf '%s' "$a" | tr '[:upper:]' '[:lower:]')" in
    1|silent) echo silent ;;
    3|powerful) echo powerful ;;
    ""|2|optimal) echo optimal ;;
    *) warn "'$a' is not a mode; using optimal"; echo optimal ;;
  esac
}

summary() {
  local url
  url="$(probe_url)"
  cat >&2 <<EOF

NanoLLM $VERSION is installed.

  service   $SERVICE        $EXE service status
  binary    $EXE
  config    $CONFIG
  data      $DATA_DIR
  log       $(log_hint)
  cuda      $(if has_cuda_libs; then echo "installed in $PREFIX/lib"; else echo "not installed (--cuda adds it on an NVIDIA machine)"; fi)
  url       $url    ($url/ui/ in a browser)

EOF
  if [[ -n "$KEYS" ]]; then
    printf '%s\n\n' "$KEYS" >&2
  else
    printf '  The API keys are in %s (nanollm key list -config "%s").\n\n' "$CONFIG" "$CONFIG" >&2
  fi
  if [[ -n "$DEPS_MISSING" ]]; then
    printf '  Missing: %s; see the warning above.\n\n' "$DEPS_MISSING" >&2
  fi
  cat >&2 <<EOF
  curl -sS $url/v1/chat/completions -H "Authorization: Bearer <the user key>" \\
    -H 'Content-Type: application/json' \\
    -d '{"model": "gemma-4-e4b-qat", "messages": [{"role": "user", "content": "Hi"}]}'

  update     curl -fsSL https://github.com/$REPO/releases/latest/download/install.sh | bash
  uninstall  curl -fsSL https://github.com/$REPO/releases/latest/download/install.sh | bash -s -- --uninstall
EOF
}

# --- uninstall ----------------------------------------------------------------

uninstall_nanollm() {
  escalate
  if [[ -x "$EXE" ]] && can_run_service; then
    say "removing the $SERVICE service"
    svc uninstall >&2 || warn "nanollm service uninstall failed; removing the unit by hand"
  fi
  # Without the binary (or after it failed), the unit goes by hand, so that a
  # half-removed installation can still be cleaned up.
  case "$OS" in
    Linux)
      if [[ -f "/etc/systemd/system/$SERVICE.service" ]]; then
        systemd_running && { $SUDO systemctl disable --now "$SERVICE" 2>/dev/null || true; }
        $SUDO rm -f "/etc/systemd/system/$SERVICE.service"
        systemd_running && { $SUDO systemctl daemon-reload || true; }
      fi ;;
    Darwin)
      local plist domain
      if [[ "$DAEMON" == "1" ]]; then plist="/Library/LaunchDaemons/$SERVICE.plist"; domain=system
      else plist="$RUN_HOME/Library/LaunchAgents/$SERVICE.plist"; domain="gui/$(id -u "$RUN_AS")"; fi
      if [[ -f "$plist" ]]; then
        launchctl bootout "$domain/$SERVICE" 2>/dev/null || true
        $SUDO rm -f "$plist"
      fi ;;
  esac
  if [[ -d "$PREFIX" ]]; then
    say "removing $PREFIX"
    $SUDO rm -rf "$PREFIX"
  fi
  if [[ "$PURGE" == "1" ]]; then
    say "removing $DATA_DIR: the configuration, the keys, the weights and the job database"
    $SUDO rm -rf "$DATA_DIR"
    if [[ "$OS" == Darwin ]]; then
      if [[ "$DAEMON" == "1" ]]; then $SUDO rm -rf /Library/Logs/NanoLLM; else rm -rf "$RUN_HOME/Library/Logs/NanoLLM"; fi
    fi
    # Only an account this script made, which has the data directory as its
    # home: --user www-data, or nobody, must survive a purge.
    if [[ "$OS" == Linux ]] && id -u "$SVC_USER" >/dev/null 2>&1; then
      if [[ "$(getent passwd "$SVC_USER" | cut -d: -f6)" == "$DATA_DIR" ]]; then
        $SUDO userdel "$SVC_USER" 2>/dev/null || warn "the account $SVC_USER is still there"
      else
        say "kept the account $SVC_USER: this installer did not create it"
      fi
    fi
  else
    say "kept $DATA_DIR (the configuration, the keys and the weights); --purge deletes it"
  fi
  say "done"
}

case "$ACTION" in
  install)   install_nanollm ;;
  uninstall) uninstall_nanollm ;;
esac
