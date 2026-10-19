#!/usr/bin/env bash
# Staging of the benchmark service (services/bench) on hyper: benchd in a
# container on 127.0.0.1:27418, the host's own Caddy in front of it at
# nano.llmto.day, the whole site behind basic_auth until the legal review is
# done. The steps, the vendor-verified badge and how to submit a result are in
# services/bench/README.md, "Staging на hyper".
#
#   sudo scripts/staging-bench-service.sh caddy [--hash HASH]   # root: site file -> /etc/caddy/sites.d/, validate, reload
#   sudo scripts/staging-bench-service.sh caddy remove          # root: take the site off Caddy again
#   scripts/staging-bench-service.sh up                         # no sudo: build and start benchd
#   scripts/staging-bench-service.sh down                       # no sudo: stop it, the database volume stays
#   scripts/staging-bench-service.sh status                     # no sudo: read-only checks
#
# Only `caddy` needs root; `up`, `down` and `status` refuse it, because the
# admin token lives in the owner's home. Every command can be run again: the
# site file is rewritten only when it differs, the token is made once, and
# compose recreates a container only when its configuration changed.
#
# HASH is `caddy hash-password` output for the staging login (user "owner"),
# also read from $BENCH_BASIC_AUTH_HASH (sudo --preserve-env=BENCH_BASIC_AUTH_HASH).
# Once a hash is installed it is kept when the command is run again without
# one. Quote it: it contains `$`.
set -euo pipefail
cd "$(dirname "$0")/.."

SITE=nano.llmto.day
PORT=27418 # the same number as in services/bench/deploy/staging/
SITE_SRC=services/bench/deploy/staging/$SITE.caddy
SITE_DST=/etc/caddy/sites.d/$SITE.caddy
PLACEHOLDER=@BASIC_AUTH_HASH@
ENV_FILE=${BENCH_STAGING_ENV:-${XDG_CONFIG_HOME:-$HOME/.config}/nanollm-bench-staging/env}

die() {
	echo "staging-bench-service: $*" >&2
	exit 1
}

need_root() { [[ $EUID -eq 0 ]] || die "'$1' changes /etc/caddy and reloads Caddy: sudo $0 $*"; }
need_user() { [[ $EUID -ne 0 ]] || die "'$1' is not for root (the admin token and docker are the owner's): $0 $1"; }

# --- caddy (root) ----------------------------------------------------------

# validate checks the whole config as the caddy user. Run as root, `caddy
# validate` would create the site's log file owned by root, and Caddy (the
# caddy user) could not open it afterwards.
validate() {
	runuser -u caddy -- env HOME=/var/lib/caddy "$CADDY" validate --config /etc/caddy/Caddyfile --adapter caddyfile 2>&1
}

installed_hash() {
	[[ -f $SITE_DST ]] && awk '$1 == "owner" && $2 ~ /^\$2/ { print $2; exit }' "$SITE_DST"
	return 0
}

cmd_caddy() {
	need_root caddy "$@"
	CADDY=$(command -v caddy) || die "caddy is not installed"
	id caddy >/dev/null 2>&1 || die "no user 'caddy': the system Caddy is expected"
	command -v runuser >/dev/null || die "runuser (util-linux) is needed to validate as the caddy user"
	[[ -f /etc/caddy/Caddyfile ]] || die "no /etc/caddy/Caddyfile"

	local hash=${BENCH_BASIC_AUTH_HASH:-} remove=false
	while (($#)); do
		case $1 in
		remove) remove=true ;;
		--hash)
			[[ $# -ge 2 ]] || die "--hash needs a value"
			hash=$2
			shift
			;;
		*) die "caddy: unknown argument '$1'" ;;
		esac
		shift
	done

	# Global, for the trap.
	old=$(mktemp)
	new=$(mktemp)
	trap 'rm -f "$old" "$new"' EXIT
	local had=false
	if [[ -f $SITE_DST ]]; then
		had=true
		cp -p "$SITE_DST" "$old"
	fi

	if $remove; then
		if ! $had; then
			echo "$SITE_DST is not installed, nothing to remove"
			return 0
		fi
		rm -f "$SITE_DST"
	else
		local text uses
		text=$(<"$SITE_SRC")
		uses=$(grep -c "$PLACEHOLDER" "$SITE_SRC" || true)
		[[ $uses -le 1 ]] || die "$SITE_SRC holds the placeholder $PLACEHOLDER $uses times, once is the login"
		# No placeholder, no login (the template after the legal review).
		if [[ $uses -eq 1 ]]; then
			[[ -n $hash ]] || hash=$(installed_hash)
			if [[ -z $hash ]]; then
				die "no password hash for the staging login (user owner). Make one with
    caddy hash-password
and run again with
    sudo $0 caddy --hash '\$2a\$14\$...'"
			fi
			local bcrypt='^\$2[abxy]\$[0-9]{2}\$[./A-Za-z0-9]{53}$'
			[[ $hash =~ $bcrypt ]] || die "the hash is not a bcrypt hash from 'caddy hash-password' (quote it: it contains \$)"
			text=${text//"$PLACEHOLDER"/$hash}
		fi
		printf '%s\n' "$text" >"$new"
		if $had && cmp -s "$new" "$SITE_DST"; then
			echo "$SITE_DST is up to date"
		else
			install -m 0644 -o root -g root "$new" "$SITE_DST"
			echo "installed $SITE_DST"
		fi
	fi

	local out
	if ! out=$(validate); then
		# Put the previous state back: a broken sites.d file would make the
		# next restart of Caddy fail for every site on this host.
		if $had; then install -m 0644 -o root -g root "$old" "$SITE_DST"; else rm -f "$SITE_DST"; fi
		echo "$out" >&2
		die "caddy validate failed; $SITE_DST is back as it was"
	fi
	echo "caddy validate: ok"
	systemctl reload caddy || die "systemctl reload caddy failed; $SITE_DST stays as validated, see: journalctl -u caddy -n 30"
	echo "caddy reloaded"
	if ! $remove; then
		echo "next: scripts/staging-bench-service.sh up   (then: status)"
	fi
}

# --- compose (no root) -----------------------------------------------------

compose() {
	local args=()
	if [[ -f $ENV_FILE ]]; then
		args+=(--env-file "$ENV_FILE")
	else
		# down and status only interpolate the compose file.
		export BENCHD_ADMIN_TOKEN=${BENCHD_ADMIN_TOKEN:-unused}
	fi
	docker compose "${args[@]}" -f services/bench/deploy/docker-compose.yml -f services/bench/deploy/staging/docker-compose.staging.yml "$@"
}

has_docker() { command -v docker >/dev/null && docker compose version >/dev/null 2>&1; }
need_docker() { has_docker || die "docker compose is not available to ${USER:-this user}"; }

make_env() {
	[[ -f $ENV_FILE ]] && return 0
	(
		umask 077
		mkdir -p "$(dirname "$ENV_FILE")"
		printf 'BENCHD_ADMIN_TOKEN=%s\n' "$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')" >"$ENV_FILE"
	)
	echo "made $ENV_FILE with a new BENCHD_ADMIN_TOKEN (admin API, mode 600)"
}

cmd_up() {
	need_user up
	need_docker
	make_env
	# The first build downloads the Go toolchain image and modules.
	compose up -d --build benchd
	local i
	for i in $(seq 1 30); do
		if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
			echo "benchd answers on http://127.0.0.1:$PORT"
			return 0
		fi
		sleep 1
	done
	compose logs --tail 20 benchd >&2 || true
	die "benchd does not answer on 127.0.0.1:$PORT"
}

cmd_down() {
	need_user down
	need_docker
	compose down
	echo "stopped; the database stays in the volume nanollm-bench-staging_bench-data"
}

code() { curl -sS -o /dev/null -w '%{http_code}' --max-time 10 "$@" 2>/dev/null || true; }

cmd_status() {
	need_user status
	echo "== Caddy site file"
	if [[ ! -f $SITE_DST ]]; then
		echo "not installed ($SITE_DST): sudo $0 caddy --hash '...'"
	elif grep -q "$PLACEHOLDER" "$SITE_DST"; then
		echo "installed with the PLACEHOLDER hash; run caddy again with --hash"
	elif ! grep -q "127.0.0.1:$PORT" "$SITE_DST"; then
		echo "installed, but not for port $PORT: $SITE_DST differs from the repository"
	else
		echo "installed, reverse proxy to 127.0.0.1:$PORT"
	fi
	echo "== container"
	if has_docker; then compose ps benchd || true; else echo "docker compose is not available"; fi
	echo "== benchd on the loopback"
	local h
	h=$(code "http://127.0.0.1:$PORT/healthz")
	echo "GET /healthz -> ${h:-000} (200 expected)"
	echo "== through Caddy (resolved to this host)"
	h=$(code --resolve "$SITE:443:127.0.0.1" "https://$SITE/benchmarks")
	echo "GET /benchmarks without a login -> ${h:-000} (401 expected; 000: no certificate yet or the site is not loaded)"
	h=$(code --resolve "$SITE:443:127.0.0.1" "https://$SITE/admin/")
	echo "GET /admin/ from the host -> ${h:-000} (200 expected: the page, its API needs the token)"
}

case ${1:-} in
caddy)
	shift
	cmd_caddy "$@"
	;;
up) cmd_up ;;
down) cmd_down ;;
status) cmd_status ;;
*)
	sed -n '2,/^set -euo/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//' >&2
	exit 2
	;;
esac
