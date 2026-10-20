#!/usr/bin/env bash
# Staging of the benchmark service (services/bench) and of the static site (site/)
# on hyper: benchd in a container on 127.0.0.1:27418, the site (the landing page
# of Nano* and the product pages) in another on 127.0.0.1:27419, and the host's
# own Caddy in front of both at nano.llmto.day. The steps, the vendor-verified
# badge and how to submit a result are in services/bench/README.md, "Staging на hyper".
#
#   sudo scripts/staging-bench-service.sh caddy   # root: site file -> /etc/caddy/sites.d/, validate, reload
#   sudo scripts/staging-bench-service.sh caddy remove   # root: take the site off Caddy again
#   scripts/staging-bench-service.sh up       # no sudo: build and start benchd (and the site, if site/dist is built)
#   scripts/staging-bench-service.sh site     # no sudo: build site/ (npm) and start or refresh its container
#   scripts/staging-bench-service.sh down     # no sudo: stop both, the database volume stays
#   scripts/staging-bench-service.sh status   # no sudo: read-only checks
#
# Only `caddy` needs root; the rest refuse it, because the admin token lives in
# the owner's home. Every command can be run again: the site file is rewritten
# only when it differs, the token is made once, and compose recreates a
# container only when its configuration changed. The site is open (the owner
# accepted the privacy and consent texts); an older site file with a login
# (basic_auth, a password hash) is replaced by the one in the repository.
#
# If the site file in the repository ever holds the placeholder @BASIC_AUTH_HASH@
# again, `caddy` wants a login hash: --hash HASH (`caddy hash-password` output
# for the user "owner"), or $BENCH_BASIC_AUTH_HASH (sudo --preserve-env=...).
# Quote it: it contains `$`.
set -euo pipefail
cd "$(dirname "$0")/.."

SITE=nano.llmto.day
PORT=27418 # the same numbers as in services/bench/deploy/staging/
SITE_PORT=27419
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
		echo "next: scripts/staging-bench-service.sh up and site   (then: status)"
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
	if [[ -f site/dist/index.html ]]; then
		compose up -d --build benchd site
	else
		compose up -d --build benchd
	fi
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

# cmd_site builds site/ and starts or refreshes the container that serves it.
# The build is npm (a few minutes of one core the first time): run it when no
# measurement is going.
cmd_site() {
	need_user site
	need_docker
	make_env
	command -v npm >/dev/null || die "npm is needed to build site/"
	(cd site && nice -n 10 npm ci --no-audit --no-fund && nice -n 10 npm run build)
	compose up -d site
	local i
	for i in $(seq 1 30); do
		if curl -fsS --max-time 2 "http://127.0.0.1:$SITE_PORT/" >/dev/null 2>&1; then
			echo "the site answers on http://127.0.0.1:$SITE_PORT"
			return 0
		fi
		sleep 1
	done
	compose logs --tail 20 site >&2 || true
	die "the site does not answer on 127.0.0.1:$SITE_PORT"
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
		echo "not installed ($SITE_DST): sudo $0 caddy"
	elif grep -q "$PLACEHOLDER\|basic_auth" "$SITE_DST"; then
		echo "installed with a login (basic_auth): the site is still closed; run: sudo $0 caddy"
	elif ! grep -q "127.0.0.1:$PORT" "$SITE_DST" || ! grep -q "127.0.0.1:$SITE_PORT" "$SITE_DST"; then
		echo "installed, but not for ports $PORT and $SITE_PORT: $SITE_DST differs from the repository; run: sudo $0 caddy"
	else
		echo "installed: benchd on 127.0.0.1:$PORT, the site on 127.0.0.1:$SITE_PORT, no login"
	fi
	echo "== containers"
	if has_docker; then compose ps benchd site || true; else echo "docker compose is not available"; fi
	local h
	echo "== on the loopback"
	h=$(code "http://127.0.0.1:$PORT/healthz")
	echo "benchd GET /healthz -> ${h:-000} (200 expected)"
	h=$(code "http://127.0.0.1:$SITE_PORT/")
	echo "site GET / -> ${h:-000} (200 expected; run: $0 site)"
	echo "== through Caddy (resolved to this host)"
	h=$(code --resolve "$SITE:443:127.0.0.1" "https://$SITE/")
	echo "GET / -> ${h:-000} (200 expected: the landing page; 401: still behind a login; 000: no certificate yet or the site is not loaded)"
	h=$(code --resolve "$SITE:443:127.0.0.1" "https://$SITE/nanollm/")
	echo "GET /nanollm/ -> ${h:-000} (200 expected)"
	h=$(code --resolve "$SITE:443:127.0.0.1" "https://$SITE/benchmarks")
	echo "GET /benchmarks -> ${h:-000} (200 expected)"
	h=$(code --resolve "$SITE:443:127.0.0.1" "https://$SITE/admin/")
	echo "GET /admin/ -> ${h:-000} (200 expected: the page, its API needs the token)"
}

case ${1:-} in
caddy)
	shift
	cmd_caddy "$@"
	;;
up) cmd_up ;;
site) cmd_site ;;
down) cmd_down ;;
status) cmd_status ;;
*)
	sed -n '2,/^set -euo/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//' >&2
	exit 2
	;;
esac
