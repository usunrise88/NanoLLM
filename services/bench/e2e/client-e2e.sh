#!/usr/bin/env bash
# The real NanoLLM client against benchd: `nanollm bench submit` signs and
# sends testdata/sample-result.json (the client speaks only HTTPS, so
# benchd serves a throwaway self-signed certificate the client trusts via
# SSL_CERT_FILE), the pages show it, and `nanollm bench withdraw` removes it.
#
#   services/bench/e2e/client-e2e.sh path/to/nanollm path/to/benchd
set -euo pipefail

nanollm=$(realpath "${1:?usage: client-e2e.sh NANOLLM BENCHD}")
benchd=$(realpath "${2:?usage: client-e2e.sh NANOLLM BENCHD}")
here=$(cd "$(dirname "$0")" && pwd)
sample="$here/../testdata/sample-result.json"
port=${PORT:-18443}
base="https://localhost:$port"

work=$(mktemp -d)
pid=
cleanup() {
	[ -n "$pid" ] && kill "$pid" 2>/dev/null || true
	rm -rf "$work"
}
trap cleanup EXIT

openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days 1 \
	-subj /CN=localhost -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" \
	-keyout "$work/key.pem" -out "$work/cert.pem" 2>/dev/null

BENCHD_ADDR="127.0.0.1:$port" BENCHD_DB="$work/bench.db" BENCHD_PUBLIC_URL="$base" \
	BENCHD_TLS_CERT="$work/cert.pem" BENCHD_TLS_KEY="$work/key.pem" \
	"$benchd" serve >"$work/benchd.log" 2>&1 &
pid=$!
for _ in $(seq 100); do
	curl -sf --cacert "$work/cert.pem" "$base/healthz" >/dev/null && break
	sleep 0.1
done

cat >"$work/nanollm.yaml" <<EOF
server:
  data_dir: $work/data
bench:
  submit_url: $base
EOF

fetch() { curl -sS --cacert "$work/cert.pem" -H 'Accept-Language: en' "$@"; }
fail() {
	echo "client-e2e: $*" >&2
	echo "--- benchd log" >&2
	cat "$work/benchd.log" >&2
	exit 1
}

export SSL_CERT_FILE="$work/cert.pem"
out=$("$nanollm" bench submit -config "$work/nanollm.yaml" -yes "$sample") || fail "submit failed: $out"
echo "$out"
id=$(awk '/^sent:/ {print $2}' <<<"$out")
[ -n "$id" ] || fail "no submission id in: $out"

fetch "$base/benchmarks/s/$id" | grep -q "Intel Core i7-12700H" || fail "the submission page does not show the result"
fetch "$base/benchmarks" | grep -q "ASUS ROG Zephyrus M16" || fail "/benchmarks does not show the result"
fetch "$base/api/v1/results?system=asus-rog-zephyrus-m16" | grep -q '"installs": 1' || fail "the results API does not count the install"

"$nanollm" bench withdraw -config "$work/nanollm.yaml" "$id" || fail "withdraw failed"
code=$(fetch -o /dev/null -w '%{http_code}' "$base/benchmarks/s/$id")
[ "$code" = 410 ] || fail "a withdrawn submission answers $code, want 410"
echo "client-e2e: ok ($id submitted, shown and withdrawn)"
