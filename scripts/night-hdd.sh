#!/usr/bin/env bash
# The HDD night queue of docs/STAGE2.md (stage 2 optimizations that need a
# quiet host): OPT-8 layer streaming on M8/M9, OPT-5 prompt lookup, OPT-14
# parallel repack. Every result is a file under bench/results/<host>/; a step
# that fails is logged and the queue goes on.
#
#   scripts/hyper-exclusive.sh -- scripts/night-hdd.sh [LABEL]
#
# Run it under hyper-exclusive.sh, which waits for the host to be quiet and
# keeps the other benchmark jobs out for the whole queue. The disk here is a
# hard disk (RAID1, ~110 MB/s sequential); the same queue is meant to be run
# again on an SSD (SSD=1 changes the labels, not the steps).
set -uo pipefail

cd "$(dirname "$0")/.."
LABEL=${1:-night}
[[ ${SSD:-0} == 1 ]] && LABEL=$LABEL-ssd
NANOLLM=${NANOLLM:-./dist/nanollm-stage}
NEXT=${NEXT:-./dist/nanollm-next}
CONFIG=${CONFIG:-$HOME/nanollm-data/base.yaml}
NEXT_CONFIG=${NEXT_CONFIG:-$HOME/nanollm-data/next.yaml}
SLOW_TIMEOUT=${SLOW_TIMEOUT:-14m}
STEPS=${STEPS-"stream ngram repack thp tune"}

bench() { # bench BIN CONFIG NAME ARGS...
	local bin=$1 cfg=$2 name=$3
	shift 3
	echo "=== $(date +%T) $name"
	nice -n 5 "$bin" bench --config "$cfg" --suite dev --label "$LABEL-$name" "$@" || echo "!!! $name failed (rc=$?)"
}
slow() { local name=$1; shift; bench "$NANOLLM" "$CONFIG" "$name" --slow --reps 1 --timeout "$SLOW_TIMEOUT" "$@"; }
full() { local name=$1; shift; bench "$NANOLLM" "$CONFIG" "$name" --reps "${REPS:-3}" --timeout 20m "$@"; }

for step in $STEPS; do
	case $step in
	stream)
		# OPT-8. The three presets that read weights from the disk on every
		# token: M8 at ram-2g and ram-1.5g, M9 at ram-1g. Variants: read ahead
		# only; evict behind with nothing kept; ahead + evict + keep N layers.
		for p in ram-2g ram-1.5g ram-1g; do
			slow "stream-ahead-$p" --profile strong --preset "$p" --set stream_ahead=0,1,2,4,8
			slow "stream-evict-$p" --profile strong --preset "$p" --set stream_ahead=0,4 --set stream_evict=on
			slow "stream-keep-$p" --profile strong --preset "$p" --set stream_ahead=4 --set stream_evict=on --set stream_keep=2,4,8
		done
		# The weak profile (4 cores of one node): the baseline and two settings.
		slow "stream-weak-ram-2g" --profile weak --preset ram-2g --set stream_ahead=0,4 --set stream_evict=off,on
		# The cost of the callback alone, where nothing streams (M6, resident).
		full "stream-probe-ram-8g" --profile strong --preset ram-8g --set env.NANOLLM_STREAM_PROBE=0,1
		;;
	ngram)
		# OPT-5. ngram=0 is the baseline (and MTP on echo, if the drafter loads);
		# chat-ru is the worst case for a lookup, echo the best.
		full "ngram-strong" --profile strong --preset ram-8g --set ngram=0
		full "ngram-strong-n3" --profile strong --preset ram-8g --set ngram=3 --set draft_max=2,4,8
		full "ngram-strong-n24" --profile strong --preset ram-8g --set ngram=2,4 --set draft_max=4
		full "ngram-weak" --profile weak --preset ram-4g --set ngram=0,3 --set draft_max=4
		;;
	repack)
		# OPT-14. Load time with the repack on 1, 4 and 16 threads; warm (the
		# file is in the page cache) and cold (the strict budget evicts it).
		# The binary is stamped for the series under test and says so.
		export NANOLLM_DEV_SERIES
		NANOLLM_DEV_SERIES=$(cat .cache/next.commit)
		bench "$NEXT" "$NEXT_CONFIG" "repack-cold" --reps 1 --timeout 20m --profile strong --preset ram-8g --set env.GGML_REPACK_THREADS=1,4,16
		bench "$NEXT" "$NEXT_CONFIG" "repack-warm" --reps 1 --timeout 20m --advisory --profile strong --preset ram-8g --set env.GGML_REPACK_THREADS=1,4,16
		unset NANOLLM_DEV_SERIES
		;;
	thp)
		# OPT-6, transparent huge pages for the repacked weights: glibc's malloc
		# madvises the big mappings (the repack buffer, the KV cache) with
		# MADV_HUGEPAGE when glibc.malloc.hugetlb=1; the host's THP mode is
		# madvise. Fewer TLB misses on a 2.5 GB scan per token.
		full "thp-strong" --profile strong --preset ram-8g --set env.GLIBC_TUNABLES=glibc.malloc.hugetlb=0,glibc.malloc.hugetlb=1
		full "thp-weak" --profile weak --preset ram-4g --set env.GLIBC_TUNABLES=glibc.malloc.hugetlb=0,glibc.malloc.hugetlb=1
		;;
	tune)
		# OPT-10: compiler tuning of the CPU kernels. The same series built
		# with -mtune=znver1, with LTO, with both; llama-bench interleaved.
		if [[ -d dist/tune-both/tools ]]; then
			echo "=== $(date +%T) tune A/B"
			scripts/tools-ab.sh tune dist/linux_amd64-next/tools dist/tune-znver1/tools dist/tune-lto/tools dist/tune-both/tools -- 2 || echo "!!! tune failed"
		fi
		;;
	esac
done
echo "=== $(date +%T) done"
