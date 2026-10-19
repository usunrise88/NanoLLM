#!/usr/bin/env bash
# The HDD night queue of docs/STAGE2.md (stage 2 optimizations that need a
# quiet host): OPT-8 layer streaming on M8/M9, OPT-5 prompt lookup, OPT-14
# parallel repack. Every result is a file under bench/results/<host>/; a step
# that fails is logged and the queue goes on.
#
#   scripts/night-hdd.sh [LABEL]
#
# Every measurement takes the host lock (scripts/hyper-exclusive.sh: it waits
# for the host to be quiet) for itself only, so other benchmark jobs, such as
# the kernel A/B of another agent, interleave between the queue's runs instead
# of waiting for the whole night. The disk here is a
# hard disk (RAID1, ~110 MB/s sequential); the same queue is meant to be run
# again on an SSD (SSD=1 changes the labels, DATA points at the copy on it).
set -uo pipefail

cd "$(dirname "$0")/.."
LABEL=${1:-night}
[[ ${SSD:-0} == 1 ]] && LABEL=$LABEL-ssd
# DATA is the data directory the configs and the models live in; for the SSD
# run it is a copy on the SSD (scripts/ssd-setup.sh).
DATA=${DATA:-$HOME/nanollm-data}
NANOLLM=${NANOLLM:-./dist/nanollm-stage}
NEXT=${NEXT:-./dist/nanollm-next}
CONFIG=${CONFIG:-$DATA/base.yaml}
NEXT_CONFIG=${NEXT_CONFIG:-$DATA/next.yaml}
SLOW_TIMEOUT=${SLOW_TIMEOUT:-14m}
STEPS=${STEPS-"diskbw numa batch stream ngram evalngram repack thp tune standard"}

x() { scripts/hyper-exclusive.sh --max-load 6 -- "$@"; }

bench() { # bench BIN CONFIG NAME ARGS...
	local bin=$1 cfg=$2 name=$3
	shift 3
	echo "=== $(date +%T) $name"
	x nice -n 5 "$bin" bench --config "$cfg" --suite dev --label "$LABEL-$name" "$@" || echo "!!! $name failed (rc=$?)"
}
slow() { local name=$1; shift; bench "$NANOLLM" "$CONFIG" "$name" --slow --reps 1 --timeout "$SLOW_TIMEOUT" "$@"; }
full() { local name=$1; shift; bench "$NANOLLM" "$CONFIG" "$name" --reps "${REPS:-3}" --timeout 20m "$@"; }

for step in $STEPS; do
	case $step in
	diskbw)
		# What the disk can deliver by each way weights reach memory (OPT-8
		# against OPT-9). Two regions: the layers' area and the PLE table's.
		MODEL=$(ls "$DATA"/models/gemma-4-e4b-qat/*/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf | head -1)
		OUTF=bench/results/$(hostname -s)/$(date +%F)-diskbw$([[ ${SSD:-0} == 1 ]] && echo -ssd).txt
		{
			echo "# $(date -Is) $(findmnt -T "$MODEL" -no SOURCE,FSTYPE)"
			x ./dist/diskbw -file "$MODEL" -offset-mib 2100 -size-mib 1024 -threads 1,4,16
		} | tee "$OUTF"
		;;
	numa)
		# OPT-6, NUMA-local weights: the interleave policy (the default of the
		# server) against ggml's distribute strategy, whose threads are pinned
		# one per node and whose repack threads first-touch each slab on the
		# node that will read it (patches/next/0003). Same libraries, same
		# model, llama-bench, three rounds.
		MODEL=$(ls "$DATA"/models/gemma-4-e4b-qat/*/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf | head -1)
		OUTF=bench/results/$(hostname -s)/$(date +%F)-numa-distribute$([[ ${SSD:-0} == 1 ]] && echo -ssd).txt
		T=dist/linux_amd64-next/tools
		{
			echo "# $(date -Is) $MODEL"
			for round in 1 2 3; do
				for th in 8 16 24 32; do
					echo "## round $round threads $th: interleave, repack 1 thread (the current default)"
					x env GGML_REPACK_THREADS=1 nice -n 5 numactl --interleave=all $T/llama-bench -m "$MODEL" -t $th -p 512 -n 128 -fa 1 -r 2 -o md 2>/dev/null
					echo "## round $round threads $th: interleave, parallel repack"
					x env GGML_REPACK_THREADS=$th nice -n 5 numactl --interleave=all $T/llama-bench -m "$MODEL" -t $th -p 512 -n 128 -fa 1 -r 2 -o md 2>/dev/null
					echo "## round $round threads $th: distribute, repack threads = decode threads (node-local weights)"
					x env GGML_REPACK_THREADS=$th nice -n 5 $T/llama-bench -m "$MODEL" -t $th -p 512 -n 128 -fa 1 -r 2 --numa distribute -o md 2>/dev/null
					echo "## round $round threads $th: distribute, serial repack (all weights on one node)"
					x env GGML_REPACK_THREADS=1 nice -n 5 $T/llama-bench -m "$MODEL" -t $th -p 512 -n 128 -fa 1 -r 2 --numa distribute -o md 2>/dev/null
				done
			done
		} | tee "$OUTF"
		;;
	batch)
		# Aggregate decode throughput with N sequences in one batch: the weights
		# are read once for all of them, so on a memory-bound CPU the total
		# should grow almost with N until arithmetic takes over (the planner's
		# slot count, docs/STAGE2.md 2.8).
		MODEL=$(ls "$DATA"/models/gemma-4-e4b-qat/*/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf | head -1)
		OUTF=bench/results/$(hostname -s)/$(date +%F)-batched$([[ ${SSD:-0} == 1 ]] && echo -ssd).txt
		T=dist/linux_amd64-next/tools
		{
			echo "# $(date -Is) $MODEL"
			echo "## strong: 16 threads, interleave"
			x nice -n 5 numactl --interleave=all $T/llama-batched-bench -m "$MODEL" -c 6144 -b 2048 -ub 512 -npp 128 -ntg 64 -npl 1,2,4,8 -t 16 -fa on 2>/dev/null | tail -8
			echo "## weak: 4 cores of node 0"
			x nice -n 5 numactl --preferred=0 taskset -c 0,1,2,3 $T/llama-batched-bench -m "$MODEL" -c 6144 -b 2048 -ub 512 -npp 128 -ntg 64 -npl 1,2,4,8 -t 4 -fa on 2>/dev/null | tail -8
		} | tee "$OUTF"
		;;
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
	evalngram)
		# OPT-5 on real tasks: the same suites with and without prompt lookup.
		# Greedy answers must match; the per-item time says what it gained
		# (ru/en instructions: free text, json-tools: answers built from the
		# given data).
		EV=bench/results/$(hostname -s)
		for suite in ru-instruct json-tools; do
			for cfg in ref ref-ngram; do
				echo "=== $(date +%T) eval $suite $cfg"
				x nice -n 5 "$NANOLLM" eval --config "$DATA/$cfg.yaml" --suite "$suite" --thinking off \
					--out "$EV/$(date +%F)-eval-$suite-$cfg-timed.json" || echo "!!! eval $suite $cfg failed"
			done
		done
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
			x scripts/tools-ab.sh tune dist/linux_amd64-next/tools dist/tune-znver1/tools dist/tune-lto/tools dist/tune-both/tools -- 2 || echo "!!! tune failed"
		fi
		;;
	standard)
		# 2.9: the standard suite as a user runs it (auto and two presets,
		# strict budgets, three repetitions), not submitted anywhere.
		echo "=== $(date +%T) standard quick"
		x nice -n 5 "$NANOLLM" bench --config "$CONFIG" --suite standard --label "$LABEL-standard" || echo "!!! standard failed (rc=$?)"
		;;
	esac
done
echo "=== $(date +%T) done"
