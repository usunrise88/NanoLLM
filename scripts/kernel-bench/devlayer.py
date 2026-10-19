#!/usr/bin/env python3
"""Dev-only layer for a patched llama.cpp checkout: adds every candidate of kernel-bench (new_kernels.inc,
new_gemm.inc) to ggml/src/ggml-cpu/arch/x86/repack.cpp as run-time switches, so that one libggml-cpu-haswell.so
can be timed with each of them (llama-bench next to itself, same build, same machine state):

    Q4K_GEMV  0 old LUT kernel, 1 the patch, 2 hadd_epi32, 3 shuffle+add, 4 vphaddw with 2 column groups, 5 vphaddw with prefetch
    Q4K_GEMM  0 old LUT kernel, 1 the patch (8 rows, vphaddw), 2 4 rows hadd, 3 4 rows vphaddw, 4 8 rows hadd,
              5 12 rows, 6 16 rows (vphaddw), 7 4 rows shuffle, 8 8 rows shuffle

    scripts/kernel-bench/devlayer.py [LLAMA_SRC]     # edits the checkout in place; `git checkout -- .` takes it out again
    scripts/kernel-bench/devbuild.sh                 # builds only libggml-cpu-haswell (about a minute)

Never ship a library built this way: it reads the environment in the hot path's dispatcher.
"""
import os
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, '.cache', 'llama.cpp', 'src')
F = os.path.join(src, 'ggml/src/ggml-cpu/arch/x86/repack.cpp')
KB = os.path.dirname(os.path.abspath(__file__)) + '/'
s = open(F).read()
if 'namespace devvar' in s:
    sys.exit('already applied')

layer = ('// DEV-ONLY variants, selected at run time by Q4K_GEMV / Q4K_GEMM; not part of the patch\n'
         '#if defined(GGML_REPACK_Q4_0_GEMV_U8)\nnamespace devvar {\n' + open(KB + 'new_kernels.inc').read() +
         open(KB + 'new_gemm.inc').read() + '}\n#endif\n\n')
i = s.index('void ggml_gemv_q4_0_8x8_q8_0(')
s = s[:i] + layer + s[i:]

LUT = """    {
        __m256i signextendlut = _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
        signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);
        %s<block_q4_0x8>(n, s, bs, vx, vy, nr, nc, signextendlut);
        return;
    }
#elif defined(__AVX2__) || defined(__AVX512F__)"""

old = """#if defined(GGML_REPACK_Q4_0_GEMV_U8)
    gemv_q4_0_8x8_q8_0_u8_avx<block_q4_0x8>(n, s, bs, vx, vy, nr, nc);
    return;
#elif defined(__AVX2__) || defined(__AVX512F__)"""
new = """#if defined(GGML_REPACK_Q4_0_GEMV_U8)
    static const int dev_var = getenv("Q4K_GEMV") ? atoi(getenv("Q4K_GEMV")) : 1;
    switch (dev_var) {
        case 1: gemv_q4_0_8x8_q8_0_u8_avx<block_q4_0x8>(n, s, bs, vx, vy, nr, nc); return;
        case 2: devvar::gemv_u8<devvar::RED_HADDD, 1, 1, 0>(n, s, bs, vx, vy, nr, nc); return;
        case 3: devvar::gemv_u8<devvar::RED_SHUF, 1, 1, 0>(n, s, bs, vx, vy, nr, nc); return;
        case 4: devvar::gemv_u8<devvar::RED_PHADDW, 1, 2, 0>(n, s, bs, vx, vy, nr, nc); return;
        case 5: devvar::gemv_u8<devvar::RED_PHADDW, 1, 1, 8>(n, s, bs, vx, vy, nr, nc); return;
        default: break;
    }
""" + LUT % 'gemv_q4_b32_8x8_q8_0_lut_avx'
assert s.count(old) == 1
s = s.replace(old, new)

old = """#if defined(GGML_REPACK_Q4_0_GEMM_U8)
    gemm_q4_0_8x8_q8_0_u8_avx<block_q4_0x8>(n, s, bs, vx, vy, nr, nc);
    return;
#elif defined(__AVX2__) || defined(__AVX512F__)"""
new = """#if defined(GGML_REPACK_Q4_0_GEMM_U8)
    static const int dev_var = getenv("Q4K_GEMM") ? atoi(getenv("Q4K_GEMM")) : 1;
    switch (dev_var) {
        case 1: gemm_q4_0_8x8_q8_0_u8_avx<block_q4_0x8>(n, s, bs, vx, vy, nr, nc); return;
        case 2: devvar::gemm_u8<0, 1, 1>(n, s, bs, vx, vy, nr, nc); return;
        case 3: devvar::gemm_u8<2, 1, 1>(n, s, bs, vx, vy, nr, nc); return;
        case 4: devvar::gemm_u8<0, 1, 2>(n, s, bs, vx, vy, nr, nc); return;
        case 5: devvar::gemm_u8<2, 1, 3>(n, s, bs, vx, vy, nr, nc); return;
        case 6: devvar::gemm_u8<2, 1, 4>(n, s, bs, vx, vy, nr, nc); return;
        case 7: devvar::gemm_u8<1, 1, 1>(n, s, bs, vx, vy, nr, nc); return;
        case 8: devvar::gemm_u8<1, 1, 2>(n, s, bs, vx, vy, nr, nc); return;
        default: break;
    }
""" + LUT % 'gemm_q4_b32_8x8_q8_0_lut_avx'
assert s.count(old) == 1
s = s.replace(old, new)
open(F, 'w').write(s)
print('dev layer applied to', F)
