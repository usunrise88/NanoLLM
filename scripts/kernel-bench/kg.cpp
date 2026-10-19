// Standalone harness for the GEMM: old Q4_0 8x8 x Q8_0 4x8 AVX2 kernel (b11600) against candidates, bit-exact check and timing.
#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <type_traits>
#include <x86intrin.h>

#define GGML_RESTRICT __restrict
#define UNUSED(x) (void)(x)
#define GGML_CPU_FP16_TO_FP32(x) _cvtsh_ss(x)
#define GGML_CPU_E8M0_TO_FP32_HALF(x) 0.0f
#define QK8_0 32
typedef uint16_t ggml_half;
struct block_q4_0x8 { ggml_half d[8]; int8_t qs[128]; };
struct block_iq4_nlx8 { ggml_half d[8]; int8_t qs[128]; };
struct block_mxfp4x8 { uint8_t e[8]; int8_t qs[128]; };
struct block_q8_0 { ggml_half d; int8_t qs[32]; };
struct block_q8_0x4 { ggml_half d[4]; int8_t qs[128]; };
#define GGML_F32Cx8_LOAD(x)     _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x)))
#define GGML_F32Cx8_REPEAT_LOAD(x, loadMask)     _mm256_cvtph_ps(_mm_shuffle_epi32(_mm_maskload_epi32((int const*)(x), loadMask), 68))
#define GGML_F32Cx8_REARRANGE_LOAD(x, arrangeMask)     _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) x), arrangeMask))

#include "old_helpers.inc"
#include "old_gemm.inc"
#include "new_gemm.inc"
#include "real.inc"

static uint64_t rng_state = 88172645463325252ULL;
static inline uint64_t rnd() { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17; return rng_state; }
static inline float frnd() { return (float)((rnd() >> 11) * (1.0 / 9007199254740992.0)); }

struct Q4Block { ggml_half d; uint8_t qs[16]; };

static block_q4_0x8 make_x8(Q4Block * in) {
    block_q4_0x8 out;
    for (int i = 0; i < 8; i++) out.d[i] = in[i].d;
    for (int i = 0; i < 16; ++i) {
        int src_id = i % 8;
        int src_offset = (i / 8) * 8;
        int dst_offset = i * 8;
        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], 8);
        elems ^= 0x8888888888888888ULL;
        memcpy(&out.qs[dst_offset], &elems, 8);
    }
    return out;
}

static block_q8_0x4 make_x4(block_q8_0 * in) {
    block_q8_0x4 out;
    for (int i = 0; i < 4; i++) out.d[i] = in[i].d;
    for (int i = 0; i < 16; ++i) {
        int src_id = i % 4;
        int src_offset = (i / 4) * 8;
        memcpy(&out.qs[i * 8], &in[src_id].qs[src_offset], 8);
    }
    return out;
}

// mode 0 random, 1 extremes, 2 all max, 3 all min
static void gen(int n, int nc, int nr, int mode, std::vector<block_q4_0x8> & W, std::vector<block_q8_0x4> & Y) {
    const int nb = n / 32;
    W.resize((size_t)nc / 8 * nb);
    for (int g = 0; g < nc / 8; g++) {
        for (int b = 0; b < nb; b++) {
            Q4Block rows[8];
            for (int r = 0; r < 8; r++) {
                float d = (frnd() - 0.5f) * 0.2f;
                rows[r].d = _cvtss_sh(d, 0);
                for (int j = 0; j < 16; j++) {
                    uint8_t lo, hi;
                    if (mode == 0) { lo = rnd() & 15; hi = rnd() & 15; }
                    else if (mode == 1) { lo = (rnd() & 1) ? 15 : 0; hi = (rnd() & 1) ? 15 : 0; }
                    else if (mode == 2) { lo = 15; hi = 15; }
                    else { lo = 0; hi = 0; }
                    rows[r].qs[j] = lo | (hi << 4);
                }
            }
            W[(size_t)g * nb + b] = make_x8(rows);
        }
    }
    Y.resize((size_t)nr / 4 * nb);
    for (int g = 0; g < nr / 4; g++) {
        for (int b = 0; b < nb; b++) {
            block_q8_0 rows[4];
            for (int r = 0; r < 4; r++) {
                float x[32];
                for (int j = 0; j < 32; j++) {
                    if (mode == 0) x[j] = (frnd() - 0.5f) * 4.f;
                    else if (mode == 1) x[j] = (rnd() & 1) ? 1.f : -1.f;
                    else if (mode == 2) x[j] = 1.f;
                    else x[j] = -1.f;
                }
                float amax = 0; for (int j = 0; j < 32; j++) amax = std::max(amax, fabsf(x[j]));
                float d = amax / 127.f, id = d ? 1.f / d : 0.f;
                rows[r].d = _cvtss_sh(d, 0);
                for (int j = 0; j < 32; j++) rows[r].qs[j] = (int8_t)roundf(x[j] * id);
            }
            Y[(size_t)g * nb + b] = make_x4(rows);
        }
    }
}

typedef void (*kfn)(int, float *, size_t, const void *, const void *, int, int);

static void old_fn(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    __m256i lut = _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
    lut = _mm256_permute2f128_si256(lut, lut, 0);
    gemm_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8>(n, s, bs, vx, vy, nr, nc, lut);
}

struct Variant { const char * name; kfn fn; };
static Variant variants[] = {
    {"old", old_fn},
#define VARIANT(nm, f) {nm, f},
#include "variants_gemm.inc"
#undef VARIANT
};

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char ** argv) {
    const int nv = sizeof(variants) / sizeof(variants[0]);
    int fails = 0;
    for (int mode = 0; mode < 4; mode++) {
        for (int trial = 0; trial < (mode == 0 ? 8 : 1); trial++) {
            int n = 32 * (1 + (int)(rnd() % 60));
            if (mode == 0 && trial == 7) n = 32 * (510 + (int)(rnd() % 20)); // around the stack buffer size
            int nc = 8 * (1 + (int)(rnd() % 20));
            int nr = 4 * (1 + (int)(rnd() % 12));
            std::vector<block_q4_0x8> W; std::vector<block_q8_0x4> Y;
            gen(n, nc, nr, mode, W, Y);
            std::vector<float> ref((size_t)nr * nc), got((size_t)nr * nc);
            old_fn(n, ref.data(), nc, W.data(), Y.data(), nr, nc);
            for (int v = 1; v < nv; v++) {
                if (variants[v].name[strlen(variants[v].name) - 1] == '*') continue; // timing-only variant
                std::fill(got.begin(), got.end(), -1.f);
                variants[v].fn(n, got.data(), nc, W.data(), Y.data(), nr, nc);
                if (memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) != 0) {
                    int bad = 0; for (size_t i = 0; i < ref.size(); i++) if (memcmp(&ref[i], &got[i], 4)) bad++;
                    printf("MISMATCH variant %s mode %d n %d nc %d nr %d: %d of %zu differ (first ref %g got %g)\n", variants[v].name, mode, n, nc, nr, bad, ref.size(), ref[0], got[0]);
                    fails++;
                }
            }
        }
    }
    printf("gemm correctness: %s\n", fails ? "FAILED" : "all variants bit-identical to old");
    if (argc > 1 && !strcmp(argv[1], "check")) return fails != 0;
    if (argc > 5 && !strcmp(argv[1], "run")) {
        int v = -1; for (int i = 0; i < nv; i++) if (!strcmp(variants[i].name, argv[2])) v = i;
        if (v < 0) { printf("no such variant\n"); return 2; }
        int n = atoi(argv[3]), nc = atoi(argv[4]), nr = atoi(argv[5]); double secs = argc > 6 ? atof(argv[6]) : 1.5;
        std::vector<block_q4_0x8> W; std::vector<block_q8_0x4> Y;
        gen(n, nc, nr, 0, W, Y);
        std::vector<float> out((size_t)nr * nc);
        double t0 = now(); long calls = 0;
        while (now() - t0 < secs) { for (int r = 0; r < 20; r++) variants[v].fn(n, out.data(), nc, W.data(), Y.data(), nr, nc); calls += 20; }
        double rb = (double)nc / 8 * (n / 32) * nr * calls;
        printf("%s: %.2f ns per row-block (8 weight rows x 32 x 1 act row)\n", argv[2], (now() - t0) * 1e9 / rb);
        return 0;
    }

    struct Cfg { const char * name; int n, nc, nr, reps; } cfgs[] = {
        {"n=2560 nc=64 nr=16 ", 2560, 64, 16, 40},
        {"n=2560 nc=160 nr=512", 2560, 160, 512, 2},
        {"n=10240 nc=64 nr=128", 10240, 64, 128, 3},
        {"n=2560 nc=32 nr=4  ", 2560, 32, 4, 200},
    };
    for (auto & c : cfgs) {
        std::vector<block_q4_0x8> W; std::vector<block_q8_0x4> Y;
        gen(c.n, c.nc, c.nr, 0, W, Y);
        std::vector<float> out((size_t)c.nr * c.nc);
        double rowblocks = (double)c.nc / 8 * (c.n / 32) * c.nr;
        printf("%s (%.0f row-blocks/call)\n", c.name, rowblocks);
        std::vector<double> best(nv, 1e30);
        for (int round = 0; round < (getenv("KB_ROUNDS") ? atoi(getenv("KB_ROUNDS")) : 7); round++) {
            for (int v = 0; v < nv; v++) {
                double t0 = now();
                for (int r = 0; r < c.reps; r++) {
                    variants[v].fn(c.n, out.data(), c.nc, W.data(), Y.data(), c.nr, c.nc);
                    __asm__ volatile("" ::: "memory");
                }
                double t = (now() - t0) / c.reps;
                best[v] = std::min(best[v], t);
            }
        }
        for (int v = 0; v < nv; v++)
            printf("  %-14s %9.2f us/call  %7.2f ns/row-block  %.2fx\n", variants[v].name, best[v] * 1e6, best[v] * 1e9 / rowblocks, best[0] / best[v]);
    }
    return 0;
}
