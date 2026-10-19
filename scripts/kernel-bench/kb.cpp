// Standalone harness: old Q4_0 8x8 GEMV (b11600) against new kernels, bit-exact check and timing.
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
#define GGML_F32Cx8_REARRANGE_LOAD(x, arrangeMask) _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) x), arrangeMask))

struct block_q8_0x4 { ggml_half d[4]; int8_t qs[128]; };
#include "old_helpers.inc"
#include "old_kernel.inc"
#include "new_kernels.inc"
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

// mode 0 random, 1 extremes (nibble 0/15, y +-127)
static void gen(int n, int nc, int mode, std::vector<block_q4_0x8> & W, std::vector<block_q8_0> & Y) {
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
    Y.resize(nb);
    for (int b = 0; b < nb; b++) {
        float x[32];
        for (int j = 0; j < 32; j++) {
            if (mode == 0) x[j] = (frnd() - 0.5f) * 4.f;
            else if (mode == 1) x[j] = (rnd() & 1) ? 1.f : -1.f;
            else if (mode == 2) x[j] = 1.f;
            else x[j] = -1.f;
        }
        float amax = 0; for (int j = 0; j < 32; j++) amax = std::max(amax, fabsf(x[j]));
        float d = amax / 127.f, id = d ? 1.f / d : 0.f;
        Y[b].d = _cvtss_sh(d, 0);
        for (int j = 0; j < 32; j++) Y[b].qs[j] = (int8_t)roundf(x[j] * id);
    }
}

typedef void (*kfn)(int, float *, size_t, const void *, const void *, int, int);

static void old_fn(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    __m256i lut = _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
    lut = _mm256_permute2f128_si256(lut, lut, 0);
    gemv_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8>(n, s, bs, vx, vy, nr, nc, lut);
}

struct Variant { const char * name; kfn fn; };
static Variant variants[] = {
    {"old", old_fn},
#define VARIANT(nm, f) {nm, f},
#include "variants.inc"
#undef VARIANT
};

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char ** argv) {
    const int nv = sizeof(variants) / sizeof(variants[0]);
    // ---- correctness
    int fails = 0;
    for (int mode = 0; mode < 4; mode++) {
        for (int trial = 0; trial < (mode == 0 ? 6 : 1); trial++) {
            int n = 32 * (1 + (int)(rnd() % 90));
            if (mode == 0 && trial == 5) n = 32 * (1020 + (int)(rnd() % 60)); // more blocks than the stack buffers hold
            int nc = 8 * (1 + (int)(rnd() % 40));
            std::vector<block_q4_0x8> W; std::vector<block_q8_0> Y;
            gen(n, nc, mode, W, Y);
            std::vector<float> ref(nc), got(nc);
            old_fn(n, ref.data(), 1, W.data(), Y.data(), 1, nc);
            for (int v = 1; v < nv; v++) {
                if (variants[v].name[strlen(variants[v].name) - 1] == '*') continue; // timing-only variant
                std::fill(got.begin(), got.end(), -1.f);
                variants[v].fn(n, got.data(), 1, W.data(), Y.data(), 1, nc);
                if (memcmp(ref.data(), got.data(), nc * sizeof(float)) != 0) {
                    int bad = 0; for (int i = 0; i < nc; i++) if (memcmp(&ref[i], &got[i], 4)) bad++;
                    printf("MISMATCH variant %s mode %d n %d nc %d: %d of %d differ (first ref %g got %g)\n", variants[v].name, mode, n, nc, bad, nc, ref[0], got[0]);
                    fails++;
                }
            }
        }
    }
    printf("correctness: %s\n", fails ? "FAILED" : "all variants bit-identical to old");
    if (argc > 1 && !strcmp(argv[1], "check")) return fails != 0;
    if (argc > 3 && !strcmp(argv[1], "run")) {
        // kb run VARIANT N NC SECONDS : one variant only, for perf stat
        int v = -1; for (int i = 0; i < nv; i++) if (!strcmp(variants[i].name, argv[2])) v = i;
        if (v < 0) { printf("no such variant\n"); return 2; }
        int n = atoi(argv[3]), nc = atoi(argv[4]); double secs = atof(argv[5]);
        std::vector<block_q4_0x8> W; std::vector<block_q8_0> Y;
        gen(n, nc, 0, W, Y);
        std::vector<float> out(nc);
        double t0 = now(); long calls = 0;
        while (now() - t0 < secs) { for (int r = 0; r < 200; r++) variants[v].fn(n, out.data(), 1, W.data(), Y.data(), 1, nc); calls += 200; }
        double blocks = (double)nc / 8 * (n / 32) * calls;
        printf("%s: %.2f ns/block, %.0f blocks\n", argv[2], (now() - t0) * 1e9 / blocks, blocks);
        return 0;
    }

    // ---- timing
    struct Cfg { const char * name; int n, nc, reps; } cfgs[] = {
        {"L1  n=2560 nc=16 ", 2560, 16, 20000},
        {"L2  n=2560 nc=256", 2560, 256, 1500},
        {"mem n=2560 nc=8192", 2560, 8192, 40},
        {"mem n=10240 nc=2560", 10240, 2560, 40},
    };
    for (auto & c : cfgs) {
        std::vector<block_q4_0x8> W; std::vector<block_q8_0> Y;
        gen(c.n, c.nc, 0, W, Y);
        std::vector<float> out(c.nc);
        double blocks = (double)c.nc / 8 * (c.n / 32);
        printf("%s (%.0f blocks/call)\n", c.name, blocks);
        // interleave variants, take min of several rounds
        std::vector<double> best(nv, 1e30);
        for (int round = 0; round < 7; round++) {
            for (int v = 0; v < nv; v++) {
                double t0 = now();
                for (int r = 0; r < c.reps; r++) {
                    variants[v].fn(c.n, out.data(), 1, W.data(), Y.data(), 1, c.nc);
                    __asm__ volatile("" ::: "memory");
                }
                double t = (now() - t0) / c.reps;
                best[v] = std::min(best[v], t);
            }
        }
        for (int v = 0; v < nv; v++)
            printf("  %-14s %9.2f us/call  %7.2f ns/block  %.2fx\n", variants[v].name, best[v] * 1e6, best[v] * 1e9 / blocks, best[0] / best[v]);
    }
    return 0;
}
