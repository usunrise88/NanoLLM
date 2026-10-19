// ---- new kernels (candidates) ----
struct y_pre_t { float d; int32_t bias8; };

// Per-block prepass on the Q8_0 activation row: the scale as float and 8 * (sum of the 32 quants).
static inline void y_prepass(const block_q8_0 * a, int nb, y_pre_t * pre) {
    const __m256i ones8  = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (int b = 0; b < nb; b++) {
        const __m256i yv = _mm256_loadu_si256((const __m256i *)a[b].qs);
        const __m256i t  = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, yv), ones16);
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
        pre[b].d     = GGML_CPU_FP16_TO_FP32(a[b].d);
        pre[b].bias8 = 8 * _mm_cvtsi128_si32(s4);
    }
}

enum { RED_SHUF = 0, RED_HADDD = 1, RED_PHADDW = 2 };

// one block of 8 rows: acc += float(dot - bias) * (col scale * row scale)
template<int RED, int XOR>
static inline __attribute__((always_inline)) __m256 u8_block(const block_q4_0x8 & blk, __m256i y0, __m256i y1, __m256i y2, __m256i y3,
                                                           int32_t bias8, float d, __m256 acc_row) {
    const __m256i m4b    = _mm256_set1_epi8(0x0F);
    const __m256i m88    = _mm256_set1_epi8((char) 0x88);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m128i dmask  = _mm_set_epi8(15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0);

    __m256i ra = _mm256_loadu_si256((const __m256i *) (blk.qs));
    __m256i rb = _mm256_loadu_si256((const __m256i *) (blk.qs + 32));
    __m256i rc = _mm256_loadu_si256((const __m256i *) (blk.qs + 64));
    __m256i rd = _mm256_loadu_si256((const __m256i *) (blk.qs + 96));
    if constexpr (XOR) {
        ra = _mm256_xor_si256(ra, m88);
        rb = _mm256_xor_si256(rb, m88);
        rc = _mm256_xor_si256(rc, m88);
        rd = _mm256_xor_si256(rd, m88);
    }
    __m256i S = _mm256_maddubs_epi16(_mm256_and_si256(ra, m4b), y0);
    __m256i T = _mm256_maddubs_epi16(_mm256_and_si256(rb, m4b), y0);
    S = _mm256_add_epi16(S, _mm256_maddubs_epi16(_mm256_and_si256(rc, m4b), y1));
    T = _mm256_add_epi16(T, _mm256_maddubs_epi16(_mm256_and_si256(rd, m4b), y1));
    S = _mm256_add_epi16(S, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(ra, 4), m4b), y2));
    T = _mm256_add_epi16(T, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(rb, 4), m4b), y2));
    S = _mm256_add_epi16(S, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(rc, 4), m4b), y3));
    T = _mm256_add_epi16(T, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(rd, 4), m4b), y3));

    __m256i iacc;
    if constexpr (RED == RED_SHUF) {
        const __m256i mS = _mm256_madd_epi16(S, ones16);
        const __m256i mT = _mm256_madd_epi16(T, ones16);
        iacc = _mm256_add_epi32(
            _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(mS), _mm256_castsi256_ps(mT), 0x88)),
            _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(mS), _mm256_castsi256_ps(mT), 0xDD)));
    } else if constexpr (RED == RED_HADDD) {
        iacc = _mm256_hadd_epi32(_mm256_madd_epi16(S, ones16), _mm256_madd_epi16(T, ones16));
    } else {
        iacc = _mm256_madd_epi16(_mm256_hadd_epi16(S, T), ones16);
    }
    iacc = _mm256_sub_epi32(iacc, _mm256_set1_epi32(bias8));

    const __m256 col_scale_f32 = _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) blk.d), dmask));
    const __m256 row_scale_f32 = _mm256_set1_ps(d);
    return _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc), _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
}

// Timing-only: the instruction mix of ik_llama.cpp's Q4_0_R8 on the same 144-byte blocks (4-byte pieces, lanes = rows, no xor,
// no horizontal add, one madd). The result is not the dot product of anything: it only shows what a layout change could buy.
static void gemv_ik_mix(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int nb = n / QK8_0;
    UNUSED(bs);
    const __m256i m4b    = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    y_pre_t   stackbuf[512];
    y_pre_t * pre = nb <= 512 ? stackbuf : (y_pre_t *) malloc(sizeof(y_pre_t) * nb);
    const block_q4_0x8 * b_ptr_start = (const block_q4_0x8 *) vx;
    const block_q8_0   * a_ptr_start = (const block_q8_0 *) vy;
    for (int64_t y = 0; y < nr; y++) {
        const block_q8_0 * a_ptr = a_ptr_start + (y * nb);
        y_prepass(a_ptr, nb, pre);
        for (int64_t x = 0; x < nc / 8; x++) {
            const block_q4_0x8 * b_ptr = b_ptr_start + (x * nb);
            __m256 acc_row = _mm256_setzero_ps();
            for (int64_t b = 0; b < nb; b++) {
                const __m256i r0 = _mm256_loadu_si256((const __m256i *) (b_ptr[b].qs));
                const __m256i r1 = _mm256_loadu_si256((const __m256i *) (b_ptr[b].qs) + 1);
                const __m256i r2 = _mm256_loadu_si256((const __m256i *) (b_ptr[b].qs) + 2);
                const __m256i r3 = _mm256_loadu_si256((const __m256i *) (b_ptr[b].qs) + 3);
                int32_t q[8];
                memcpy(q, a_ptr[b].qs, 32);
                __m256i S = _mm256_maddubs_epi16(_mm256_and_si256(r0, m4b), _mm256_set1_epi32(q[0]));
                S = _mm256_add_epi16(S, _mm256_maddubs_epi16(_mm256_and_si256(r1, m4b), _mm256_set1_epi32(q[1])));
                S = _mm256_add_epi16(S, _mm256_maddubs_epi16(_mm256_and_si256(r2, m4b), _mm256_set1_epi32(q[2])));
                S = _mm256_add_epi16(S, _mm256_maddubs_epi16(_mm256_and_si256(r3, m4b), _mm256_set1_epi32(q[3])));
                S = _mm256_add_epi16(S, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(r0, 4), m4b), _mm256_set1_epi32(q[4])));
                S = _mm256_add_epi16(S, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(r1, 4), m4b), _mm256_set1_epi32(q[5])));
                S = _mm256_add_epi16(S, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(r2, 4), m4b), _mm256_set1_epi32(q[6])));
                S = _mm256_add_epi16(S, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(r3, 4), m4b), _mm256_set1_epi32(q[7])));
                __m256i iacc = _mm256_madd_epi16(S, ones16);
                iacc = _mm256_sub_epi32(iacc, _mm256_set1_epi32(pre[b].bias8));
                const __m256 col_scale_f32 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) b_ptr[b].d));
                acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc), _mm256_mul_ps(col_scale_f32, _mm256_set1_ps(pre[b].d)), acc_row);
            }
            _mm256_storeu_ps(s + (y * nr + x * 8), acc_row);
        }
    }
    if (pre != stackbuf) free(pre);
}

// Timing-only: the same kernel with the activation prepass done once per buffer instead of once per call, to see what the prepass costs.
static void gemv_u8_noprepass(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int nb = n / QK8_0;
    UNUSED(bs);
    UNUSED(nr);
    const __m256i unperm = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    static y_pre_t cache[4096];
    static const void * cached = nullptr;
    if (cached != vy) { y_prepass((const block_q8_0 *) vy, nb, cache); cached = vy; }
    const block_q4_0x8 * b_ptr_start = (const block_q4_0x8 *) vx;
    const block_q8_0   * a_ptr = (const block_q8_0 *) vy;
    for (int64_t x = 0; x < nc / 8; x++) {
        const block_q4_0x8 * b_ptr = b_ptr_start + (x * nb);
        __m256 acc_row = _mm256_setzero_ps();
        for (int64_t b = 0; b < nb; b++) {
            int64_t q0, q1, q2, q3;
            memcpy(&q0, a_ptr[b].qs +  0, 8);
            memcpy(&q1, a_ptr[b].qs +  8, 8);
            memcpy(&q2, a_ptr[b].qs + 16, 8);
            memcpy(&q3, a_ptr[b].qs + 24, 8);
            acc_row = u8_block<RED_PHADDW, 1>(b_ptr[b], _mm256_set1_epi64x(q0), _mm256_set1_epi64x(q1), _mm256_set1_epi64x(q2), _mm256_set1_epi64x(q3),
                                              cache[b].bias8, cache[b].d, acc_row);
        }
        _mm256_storeu_ps(s + x * 8, _mm256_permutevar8x32_ps(acc_row, unperm));
    }
}

// NG: column groups (8 rows each) per pass; PF: prefetch distance in blocks (0 = none)
template<int RED, int XOR, int NG, int PF>
static void gemv_u8(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int nb = n / QK8_0;
    UNUSED(bs);

    const __m256i unperm = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);

    y_pre_t   stackbuf[512];
    y_pre_t * pre = nb <= 512 ? stackbuf : (y_pre_t *) malloc(sizeof(y_pre_t) * nb);

    const block_q4_0x8 * b_ptr_start = (const block_q4_0x8 *) vx;
    const block_q8_0   * a_ptr_start = (const block_q8_0 *) vy;

    for (int64_t y = 0; y < nr; y++) {
        const block_q8_0 * a_ptr = a_ptr_start + (y * nb);
        y_prepass(a_ptr, nb, pre);

        int64_t x = 0;
        if constexpr (NG == 2) {
            for (; x + 1 < nc / 8; x += 2) {
                const block_q4_0x8 * b_ptr0 = b_ptr_start + (x * nb);
                const block_q4_0x8 * b_ptr1 = b_ptr0 + nb;
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                for (int64_t b = 0; b < nb; b++) {
                    if constexpr (PF > 0) {
                        _mm_prefetch((const char *) (b_ptr0 + b + PF), _MM_HINT_T0);
                        _mm_prefetch((const char *) (b_ptr0 + b + PF) + 64, _MM_HINT_T0);
                        _mm_prefetch((const char *) (b_ptr0 + b + PF) + 128, _MM_HINT_T0);
                        _mm_prefetch((const char *) (b_ptr1 + b + PF), _MM_HINT_T0);
                        _mm_prefetch((const char *) (b_ptr1 + b + PF) + 64, _MM_HINT_T0);
                        _mm_prefetch((const char *) (b_ptr1 + b + PF) + 128, _MM_HINT_T0);
                    }
                    int64_t q0, q1, q2, q3;
                    memcpy(&q0, a_ptr[b].qs +  0, 8);
                    memcpy(&q1, a_ptr[b].qs +  8, 8);
                    memcpy(&q2, a_ptr[b].qs + 16, 8);
                    memcpy(&q3, a_ptr[b].qs + 24, 8);
                    const __m256i y0 = _mm256_set1_epi64x(q0), y1 = _mm256_set1_epi64x(q1), y2 = _mm256_set1_epi64x(q2), y3 = _mm256_set1_epi64x(q3);
                    acc0 = u8_block<RED, XOR>(b_ptr0[b], y0, y1, y2, y3, pre[b].bias8, pre[b].d, acc0);
                    acc1 = u8_block<RED, XOR>(b_ptr1[b], y0, y1, y2, y3, pre[b].bias8, pre[b].d, acc1);
                }
                _mm256_storeu_ps(s + (y * nr + x * 8),     _mm256_permutevar8x32_ps(acc0, unperm));
                _mm256_storeu_ps(s + (y * nr + x * 8 + 8), _mm256_permutevar8x32_ps(acc1, unperm));
            }
        }
        for (; x < nc / 8; x++) {
            const block_q4_0x8 * b_ptr = b_ptr_start + (x * nb);
            __m256 acc_row = _mm256_setzero_ps();
            for (int64_t b = 0; b < nb; b++) {
                if constexpr (PF > 0) {
                    _mm_prefetch((const char *) (b_ptr + b + PF), _MM_HINT_T0);
                    _mm_prefetch((const char *) (b_ptr + b + PF) + 64, _MM_HINT_T0);
                    _mm_prefetch((const char *) (b_ptr + b + PF) + 128, _MM_HINT_T0);
                }
                int64_t q0, q1, q2, q3;
                memcpy(&q0, a_ptr[b].qs +  0, 8);
                memcpy(&q1, a_ptr[b].qs +  8, 8);
                memcpy(&q2, a_ptr[b].qs + 16, 8);
                memcpy(&q3, a_ptr[b].qs + 24, 8);
                acc_row = u8_block<RED, XOR>(b_ptr[b], _mm256_set1_epi64x(q0), _mm256_set1_epi64x(q1), _mm256_set1_epi64x(q2), _mm256_set1_epi64x(q3),
                                             pre[b].bias8, pre[b].d, acc_row);
            }
            _mm256_storeu_ps(s + (y * nr + x * 8), _mm256_permutevar8x32_ps(acc_row, unperm));
        }
    }
    if (pre != stackbuf) free(pre);
}
