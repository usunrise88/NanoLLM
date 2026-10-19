// ---- new GEMM candidates ----
struct q8_0x4_pre { float d[4]; int32_t sum8[4]; };

static inline void q8_0x4_prepare(const block_q8_0x4 * a, int nb, q8_0x4_pre * pre) {
    const __m256i ones8  = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (int b = 0; b < nb; b++) {
        const __m256i * q = (const __m256i *) a[b].qs;
        __m256i t0 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, _mm256_loadu_si256(q + 0)), ones16);
        __m256i t1 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, _mm256_loadu_si256(q + 1)), ones16);
        __m256i t2 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, _mm256_loadu_si256(q + 2)), ones16);
        __m256i t3 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, _mm256_loadu_si256(q + 3)), ones16);
        __m256i t  = _mm256_add_epi32(_mm256_add_epi32(t0, t1), _mm256_add_epi32(t2, t3));
        t = _mm256_hadd_epi32(t, t);
        const __m128i rows = _mm_unpacklo_epi64(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));
        _mm_storeu_si128((__m128i *) pre[b].sum8, _mm_slli_epi32(rows, 3));
        _mm_storeu_ps(pre[b].d, _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *) a[b].d)));
    }
}

// RED: 0 hadd_epi32, 1 shuffle+add, 2 hadd_epi16+madd;  XOR: undo the xor 0x88 of the stored nibbles;  RG: 4-row groups per tile
template<int RED, int XOR, int RG>
static void gemm_u8(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int nb = n / QK8_0;

    const __m256i m4b    = _mm256_set1_epi8(0x0F);
    const __m256i m88    = _mm256_set1_epi8((char) 0x88);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m128i scale_order = _mm_set_epi8(15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0);
    const __m256i lane_to_row = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);

    q8_0x4_pre   pre_stack[RG * 512];
    q8_0x4_pre * pre = nb <= 512 ? pre_stack : (q8_0x4_pre *) malloc(sizeof(q8_0x4_pre) * nb * RG);

    const block_q4_0x8  * b_ptr_start = (const block_q4_0x8  *) vx;
    const block_q8_0x4  * a_ptr_start = (const block_q8_0x4  *) vy;

    int64_t y = 0;
    while (y < nr / 4) {
        const int rg = (nr / 4 - y) < RG ? (int) (nr / 4 - y) : RG;
        const block_q8_0x4 * a_ptr[RG];
        for (int g = 0; g < rg; g++) {
            a_ptr[g] = a_ptr_start + ((y + g) * nb);
            q8_0x4_prepare(a_ptr[g], nb, pre + g * nb);
        }

        for (int64_t x = 0; x < nc / 8; x++) {
            const block_q4_0x8 * b_ptr = b_ptr_start + (x * nb);

            __m256 acc_rows[4 * RG];
            for (int i = 0; i < 4 * RG; i++) acc_rows[i] = _mm256_setzero_ps();

            for (int64_t b = 0; b < nb; b++) {
                __m256i raw_0123_0 = _mm256_loadu_si256((const __m256i *) (b_ptr[b].qs));
                __m256i raw_4567_0 = _mm256_loadu_si256((const __m256i *) (b_ptr[b].qs) + 1);
                __m256i raw_0123_1 = _mm256_loadu_si256((const __m256i *) (b_ptr[b].qs) + 2);
                __m256i raw_4567_1 = _mm256_loadu_si256((const __m256i *) (b_ptr[b].qs) + 3);
                if constexpr (XOR) {
                    raw_0123_0 = _mm256_xor_si256(raw_0123_0, m88);
                    raw_4567_0 = _mm256_xor_si256(raw_4567_0, m88);
                    raw_0123_1 = _mm256_xor_si256(raw_0123_1, m88);
                    raw_4567_1 = _mm256_xor_si256(raw_4567_1, m88);
                }

                const __m256i w0_0123 = _mm256_and_si256(raw_0123_0, m4b);
                const __m256i w0_4567 = _mm256_and_si256(raw_4567_0, m4b);
                const __m256i w1_0123 = _mm256_and_si256(raw_0123_1, m4b);
                const __m256i w1_4567 = _mm256_and_si256(raw_4567_1, m4b);
                const __m256i w2_0123 = _mm256_and_si256(_mm256_srli_epi16(raw_0123_0, 4), m4b);
                const __m256i w2_4567 = _mm256_and_si256(_mm256_srli_epi16(raw_4567_0, 4), m4b);
                const __m256i w3_0123 = _mm256_and_si256(_mm256_srli_epi16(raw_0123_1, 4), m4b);
                const __m256i w3_4567 = _mm256_and_si256(_mm256_srli_epi16(raw_4567_1, 4), m4b);

                const __m256 col_scale_f32 = _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) b_ptr[b].d), scale_order));

                #pragma GCC unroll 16
                for (int i = 0; i < 4 * RG; i++) {
                    if (i >= 4 * rg) break;
                    const int g = i >> 2, r = i & 3;
                    int64_t a0, a1, a2, a3;
                    memcpy(&a0, a_ptr[g][b].qs +      8 * r, 8);
                    memcpy(&a1, a_ptr[g][b].qs + 32 + 8 * r, 8);
                    memcpy(&a2, a_ptr[g][b].qs + 64 + 8 * r, 8);
                    memcpy(&a3, a_ptr[g][b].qs + 96 + 8 * r, 8);
                    const __m256i l0 = _mm256_set1_epi64x(a0), l1 = _mm256_set1_epi64x(a1), l2 = _mm256_set1_epi64x(a2), l3 = _mm256_set1_epi64x(a3);

                    __m256i S = _mm256_maddubs_epi16(w0_0123, l0);
                    __m256i T = _mm256_maddubs_epi16(w0_4567, l0);
                    S = _mm256_add_epi16(S, _mm256_maddubs_epi16(w1_0123, l1));
                    T = _mm256_add_epi16(T, _mm256_maddubs_epi16(w1_4567, l1));
                    S = _mm256_add_epi16(S, _mm256_maddubs_epi16(w2_0123, l2));
                    T = _mm256_add_epi16(T, _mm256_maddubs_epi16(w2_4567, l2));
                    S = _mm256_add_epi16(S, _mm256_maddubs_epi16(w3_0123, l3));
                    T = _mm256_add_epi16(T, _mm256_maddubs_epi16(w3_4567, l3));

                    __m256i iacc;
                    if constexpr (RED == 0) {
                        iacc = _mm256_hadd_epi32(_mm256_madd_epi16(S, ones16), _mm256_madd_epi16(T, ones16));
                    } else if constexpr (RED == 1) {
                        const __m256i mS = _mm256_madd_epi16(S, ones16);
                        const __m256i mT = _mm256_madd_epi16(T, ones16);
                        iacc = _mm256_add_epi32(
                            _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(mS), _mm256_castsi256_ps(mT), 0x88)),
                            _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(mS), _mm256_castsi256_ps(mT), 0xDD)));
                    } else {
                        iacc = _mm256_madd_epi16(_mm256_hadd_epi16(S, T), ones16);
                    }
                    iacc = _mm256_sub_epi32(iacc, _mm256_set1_epi32(pre[g * nb + b].sum8[r]));
                    const __m256 row_scale_f32 = _mm256_set1_ps(pre[g * nb + b].d[r]);
                    acc_rows[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc), _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_rows[i]);
                }
            }

            for (int i = 0; i < 4 * rg; i++) {
                const int g = i >> 2, r = i & 3;
                _mm256_storeu_ps((float *) (s + (((y + g) * 4 + r) * bs + x * 8)), _mm256_permutevar8x32_ps(acc_rows[i], lane_to_row));
            }
        }
        y += rg;
    }
    if (pre != pre_stack) free(pre);
}
