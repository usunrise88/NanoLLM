// Throughput and FP-pipe uop count of single instructions on this CPU.  ./uops NAME  (run under perf stat)
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <immintrin.h>

alignas(64) static uint8_t buf[256];

#define REP8(ins) ins(ymm8, ymm0, ymm1) ins(ymm9, ymm0, ymm1) ins(ymm10, ymm0, ymm1) ins(ymm11, ymm0, ymm1) \
                  ins(ymm12, ymm0, ymm1) ins(ymm13, ymm0, ymm1) ins(ymm14, ymm0, ymm1) ins(ymm15, ymm0, ymm1)

#define RRR(op) [](long n) { for (long i = 0; i < n; i++) { __asm__ volatile( \
   REP8(op##_r) : : "r"(buf) : "xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15","memory"); } }

#define STR(...) #__VA_ARGS__
#define OP3(name, text) \
   static void name(long n) { for (long i = 0; i < n; i++) { __asm__ volatile( \
     text(%%ymm8) text(%%ymm9) text(%%ymm10) text(%%ymm11) text(%%ymm12) text(%%ymm13) text(%%ymm14) text(%%ymm15) \
     : : "r"(buf) : "xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15","memory"); } }

#define T_maddubs(d) "vpmaddubsw %%ymm1, %%ymm0, " #d "\n"
#define T_maddwd(d) "vpmaddwd %%ymm1, %%ymm0, " #d "\n"
#define T_addw(d) "vpaddw %%ymm1, %%ymm0, " #d "\n"
#define T_and(d) "vpand %%ymm1, %%ymm0, " #d "\n"
#define T_srlw(d) "vpsrlw $4, %%ymm0, " #d "\n"
#define T_pshufb(d) "vpshufb %%ymm1, %%ymm0, " #d "\n"
#define T_phaddd(d) "vphaddd %%ymm1, %%ymm0, " #d "\n"
#define T_phaddw(d) "vphaddw %%ymm1, %%ymm0, " #d "\n"
#define T_pshufd(d) "vpshufd $0x4e, %%ymm0, " #d "\n"
#define T_shufps(d) "vshufps $0x88, %%ymm1, %%ymm0, " #d "\n"
#define T_blendd(d) "vpblendd $0xaa, %%ymm1, %%ymm0, " #d "\n"
#define T_bcastq(d) "vpbroadcastq 8(%0), " #d "\n"
#define T_bcastd(d) "vpbroadcastd 4(%0), " #d "\n"
#define T_bcastss(d) "vbroadcastss 4(%0), " #d "\n"
#define T_bcasti128(d) "vbroadcasti128 16(%0), " #d "\n"
#define T_permq(d) "vpermq $0x44, %%ymm0, " #d "\n"
#define T_perm2i(d) "vperm2i128 $0x20, %%ymm1, %%ymm0, " #d "\n"
#define T_cvtdq2ps(d) "vcvtdq2ps %%ymm0, " #d "\n"
#define T_cvtph2ps(d) "vcvtph2ps %%xmm0, " #d "\n"
#define T_mulps(d) "vmulps %%ymm1, %%ymm0, " #d "\n"
#define T_fma(d) "vfmadd231ps %%ymm1, %%ymm0, " #d "\n"
#define T_subd(d) "vpsubd %%ymm1, %%ymm0, " #d "\n"
#define T_xor(d) "vpxor %%ymm1, %%ymm0, " #d "\n"
#define T_load(d) "vmovdqu 32(%0), " #d "\n"
#define T_xorm(d) "vpxor 32(%0), %%ymm0, " #d "\n"
#define T_maddubsm(d) "vpmaddubsw 32(%0), %%ymm0, " #d "\n"
#define T_unpck(d) "vpunpcklqdq %%ymm1, %%ymm0, " #d "\n"
#define T_palignr(d) "vpalignr $8, %%ymm1, %%ymm0, " #d "\n"
#define T_extract(d) "vextracti128 $1, %%ymm0, %%xmm8\n"
#define T_pmovzx(d) "vpmovzxbw %%xmm0, " #d "\n"
#define T_sad(d) "vpsadbw %%ymm1, %%ymm0, " #d "\n"
#define T_permd(d) "vpermd %%ymm1, %%ymm0, " #d "\n"
#define T_mullw(d) "vpmullw %%ymm1, %%ymm0, " #d "\n"
#define T_adds(d) "vpaddsw %%ymm1, %%ymm0, " #d "\n"
#define T_addd(d) "vpaddd %%ymm1, %%ymm0, " #d "\n"
#define T_pavgb(d) "vpavgb %%ymm1, %%ymm0, " #d "\n"
#define T_movmsk(d) "vpmovmskb %%ymm0, %%eax\n"

OP3(f_maddubs, T_maddubs) OP3(f_maddwd, T_maddwd) OP3(f_addw, T_addw) OP3(f_and, T_and) OP3(f_srlw, T_srlw)
OP3(f_pshufb, T_pshufb) OP3(f_phaddd, T_phaddd) OP3(f_phaddw, T_phaddw) OP3(f_pshufd, T_pshufd) OP3(f_shufps, T_shufps)
OP3(f_blendd, T_blendd) OP3(f_bcastq, T_bcastq) OP3(f_bcastd, T_bcastd) OP3(f_bcastss, T_bcastss) OP3(f_bcasti128, T_bcasti128)
OP3(f_permq, T_permq) OP3(f_perm2i, T_perm2i) OP3(f_cvtdq2ps, T_cvtdq2ps) OP3(f_cvtph2ps, T_cvtph2ps) OP3(f_mulps, T_mulps)
OP3(f_fma, T_fma) OP3(f_subd, T_subd) OP3(f_xor, T_xor) OP3(f_load, T_load) OP3(f_xorm, T_xorm) OP3(f_maddubsm, T_maddubsm)
OP3(f_unpck, T_unpck) OP3(f_palignr, T_palignr) OP3(f_pmovzx, T_pmovzx) OP3(f_sad, T_sad) OP3(f_permd, T_permd)
OP3(f_mullw, T_mullw) OP3(f_adds, T_adds) OP3(f_addd, T_addd) OP3(f_pavgb, T_pavgb)

struct E { const char * n; void (*f)(long); };
#define E_(x) {#x, f_##x}
static E tab[] = {E_(maddubs), E_(maddwd), E_(addw), E_(and), E_(srlw), E_(pshufb), E_(phaddd), E_(phaddw), E_(pshufd), E_(shufps), E_(blendd),
    E_(bcastq), E_(bcastd), E_(bcastss), E_(bcasti128), E_(permq), E_(perm2i), E_(cvtdq2ps), E_(cvtph2ps), E_(mulps), E_(fma), E_(subd), E_(xor),
    E_(load), E_(xorm), E_(maddubsm), E_(unpck), E_(palignr), E_(pmovzx), E_(sad), E_(permd), E_(mullw), E_(adds), E_(addd), E_(pavgb)};

int main(int argc, char ** argv) {
    for (int i = 0; i < 256; i++) buf[i] = i;
    for (auto & e : tab) if (!strcmp(e.n, argv[1])) { e.f(argc > 2 ? atol(argv[2]) : 20000000L); return 0; }
    return 1;
}
