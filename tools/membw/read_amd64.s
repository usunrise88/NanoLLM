#include "textflag.h"

// func readAVX2(p unsafe.Pointer, blocks int) uint64
//
// Sums blocks*256 bytes with four 256-bit accumulators: eight loads per
// iteration, every load an AVX2 memory operand, nothing else in the loop.
// blocks must be positive.
TEXT ·readAVX2(SB), NOSPLIT, $0-24
	MOVQ p+0(FP), SI
	MOVQ blocks+8(FP), CX
	VPXOR Y0, Y0, Y0
	VPXOR Y1, Y1, Y1
	VPXOR Y2, Y2, Y2
	VPXOR Y3, Y3, Y3

loop:
	VPADDQ 0(SI), Y0, Y0
	VPADDQ 32(SI), Y1, Y1
	VPADDQ 64(SI), Y2, Y2
	VPADDQ 96(SI), Y3, Y3
	VPADDQ 128(SI), Y0, Y0
	VPADDQ 160(SI), Y1, Y1
	VPADDQ 192(SI), Y2, Y2
	VPADDQ 224(SI), Y3, Y3
	ADDQ $256, SI
	DECQ CX
	JNZ  loop

	VPADDQ Y1, Y0, Y0
	VPADDQ Y3, Y2, Y2
	VPADDQ Y2, Y0, Y0
	VEXTRACTI128 $1, Y0, X1
	VPADDQ X1, X0, X0
	VPEXTRQ $0, X0, AX
	VPEXTRQ $1, X0, BX
	ADDQ BX, AX
	VZEROUPPER
	MOVQ AX, ret+16(FP)
	RET
