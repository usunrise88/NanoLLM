package plan

// MTP on the CPU pays where checking a draft is cheap. The stage 2 sweep
// (docs/STAGE2.md 2.7, docs/AUTOTUNE.md) found at best 1.04x with the kernels
// that check 2-3 rows one by one, and 1.35-1.41x (16 threads) / 1.21-1.25x
// (4 cores) with the multi-row Q4_0 kernels (threshold 1.2x). So it is on by
// default where the device lists them, with the setting that passes on both
// profiles, and off elsewhere with the reason in the plan. The server and the
// benchmark both use this, so what the benchmark measures is what a server
// would run.

// CPUMTPOffReason is the plan's reason for the default of MTP on the CPU.
const CPUMTPOffReason = "off by default on the CPU: without the multi-row Q4_0 kernels the measured speed-up is at best 1.04x on a 16-core EPYC 7351P (threshold 1.2x, docs/STAGE2.md 2.7); models.<id>.speculative.type: mtp turns it on"

// The MTP setting for a CPU with the multi-row kernels (docs/AUTOTUNE.md, "MTP на CPU
// с ядрами OPT-16"): 1.35-1.41x on 16 threads and 1.21-1.25x on 4 cores for draft_max 2 and
// p_min 0.5-0.75. On the i7-12700H with the AVX-VNNI kernels (gemma.json, 5 requests) p_min 0 /
// 0.3 / 0.5 / 0.75 took 130.5 / 123.6 / 125.8 / 129.8 s, so 0.5, the lower end of the range.
const (
	CPUMTPDraftMax  = 2
	CPUMTPDraftPMin = 0.5
)

// DefaultCPUMTP sets the MTP default of a plan for the CPU. multiRow says
// whether the device lists the multi-row Q4_0 kernels (backend.FeatureMultiRowQ4_0).
// A value the request already carries (draft_max, p_min) stays.
func (r *Request) DefaultCPUMTP(multiRow bool) {
	if !multiRow {
		r.NoMTP, r.NoMTPReason = true, CPUMTPOffReason
		return
	}
	if r.DraftMax == 0 {
		r.DraftMax = CPUMTPDraftMax
	}
	if r.DraftPMin == 0 {
		r.DraftPMin = CPUMTPDraftPMin
	}
	if r.VerifyRows == 0 {
		// Drafts only while one sequence decodes. With two or more the
		// checks of all of them cost more than the drafts save: 0.7-0.9x of
		// plain decoding on both profiles of the EPYC (docs/AUTOTUNE.md,
		// "MTP и несколько последовательностей").
		r.VerifyRows = 1 + r.DraftMax
	}
}

// HeadBytes is what a shortlist of rows rows costs in memory: a Q4_0 row of
// nEmbd columns is nEmbd/32 blocks of 18 bytes (registry.HeadRowBytes).
func HeadBytes(rows, nEmbd int) int64 { return int64(rows) * int64(nEmbd/32) * 18 }

// headRows is the number of rows of the shortlist the request carries, 0 for
// the whole head or when the model's head is not a tied one (the copy is made
// from the token embedding).
func (pl *planner) headRows() int {
	if pl.req.Head == "" || pl.req.Head == "full" || !pl.h.TiedOutput {
		return 0
	}
	return len(pl.req.HeadIDs)
}
