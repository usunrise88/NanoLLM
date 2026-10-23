package plan

import "testing"

func TestDefaultCPUMTP(t *testing.T) {
	var r Request
	r.DefaultCPUMTP(true)
	if r.NoMTP || r.DraftMax != CPUMTPDraftMax || r.DraftPMin != CPUMTPDraftPMin || r.VerifyRows != 1+CPUMTPDraftMax {
		t.Errorf("with the multi-row kernels: %+v", r)
	}
	r = Request{DraftMax: 4}
	r.DefaultCPUMTP(true)
	if r.DraftMax != 4 || r.DraftPMin != CPUMTPDraftPMin || r.VerifyRows != 5 {
		t.Errorf("a draft_max the request carries stays, p_min gets the default, the rows follow draft_max: %+v", r)
	}
	r = Request{VerifyRows: 8}
	r.DefaultCPUMTP(true)
	if r.VerifyRows != 8 {
		t.Errorf("a bound the request carries stays: %+v", r)
	}
	r = Request{}
	r.DefaultCPUMTP(false)
	if !r.NoMTP || r.NoMTPReason != CPUMTPOffReason {
		t.Errorf("without the kernels MTP is off with the reason: %+v", r)
	}
}
