package gguf

import (
	"bytes"
	"testing"
)

func TestClassifyGemma4(t *testing.T) {
	for name, want := range map[string]Class{
		"token_embd.weight":                {Group: G0, Layer: -1, Input: true},
		"output.weight":                    {Group: G0, Layer: -1, Output: true},
		"per_layer_token_embd.weight":      {Group: G1, Layer: -1, Input: true},
		"per_layer_model_proj.weight":      {Group: G2, Subgroup: SubPLEProj, Layer: 0},
		"per_layer_proj_norm.weight":       {Group: G4, Layer: 0},
		"output_norm.weight":               {Group: G4, Layer: -1, Output: true},
		"rope_freqs.weight":                {Group: G4, Layer: -1},
		"blk.0.attn_q.weight":              {Group: G2, Layer: 0},
		"blk.41.attn_output.weight":        {Group: G2, Layer: 41},
		"blk.3.attn_qkv.weight":            {Group: G2, Layer: 3},
		"blk.7.inp_gate.weight":            {Group: G2, Subgroup: SubPLEProj, Layer: 7},
		"blk.7.proj.weight":                {Group: G2, Subgroup: SubPLEProj, Layer: 7},
		"blk.2.ffn_down.weight":            {Group: G3, Layer: 2},
		"blk.2.post_attention_norm.weight": {Group: G4, Layer: 2},
		"blk.2.post_ffw_norm.weight":       {Group: G4, Layer: 2},
		"blk.2.post_norm.weight":           {Group: G4, Layer: 2},
		"blk.2.attn_k_norm.weight":         {Group: G4, Layer: 2},
		"blk.2.layer_output_scale.weight":  {Group: G4, Layer: 2},
	} {
		got, err := Classify(ArchGemma4, name)
		if err != nil || got != want {
			t.Errorf("%s: %+v, %v; want %+v", name, got, err, want)
		}
	}
	for _, name := range []string{
		"blk.0.ffn_gate_exps.weight", "blk.x.attn_q.weight", "blk.0", "mystery.weight",
		"token_embd", "nextn.pre_projection.weight",
	} {
		if _, err := Classify(ArchGemma4, name); err == nil {
			t.Errorf("%s: classified", name)
		}
	}
	if _, err := Classify("llama", "token_embd.weight"); err == nil {
		t.Error("an architecture without a table classified a tensor")
	}
}

func TestClassifyDrafter(t *testing.T) {
	for name, want := range map[string]Class{
		"token_embd.weight":            {Group: G6, Layer: -1, Input: true},
		"nextn.post_projection.weight": {Group: G6, Layer: -1, Output: true},
		"nextn.pre_projection.weight":  {Group: G6, Layer: 0},
		"blk.3.attn_q.weight":          {Group: G6, Layer: 3},
		"blk.3.ffn_up.weight":          {Group: G6, Layer: 3},
		"rope_freqs.weight":            {Group: G6, Layer: -1},
	} {
		got, err := Classify(ArchGemma4Assistant, name)
		if err != nil || got != want {
			t.Errorf("%s: %+v, %v; want %+v", name, got, err, want)
		}
	}
	// The drafter has no K/V projections of its own: it reads the target's
	// cache. A file that has them is not the drafter the planner sized.
	if _, err := Classify(ArchGemma4Assistant, "blk.0.attn_k.weight"); err == nil {
		t.Error("drafter K projection classified")
	}
}

func TestBuildInventory(t *testing.T) {
	b, err := TinyGemma4().Bytes()
	if err != nil {
		t.Fatal(err)
	}
	f, err := Parse(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	f.FileSize = int64(len(b))
	inv, err := BuildInventory(f)
	if err != nil {
		t.Fatal(err)
	}
	if err := inv.Check(); err != nil {
		t.Fatal(err)
	}
	var sum int64
	for _, g := range Groups {
		sum += inv.Groups[g]
	}
	if sum != inv.Total() || inv.Groups[G1] == 0 || inv.SumSub(G2, SubPLEProj) == 0 {
		t.Errorf("groups %v, total %d", inv.Groups, inv.Total())
	}

	// A hand edit is caught.
	edited := inv
	edited.Tensors = append([]Tensor(nil), inv.Tensors...)
	edited.Tensors[0].Group = G3
	if err := edited.Check(); err == nil {
		t.Error("a misclassified tensor passed the check")
	}
	edited = inv
	edited.DataSize++
	if err := edited.Check(); err == nil {
		t.Error("a wrong data area passed the check")
	}

	// An unknown tensor is listed without a group and reported.
	w := TinyGemma4().Tensor("mystery.weight", TypeF32, []int64{4}, nil)
	b, _ = w.Bytes()
	f, err = Parse(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	inv, err = BuildInventory(f)
	if err == nil {
		t.Fatal("unknown tensor not reported")
	}
	if last := inv.Tensors[len(inv.Tensors)-1]; last.Name != "mystery.weight" || last.Group != "" {
		t.Errorf("unknown tensor recorded as %+v", last)
	}
}
