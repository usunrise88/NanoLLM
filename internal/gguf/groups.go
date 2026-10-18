package gguf

import (
	"fmt"
	"strconv"
	"strings"
)

// Group is a unit of placement (SPEC §4): the planner decides per group, not
// per tensor, where weights live. G5 (KV cache) and G7 (compute buffers) are
// not tensors in the file; the planner computes them.
type Group string

const (
	G0 Group = "G0" // token embedding and LM head
	G1 Group = "G1" // per-layer embedding tables (PLE)
	G2 Group = "G2" // attention, plus the PLE projections (subgroup ple_proj)
	G3 Group = "G3" // FFN
	G4 Group = "G4" // norms and other small tensors
	G5 Group = "G5" // KV cache
	G6 Group = "G6" // MTP drafter
	G7 Group = "G7" // compute buffers
)

// Groups lists every group in order.
var Groups = []Group{G0, G1, G2, G3, G4, G5, G6, G7}

// SubPLEProj marks the G2 tensors that project the per-layer embeddings into
// the residual stream. They are read every token like attention, but they
// exist only because of PLE, so the inventory keeps them apart to answer the
// question "what does PLE cost" without double counting.
const SubPLEProj = "ple_proj"

// Architectures the classifier knows.
const (
	ArchGemma4          = "gemma4"
	ArchGemma4Assistant = "gemma4-assistant"
)

// Class is where a tensor belongs.
type Class struct {
	Group    Group
	Subgroup string
	// Layer is the repeating layer the tensor loads with, or -1 for the input
	// and output layers. It decides the device in a partial offload.
	Layer int
	// Input marks tensors llama.cpp assigns to the input layer, which it
	// always keeps in host memory whatever gpu_layers says (token_embd,
	// per_layer_token_embd).
	Input bool
	// Output marks tensors of the output layer, which goes to the GPU as soon
	// as one layer does.
	Output bool
}

// Classify assigns a tensor of a known architecture to its group. An unknown
// name is an error: placement by guess would put a tensor nobody accounted for
// on a device with no room for it.
func Classify(arch, name string) (Class, error) {
	switch arch {
	case ArchGemma4:
		return classifyGemma4(name)
	case ArchGemma4Assistant:
		c, err := classifyGemma4Assistant(name)
		if err != nil {
			return c, err
		}
		// The drafter is one group whatever its tensors are; the layer and
		// input/output roles still decide its placement.
		c.Group, c.Subgroup = G6, ""
		return c, nil
	}
	return Class{}, fmt.Errorf("tensor %q: no group table for architecture %q", name, arch)
}

// splitName parses "blk.<n>.<base>.<suffix>" or "<base>.<suffix>" into the
// layer (-1 when there is none) and the base name without the .weight/.bias
// suffix.
func splitName(name string) (layer int, base string, ok bool) {
	layer = -1
	rest := name
	if r, found := strings.CutPrefix(name, "blk."); found {
		num, tail, found := strings.Cut(r, ".")
		if !found {
			return 0, "", false
		}
		n, err := strconv.Atoi(num)
		if err != nil || n < 0 {
			return 0, "", false
		}
		layer, rest = n, tail
	}
	for _, suffix := range []string{".weight", ".bias", ".scale"} {
		if b, found := strings.CutSuffix(rest, suffix); found {
			return layer, b, true
		}
	}
	return 0, "", false
}

// isNorm matches the gemma4 norm names: attn_norm, attn_q_norm, ffn_norm,
// post_attention_norm, post_ffw_norm, post_norm and so on.
func isNorm(base string) bool {
	return base == "norm" || strings.HasSuffix(base, "_norm")
}

// classifyGemma4 follows docs/STAGE1.md 1.3. Layer assignment follows
// llama.cpp's gemma4 loader: per_layer_model_proj and per_layer_proj_norm
// have no blk prefix but are created with layer 0.
func classifyGemma4(name string) (Class, error) {
	layer, base, ok := splitName(name)
	if !ok {
		return Class{}, fmt.Errorf("tensor %q: not a gemma4 tensor name", name)
	}
	c := Class{Layer: layer}
	if layer < 0 {
		switch base {
		case "token_embd":
			c.Group, c.Input = G0, true
		case "output":
			c.Group, c.Output = G0, true
		case "per_layer_token_embd":
			c.Group, c.Input = G1, true
		case "per_layer_model_proj":
			c.Group, c.Subgroup, c.Layer = G2, SubPLEProj, 0
		case "per_layer_proj_norm":
			c.Group, c.Layer = G4, 0
		case "output_norm":
			c.Group, c.Output = G4, true
		case "rope_freqs":
			c.Group = G4
		default:
			return Class{}, fmt.Errorf("tensor %q: unknown gemma4 tensor", name)
		}
		return c, nil
	}
	switch {
	case base == "attn_q" || base == "attn_k" || base == "attn_v" || base == "attn_qkv" || base == "attn_output":
		c.Group = G2
	case base == "inp_gate" || base == "proj":
		c.Group, c.Subgroup = G2, SubPLEProj
	case base == "ffn_gate" || base == "ffn_up" || base == "ffn_down":
		c.Group = G3
	case isNorm(base) || base == "layer_output_scale" || base == "rope_freqs":
		c.Group = G4
	default:
		return Class{}, fmt.Errorf("tensor %q: unknown gemma4 tensor", name)
	}
	return c, nil
}

// classifyGemma4Assistant knows the drafter's tensors (llama.cpp
// gemma4-assistant loader). It reuses the target's token embedding for input,
// so its own token_embd serves only as the tied LM head; llama.cpp still
// loads it as an input tensor.
func classifyGemma4Assistant(name string) (Class, error) {
	layer, base, ok := splitName(name)
	if !ok {
		return Class{}, fmt.Errorf("tensor %q: not a gemma4-assistant tensor name", name)
	}
	c := Class{Layer: layer}
	if layer < 0 {
		switch base {
		case "token_embd", "masked_embd_centroids", "masked_embd_ordering":
			c.Input = true
		case "output_norm", "nextn.post_projection":
			c.Output = true
		case "nextn.pre_projection":
			c.Layer = 0
		case "rope_freqs":
		default:
			return Class{}, fmt.Errorf("tensor %q: unknown gemma4-assistant tensor", name)
		}
		return c, nil
	}
	switch {
	case base == "attn_q" || base == "attn_output",
		base == "ffn_gate" || base == "ffn_up" || base == "ffn_down",
		isNorm(base) || base == "layer_output_scale" || base == "rope_freqs":
	default:
		return Class{}, fmt.Errorf("tensor %q: unknown gemma4-assistant tensor", name)
	}
	return c, nil
}
