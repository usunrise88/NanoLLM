package bench

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/plan"
	"github.com/usunrise88/nanollm/internal/registry"
)

// Variant is one point of a development sweep: the settings it changes
// against the preset as planned. The standard suite never has one.
type Variant map[string]string

// Keys a sweep may set. Most go through the planner, so the plan stays
// consistent with them (a q8_0 cache is budgeted as q8_0); the rest are
// loader settings the planner does not choose per run.
var variantKeys = map[string]string{
	"threads":       "CPU threads for decoding (0 is one per physical core)",
	"threads_batch": "CPU threads for prompt processing (0 follows threads)",
	"n_batch":       "logical batch size",
	"n_ubatch":      "physical batch size",
	"kv":            "KV cache type for K and V: f16 or q8_0",
	"n_ctx":         "context per slot",
	"slots":         "parallel sequences",
	"batch":         "dev suite: also measure the throughput of 1, 2, 4… up to N sequences decoding at once, with and without MTP (raises slots to N)",
	"draft_max":     "MTP tokens drafted per step",
	"draft_p_min":   "MTP: stop drafting below this probability",
	"only":          "batch: run the throughput curve of the batch key and nothing else",
	"verify_rows":   "rows of one step spent on drafts and their checks, summed over the decoding sequences (0: no bound)",
	"mtp":           "on or off",
	"ngram":         "prompt-lookup drafting: the match length (2 or more), 0 is off; replaces the MTP drafter",
	"repack":        "on or off: CPU repacked weights (use_extra_bufts)",
	"mode":          "memory mode: auto, M1…M10 or a name",
	"load":          "none, mmap, mlock, mmap+mlock or dio",
	"lazy":          "off, auto or on",
	"fa":            "on, off or small: flash attention (small: on batches of up to 8 tokens only, the matmul attention for longer ones on the full-attention layers; the default where the GPU lists fa-small-batch)",
	"swa_full":      "on or off: off keeps the sliding-window layers at their window even when the full cache fits (auto is on when it fits)",
	"sampling":      "host or device: sampler chains and the MTP drafter's top-10 on the host or in the GPU graph (the default where the GPU lists backend-sampling)",
	"stream_ahead":  "layers read ahead of the compute in the file-backed modes M8, M9 (OPT-8); 0 is off",
	"stream_keep":   "leading layers the streamer never evicts",
	"stream_evict":  "on or off: drop a layer's pages once it is computed",
	"stream_method": "advise or pread: how the next layers are read (the system's readahead, or reads of the file at stream_depth)",
	"stream_depth":  "reads in flight of stream_method=pread (0 is four)",
	"numa":          "off, interleave or node:N",
	"layout":        "original or nano: the tensor order of the model file (OPT-7); nano is the copy `nanollm models relayout` wrote, and the plan is made from its offsets",
	"env.NAME":      "an environment variable for the worker (OMP_PROC_BIND, GGML_*)",
}

// EnvPrefix marks a sweep key that sets an environment variable of the
// worker: --set env.OMP_PROC_BIND=close,spread.
const EnvPrefix = "env."

// VariantKeys is the help text of --set.
func VariantKeys() string {
	keys := make([]string, 0, len(variantKeys))
	for k := range variantKeys {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, k := range keys {
		fmt.Fprintf(&b, "  %-12s %s\n", k, variantKeys[k])
	}
	return b.String()
}

// ParseSets expands "key=a,b" axes into their cartesian product, in the order
// given. No axes is one empty variant: the preset as planned.
func ParseSets(sets []string) ([]Variant, error) {
	out := []Variant{{}}
	seen := map[string]bool{}
	for _, s := range sets {
		key, vals, ok := strings.Cut(s, "=")
		key = strings.TrimSpace(key)
		if !ok || vals == "" {
			return nil, fmt.Errorf("--set %q: want key=value[,value…]", s)
		}
		if _, known := variantKeys[key]; !known && !strings.HasPrefix(key, EnvPrefix) {
			return nil, fmt.Errorf("--set: unknown key %q; keys:\n%s", key, VariantKeys())
		}
		if seen[key] {
			return nil, fmt.Errorf("--set: %s given twice", key)
		}
		seen[key] = true
		var next []Variant
		for _, base := range out {
			for _, v := range strings.Split(vals, ",") {
				v = strings.TrimSpace(v)
				if err := checkValue(key, v); err != nil {
					return nil, fmt.Errorf("--set %s: %w", key, err)
				}
				nv := Variant{}
				for k, x := range base {
					nv[k] = x
				}
				nv[key] = v
				next = append(next, nv)
			}
		}
		out = next
	}
	return out, nil
}

func checkValue(key, v string) error {
	if strings.HasPrefix(key, EnvPrefix) {
		return nil
	}
	switch key {
	case "only":
		if v != "batch" {
			return fmt.Errorf("%q is not batch", v)
		}
	case "batch":
		if n, err := strconv.Atoi(v); err != nil || n < 2 || n > 64 {
			return fmt.Errorf("%q is not a number of sequences from 2 to 64", v)
		}
	case "threads", "threads_batch", "n_batch", "n_ubatch", "n_ctx", "slots", "draft_max", "verify_rows", "stream_ahead", "stream_keep", "stream_depth", "ngram":
		n, err := strconv.Atoi(v)
		if err != nil || n < 0 {
			return fmt.Errorf("%q is not a non-negative integer", v)
		}
	case "draft_p_min":
		f, err := strconv.ParseFloat(v, 64)
		if err != nil || f < 0 || f > 1 {
			return fmt.Errorf("%q is not a probability", v)
		}
	case "kv":
		if v != "f16" && v != "q8_0" {
			return fmt.Errorf("%q is not f16 or q8_0", v)
		}
	case "fa":
		if v != "on" && v != "off" && v != "small" {
			return fmt.Errorf("%q is not on, off or small", v)
		}
	case "sampling":
		if v != "host" && v != "device" {
			return fmt.Errorf("%q is not host or device", v)
		}
	case "stream_method":
		if v != "advise" && v != "pread" {
			return fmt.Errorf("%q is not advise or pread", v)
		}
	case "mtp", "repack", "swa_full", "stream_evict":
		if v != "on" && v != "off" {
			return fmt.Errorf("%q is not on or off", v)
		}
	case "layout":
		if v != registry.LayoutOriginal && v != registry.LayoutNano {
			return fmt.Errorf("%q is not %s or %s", v, registry.LayoutOriginal, registry.LayoutNano)
		}
	case "mode":
		if _, err := plan.ParseMode(v); err != nil {
			return err
		}
	case "load":
		switch backend.LoadMode(v) {
		case backend.LoadNone, backend.LoadMMap, backend.LoadMLock, backend.LoadMMapMLock, backend.LoadDirectIO:
		default:
			return fmt.Errorf("%q is not none, mmap, mlock, mmap+mlock or dio", v)
		}
	case "lazy":
		switch backend.LazyMode(v) {
		case backend.LazyOff, backend.LazyAuto, backend.LazyOn:
		default:
			return fmt.Errorf("%q is not off, auto or on", v)
		}
	case "numa":
		if _, err := ParseNUMA(v); err != nil {
			return err
		}
	}
	return nil
}

// String is the canonical "k=v,k=v" form, keys sorted; it is the run's
// Variant label.
func (v Variant) String() string {
	keys := make([]string, 0, len(v))
	for k := range v {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, len(keys))
	for i, k := range keys {
		parts[i] = k + "=" + v[k]
	}
	return strings.Join(parts, ",")
}

func (v Variant) int(key string) (int, bool) {
	s, ok := v[key]
	if !ok {
		return 0, false
	}
	n, _ := strconv.Atoi(s)
	return n, true
}

// Plan applies the planner-level settings to the request.
func (v Variant) Plan(req *plan.Request) {
	if n, ok := v.int("n_batch"); ok {
		req.NBatch = n
	}
	if n, ok := v.int("n_ubatch"); ok {
		req.NUBatch = n
	}
	if n, ok := v.int("n_ctx"); ok {
		req.NCtx = n
	}
	if n, ok := v.int("slots"); ok {
		req.Slots = n
	}
	if n, ok := v.int("batch"); ok {
		req.Slots = max(req.Slots, n)
	}
	if n, ok := v.int("draft_max"); ok {
		req.DraftMax = n
	}
	if n, ok := v.int("verify_rows"); ok {
		req.VerifyRows = n
		if n == 0 {
			req.VerifyRows = -1 // in a sweep 0 is "no bound", over the CPU default
		}
	}
	if s, ok := v["draft_p_min"]; ok {
		req.DraftPMin, _ = strconv.ParseFloat(s, 64)
	}
	if s, ok := v["kv"]; ok {
		req.KVType = s
	}
	if v["mtp"] == "off" {
		req.NoMTP = true
	}
	if n, ok := v.int("ngram"); ok && n > 0 {
		req.NGram, req.NoMTP = n, true
		if req.DraftMax == 0 {
			req.DraftMax = 4
		}
	}
	switch v["fa"] {
	case "off":
		req.NoFlashAttention = true
	case "on":
		req.FlashAttnSmallBatch = false
	case "small":
		req.FlashAttnSmallBatch = true
	}
	switch v["sampling"] {
	case "host":
		req.BackendSampling = false
	case "device":
		req.BackendSampling = true
	}
	if v["repack"] == "off" {
		req.NoRepack = true
	}
	if v["swa_full"] == "off" {
		req.NoSWAFull = true
	}
	if s, ok := v["mode"]; ok {
		req.Mode, _ = plan.ParseMode(s)
	}
}

// Env is the environment variables the variant sets.
func (v Variant) Env() map[string]string {
	var env map[string]string
	for k, val := range v {
		if name, ok := strings.CutPrefix(k, EnvPrefix); ok {
			if env == nil {
				env = map[string]string{}
			}
			env[name] = val
		}
	}
	return env
}

// Load applies the loader-level settings to the spec the plan produced.
func (v Variant) Load(spec *backend.LoadSpec) {
	if n, ok := v.int("threads"); ok {
		spec.Threads = n
	}
	if n, ok := v.int("threads_batch"); ok {
		spec.ThreadsBatch = n
	}
	switch v["repack"] {
	case "on":
		spec.Repack = true
	case "off":
		spec.Repack = false
	}
	if s, ok := v["load"]; ok {
		spec.LoadMode = backend.LoadMode(s)
	}
	if s, ok := v["lazy"]; ok {
		spec.LazyMode = backend.LazyMode(s)
	}
	if spec.Stream != nil {
		st := *spec.Stream
		if n, ok := v.int("stream_ahead"); ok {
			st.Ahead = n
		}
		if n, ok := v.int("stream_keep"); ok {
			st.Keep = n
		}
		if s, ok := v["stream_method"]; ok {
			st.Method = s
		}
		if n, ok := v.int("stream_depth"); ok {
			st.Depth = n
		}
		switch v["stream_evict"] {
		case "on":
			st.Evict = true
		case "off":
			st.Evict = false
		}
		spec.Stream = &st
	}
}

// setsSpeculation says whether the variant names a speculation setting, in
// which case the CPU default of MTP (plan.DefaultCPUMTP) does not apply.
func (v Variant) setsSpeculation() bool {
	for _, k := range []string{"draft_max", "draft_p_min", "mtp", "ngram"} {
		if _, ok := v[k]; ok {
			return true
		}
	}
	return false
}
