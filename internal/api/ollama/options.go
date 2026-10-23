package ollama

import (
	"bytes"
	"encoding/json"
	"errors"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/usunrise88/nanollm/internal/core"
)

// options is a request's "options" object as far as this server applies it.
type options struct {
	core.Sampling
	NumPredict int
	Stop       []string
	// Ignored names the options that were sent and not applied, in order.
	Ignored []string
}

// quiet options tune the runtime Ollama runs on. NanoLLM plans placement, threads
// and batching itself, so a client that sends them (the Ollama CLI's /set
// parameter does) is not told they were left out.
var quiet = map[string]bool{
	"num_gpu": true, "main_gpu": true, "low_vram": true, "num_thread": true, "num_batch": true,
	"use_mmap": true, "use_mlock": true, "numa": true, "vocab_only": true, "f16_kv": true,
	"logits_all": true, "embedding_only": true,
}

func parseOptions(raw json.RawMessage) (options, error) {
	var out options
	if isNull(raw) {
		return out, nil
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return out, core.Errorf(core.CodeInvalidRequest, "options must be an object").WithParam("options")
	}
	names := make([]string, 0, len(m))
	for k := range m {
		names = append(names, k)
	}
	slices.Sort(names)

	var firstErr error
	fail := func(name, want string) {
		if firstErr == nil {
			firstErr = core.Errorf(core.CodeInvalidRequest, "invalid options: %s must be %s", name, want).WithParam("options")
		}
	}
	for _, name := range names {
		v := m[name]
		if isNull(v) {
			continue
		}
		f := func(dst **float64) {
			x, ok := number(v)
			if !ok {
				fail(name, "a number")
				return
			}
			*dst = &x
		}
		i := func(dst **int) {
			x, ok := integer(v)
			if !ok {
				fail(name, "an integer")
				return
			}
			n := int(x)
			*dst = &n
		}
		switch name {
		case "temperature":
			f(&out.Temperature)
		case "top_p":
			f(&out.TopP)
		case "min_p":
			f(&out.MinP)
		case "repeat_penalty":
			f(&out.RepeatPenalty)
		case "presence_penalty":
			f(&out.PresencePenalty)
		case "frequency_penalty":
			f(&out.FrequencyPenalty)
		case "top_k":
			i(&out.TopK)
		case "seed":
			x, ok := integer(v)
			if !ok {
				fail(name, "an integer")
				continue
			}
			out.Seed = &x
		case "num_predict":
			var n *int
			i(&n)
			// -1 is Ollama's "until done" and -2 "fill the context": both are
			// the server's cap here, which is what zero asks for.
			if n != nil && *n > 0 {
				out.NumPredict = *n
			}
		case "stop":
			stops, err := stopList(v)
			if err != nil {
				fail(name, "a string or an array of strings")
				continue
			}
			out.Stop = stops
		default:
			if !quiet[name] {
				out.Ignored = append(out.Ignored, name)
			}
		}
	}
	return out, firstErr
}

func isNull(raw json.RawMessage) bool {
	t := bytes.TrimSpace(raw)
	return len(t) == 0 || string(t) == "null"
}

// number reads a JSON number.
func number(raw json.RawMessage) (float64, bool) {
	var n json.Number
	if err := json.Unmarshal(raw, &n); err != nil {
		return 0, false
	}
	x, err := n.Float64()
	return x, err == nil && !math.IsInf(x, 0)
}

// integer reads a whole number. Clients send 40.0 for 40 (a JSON encoder that
// holds every number as a float does), which Go's decoder refuses for an int.
func integer(raw json.RawMessage) (int64, bool) {
	var n json.Number
	if err := json.Unmarshal(raw, &n); err != nil {
		return 0, false
	}
	if x, err := n.Int64(); err == nil {
		return x, true
	}
	x, err := n.Float64()
	if err != nil || x != math.Trunc(x) || math.Abs(x) > 1<<53 {
		return 0, false
	}
	return int64(x), true
}

func stopList(raw json.RawMessage) ([]string, error) {
	var one string
	if err := json.Unmarshal(raw, &one); err == nil {
		return []string{one}, nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err != nil {
		return nil, errors.New("stop must be a string or an array of strings")
	}
	return many, nil
}

// warning is the notice about options left out, nil when there are none.
func (o options) warning() []core.Warning {
	if len(o.Ignored) == 0 {
		return nil
	}
	return []core.Warning{{Code: "option_ignored", Message: "options not applied by this server: " + strings.Join(o.Ignored, ", ")}}
}

// --- think, format, keep_alive ---

// parseThink reads "think": a bool, or one of the effort levels some models
// take, which switch thinking on here. Absent means the server's setting.
func parseThink(raw json.RawMessage) (*bool, error) {
	if isNull(raw) {
		return nil, nil
	}
	var b bool
	if err := json.Unmarshal(raw, &b); err == nil {
		return &b, nil
	}
	var level string
	if err := json.Unmarshal(raw, &level); err == nil {
		switch level {
		case "low", "medium", "high":
			on := true
			return &on, nil
		}
	}
	return nil, core.Errorf(core.CodeInvalidRequest, `think must be true, false, "low", "medium" or "high"`).WithParam("think")
}

// parseFormat reads "format": "json", or a JSON Schema object.
func parseFormat(raw json.RawMessage) (*core.ResponseFormat, error) {
	if isNull(raw) {
		return nil, nil
	}
	var kind string
	if err := json.Unmarshal(raw, &kind); err == nil {
		switch kind {
		case "":
			return nil, nil
		case "json":
			return &core.ResponseFormat{Type: core.FormatJSONObject}, nil
		}
		return nil, core.Errorf(core.CodeInvalidRequest, `invalid format %q: use "json" or a JSON Schema`, kind).WithParam("format")
	}
	var schema map[string]json.RawMessage
	if err := json.Unmarshal(raw, &schema); err != nil {
		return nil, core.Errorf(core.CodeInvalidRequest, `format must be "json" or a JSON Schema object`).WithParam("format")
	}
	return &core.ResponseFormat{Type: core.FormatJSONSchema, Schema: raw}, nil
}

// keepAlive is what a request said about how long the model should stay loaded.
type keepAlive struct {
	// Forever: a negative value. Unload: zero.
	Forever, Unload bool
}

// parseKeepAlive reads a number of seconds or a duration string such as "5m".
func parseKeepAlive(raw json.RawMessage) (keepAlive, error) {
	var ka keepAlive
	if isNull(raw) {
		return ka, nil
	}
	var d time.Duration
	var secs float64
	var text string
	switch {
	case json.Unmarshal(raw, &secs) == nil:
		d = time.Duration(secs * float64(time.Second))
		if secs < 0 {
			d = -1
		}
	case json.Unmarshal(raw, &text) == nil:
		if parsed, err := time.ParseDuration(text); err == nil {
			d = parsed
		} else if n, err := strconv.ParseFloat(text, 64); err == nil {
			d = time.Duration(n * float64(time.Second))
			if n < 0 {
				d = -1
			}
		} else {
			return ka, core.Errorf(core.CodeInvalidRequest, "invalid keep_alive %q: use seconds or a duration such as \"5m\"", text).WithParam("keep_alive")
		}
	default:
		return ka, core.Errorf(core.CodeInvalidRequest, "keep_alive must be a number of seconds or a duration string").WithParam("keep_alive")
	}
	ka.Forever, ka.Unload = d < 0, d == 0
	return ka, nil
}
