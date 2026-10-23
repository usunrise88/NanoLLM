// Package eval runs task suites through core.Service and scores the answers
// (ROADMAP «Гейт качества», STAGE1 1.10).
//
// A suite is JSONL, one item per line: a conversation, optionally tools or a
// response format, a capability label and a checker. The runner generates
// greedily, so a report is reproducible on one configuration, and writes an
// eval.v1 report. Checkers judge only what a machine can judge — a regex, a
// tool call, a JSON shape, a needle in a long context; anything else is
// marked for a person and rated with `nanollm eval rate`.
//
// No conclusions are drawn here. Reports of two configurations are compared
// by the gate per capability, and a person decides where checkers cannot.
package eval

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/usunrise88/nanollm/internal/core"
)

// ReportSchema identifies the report format.
const ReportSchema = "nanollm.eval/v1"

// Item is one task of a suite.
type Item struct {
	ID         string               `json:"id"`
	Lang       string               `json:"lang"`
	Capability string               `json:"capability"`
	Messages   []core.Message       `json:"messages"`
	Tools      []core.Tool          `json:"tools,omitempty"`
	Response   *core.ResponseFormat `json:"response_format,omitempty"`
	Checker    Checker              `json:"checker"`
	// Thinking, when set, fixes the thinking mode for this item; otherwise
	// the run's mode applies.
	Thinking  *bool `json:"thinking,omitempty"`
	MaxTokens int   `json:"max_tokens,omitempty"`
}

// Checker types.
const (
	CheckContains = "contains" // any of Any (case-insensitive), none of None
	CheckRegex    = "regex"    // Pattern matches the answer
	CheckExact    = "exact"    // the trimmed answer equals Want
	CheckJSON     = "json"     // the answer is JSON; with Schema, it fits it
	CheckToolCall = "tool_call"
	CheckNeedle   = "needle" // Want appears in the answer
	CheckHuman    = "human"  // a person rates it
)

// Checker says how an answer is judged.
type Checker struct {
	Type    string          `json:"type"`
	Any     []string        `json:"any,omitempty"`
	None    []string        `json:"none,omitempty"`
	Pattern string          `json:"pattern,omitempty"`
	Want    string          `json:"want,omitempty"`
	Schema  json.RawMessage `json:"schema,omitempty"`
	// Name and Args check a tool call: the tool, and argument values that
	// must be present (a subset match).
	Name string         `json:"name,omitempty"`
	Args map[string]any `json:"args,omitempty"`
}

// ReadSuite parses a JSONL suite. Blank lines and lines starting with // are
// skipped, so a suite can carry notes.
func ReadSuite(r io.Reader) ([]Item, string, error) {
	h := sha256.New()
	sc := bufio.NewScanner(io.TeeReader(r, h))
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	var items []Item
	seen := map[string]bool{}
	line := 0
	for sc.Scan() {
		line++
		b := bytes.TrimSpace(sc.Bytes())
		if len(b) == 0 || bytes.HasPrefix(b, []byte("//")) {
			continue
		}
		var it Item
		dec := json.NewDecoder(bytes.NewReader(b))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&it); err != nil {
			return nil, "", fmt.Errorf("suite line %d: %w", line, err)
		}
		if it.ID == "" || seen[it.ID] {
			return nil, "", fmt.Errorf("suite line %d: missing or duplicate id %q", line, it.ID)
		}
		seen[it.ID] = true
		if len(it.Messages) == 0 {
			return nil, "", fmt.Errorf("suite line %d (%s): no messages", line, it.ID)
		}
		if err := it.Checker.validate(); err != nil {
			return nil, "", fmt.Errorf("suite line %d (%s): %w", line, it.ID, err)
		}
		items = append(items, it)
	}
	if err := sc.Err(); err != nil {
		return nil, "", err
	}
	return items, hex.EncodeToString(h.Sum(nil)), nil
}

func (c Checker) validate() error {
	switch c.Type {
	case CheckContains:
		if len(c.Any) == 0 && len(c.None) == 0 {
			return fmt.Errorf("contains needs any or none")
		}
	case CheckRegex:
		if _, err := regexp.Compile(c.Pattern); err != nil {
			return fmt.Errorf("regex: %w", err)
		}
	case CheckExact, CheckNeedle:
		if c.Want == "" {
			return fmt.Errorf("%s needs want", c.Type)
		}
	case CheckToolCall:
		if c.Name == "" {
			return fmt.Errorf("tool_call needs name")
		}
	case CheckJSON, CheckHuman:
	default:
		return fmt.Errorf("unknown checker %q", c.Type)
	}
	return nil
}

// Outcome is one item's result.
type Outcome struct {
	ID         string `json:"id"`
	Lang       string `json:"lang"`
	Capability string `json:"capability"`
	Thinking   bool   `json:"thinking"`
	// Pass is nil for an item a person has to rate.
	Pass      *bool           `json:"pass,omitempty"`
	Detail    string          `json:"detail,omitempty"`
	Answer    string          `json:"answer"`
	Reasoning string          `json:"reasoning,omitempty"`
	ToolCalls []core.ToolCall `json:"tool_calls,omitempty"`
	Finish    string          `json:"finish"`
	Usage     core.Usage      `json:"usage"`
	MS        float64         `json:"ms"`
	// TTFTMS is the time to the first token.
	TTFTMS float64 `json:"ttft_ms,omitempty"`
	Error  string  `json:"error,omitempty"`
}

// Report is eval.v1.
type Report struct {
	Schema  string             `json:"schema"`
	Suite   SuiteRef           `json:"suite"`
	Model   string             `json:"model"`
	Config  map[string]string  `json:"config,omitempty"`
	Date    string             `json:"date"`
	Results []Outcome          `json:"results"`
	Summary map[string]Tally   `json:"summary"`
	ByLang  map[string]Tally   `json:"by_lang"`
	Ratings map[string]Rating  `json:"ratings,omitempty"`
	Extra   map[string]float64 `json:"extra,omitempty"`
}

type SuiteRef struct {
	Name   string `json:"name"`
	SHA256 string `json:"sha256"`
	Items  int    `json:"items"`
}

// Tally counts machine-checked items per capability; Human counts the ones
// left to a person.
type Tally struct {
	N     int     `json:"n"`
	Pass  int     `json:"pass"`
	Rate  float64 `json:"rate"`
	Human int     `json:"human,omitempty"`
}

// Options steer a run.
type Options struct {
	Model string
	// Thinking runs every item with thinking off, on, or both (as two
	// outcomes); an item's own Thinking wins.
	Thinking  string // off | on | both
	MaxTokens int
	// MaxReasoning closes a thought after this many tokens, so the answer
	// follows (0: no cap). It only matters where thinking is on.
	MaxReasoning int
	Seed         int64
	// Progress is called after each item.
	Progress func(done, total int, o Outcome)
}

// Run generates and checks every item.
func Run(ctx context.Context, svc core.Service, suiteName, suiteSHA string, items []Item, opts Options) (*Report, error) {
	modes := []bool{false}
	switch opts.Thinking {
	case "on":
		modes = []bool{true}
	case "both":
		modes = []bool{false, true}
	case "", "off":
	default:
		return nil, fmt.Errorf("thinking must be off, on or both")
	}
	if opts.MaxTokens == 0 {
		opts.MaxTokens = 1024
	}
	rep := &Report{
		Schema: ReportSchema, Suite: SuiteRef{Name: suiteName, SHA256: suiteSHA, Items: len(items)},
		Model: opts.Model, Date: time.Now().UTC().Format("2006-01-02"),
	}
	total := 0
	for _, it := range items {
		if it.Thinking != nil {
			total++
		} else {
			total += len(modes)
		}
	}
	zero := 0.0
	done := 0
	for _, it := range items {
		ms := modes
		if it.Thinking != nil {
			ms = []bool{*it.Thinking}
		}
		for _, thinking := range ms {
			if err := ctx.Err(); err != nil {
				return rep, err
			}
			th := thinking
			req := core.ChatRequest{
				Messages: it.Messages, Tools: it.Tools, EnableThinking: &th, MaxReasoningTokens: opts.MaxReasoning,
				Generation: core.Generation{
					Model: opts.Model, MaxTokens: max(it.MaxTokens, opts.MaxTokens), ResponseFormat: it.Response,
					Sampling: core.Sampling{Temperature: &zero, Seed: &opts.Seed},
				},
			}
			start := time.Now()
			res, err := svc.Chat(ctx, req, nil)
			o := Outcome{ID: it.ID, Lang: it.Lang, Capability: it.Capability, Thinking: thinking, MS: float64(time.Since(start).Microseconds()) / 1000}
			if err != nil {
				o.Error = err.Error()
				f := false
				o.Pass = &f
				o.Detail = "generation failed"
			} else {
				o.Answer, o.Reasoning, o.ToolCalls = res.Message.Content, res.Message.ReasoningContent, res.Message.ToolCalls
				o.Finish, o.Usage, o.TTFTMS = res.FinishReason, res.Usage, res.Stats.TTFTMS
				o.Pass, o.Detail = it.Checker.Check(o.Answer, o.ToolCalls)
			}
			rep.Results = append(rep.Results, o)
			done++
			if opts.Progress != nil {
				opts.Progress(done, total, o)
			}
		}
	}
	rep.Summarize()
	return rep, nil
}

// Summarize recomputes the tallies from the results.
func (r *Report) Summarize() {
	r.Summary, r.ByLang = map[string]Tally{}, map[string]Tally{}
	add := func(m map[string]Tally, k string, o Outcome) {
		t := m[k]
		if o.Pass == nil {
			t.Human++
		} else {
			t.N++
			if *o.Pass {
				t.Pass++
			}
			t.Rate = float64(t.Pass) / float64(t.N)
		}
		m[k] = t
	}
	var ttft []float64
	for _, o := range r.Results {
		if o.TTFTMS > 0 {
			ttft = append(ttft, o.TTFTMS)
		}
		key := o.Capability
		if o.Thinking {
			key += "+thinking"
		}
		add(r.Summary, key, o)
		add(r.ByLang, o.Lang, o)
	}
	if len(ttft) > 0 {
		sort.Float64s(ttft)
		if r.Extra == nil {
			r.Extra = map[string]float64{}
		}
		r.Extra["ttft_ms_p50"] = ttft[len(ttft)/2]
		r.Extra["ttft_ms_p95"] = ttft[min(len(ttft)-1, len(ttft)*95/100)]
	}
}

// Check judges an answer. It returns nil for a checker a person has to
// apply.
func (c Checker) Check(answer string, calls []core.ToolCall) (*bool, string) {
	yes, no := true, false
	pass := func(ok bool, detail string) (*bool, string) {
		if ok {
			return &yes, detail
		}
		return &no, detail
	}
	a := strings.TrimSpace(answer)
	switch c.Type {
	case CheckHuman:
		return nil, "rated by a person"
	case CheckContains:
		low := strings.ToLower(a)
		for _, n := range c.None {
			if strings.Contains(low, strings.ToLower(n)) {
				return pass(false, "contains forbidden "+quote(n))
			}
		}
		if len(c.Any) == 0 {
			return pass(true, "")
		}
		for _, s := range c.Any {
			if strings.Contains(low, strings.ToLower(s)) {
				return pass(true, "contains "+quote(s))
			}
		}
		return pass(false, "contains none of "+quote(strings.Join(c.Any, " | ")))
	case CheckRegex:
		re := regexp.MustCompile(c.Pattern)
		return pass(re.MatchString(a), "")
	case CheckExact:
		return pass(a == c.Want, "")
	case CheckNeedle:
		return pass(strings.Contains(strings.ToLower(a), strings.ToLower(c.Want)), "")
	case CheckJSON:
		var v any
		if err := json.Unmarshal([]byte(a), &v); err != nil {
			return pass(false, "not JSON: "+err.Error())
		}
		if len(c.Schema) > 0 {
			var s map[string]any
			if err := json.Unmarshal(c.Schema, &s); err != nil {
				return pass(false, "bad schema in suite")
			}
			if err := fits(v, s, "$"); err != nil {
				return pass(false, err.Error())
			}
		}
		return pass(true, "")
	case CheckToolCall:
		for _, call := range calls {
			if call.Name != c.Name {
				continue
			}
			var args map[string]any
			if err := json.Unmarshal([]byte(call.Arguments), &args); err != nil {
				return pass(false, "arguments are not a JSON object")
			}
			for k, want := range c.Args {
				if !sameValue(args[k], want) {
					return pass(false, fmt.Sprintf("argument %s = %v, want %v", k, args[k], want))
				}
			}
			return pass(true, "")
		}
		names := make([]string, len(calls))
		for i, c := range calls {
			names[i] = c.Name
		}
		return pass(false, fmt.Sprintf("no call to %s (calls: %v)", c.Name, names))
	}
	return pass(false, "unknown checker")
}

func quote(s string) string { b, _ := json.Marshal(s); return string(b) }

// sameValue compares JSON values, strings case-insensitively and trimmed: a
// tool call to get_weather(city="paris") answers the question as well.
func sameValue(got, want any) bool {
	if gs, ok := got.(string); ok {
		if ws, ok := want.(string); ok {
			return strings.EqualFold(strings.TrimSpace(gs), strings.TrimSpace(ws))
		}
	}
	gb, _ := json.Marshal(got)
	wb, _ := json.Marshal(want)
	return bytes.Equal(gb, wb)
}

// fits checks a value against the JSON Schema subset suites use: type,
// properties, required, enum, items, minItems.
func fits(v any, s map[string]any, path string) error {
	if t, ok := s["type"].(string); ok {
		ok := false
		switch t {
		case "object":
			_, ok = v.(map[string]any)
		case "array":
			_, ok = v.([]any)
		case "string":
			_, ok = v.(string)
		case "number":
			_, ok = v.(float64)
		case "integer":
			f, isNum := v.(float64)
			ok = isNum && f == float64(int64(f))
		case "boolean":
			_, ok = v.(bool)
		case "null":
			ok = v == nil
		}
		if !ok {
			return fmt.Errorf("%s is not %s", path, t)
		}
	}
	if enum, ok := s["enum"].([]any); ok {
		found := false
		for _, e := range enum {
			if sameValue(v, e) {
				found = true
			}
		}
		if !found {
			return fmt.Errorf("%s is not one of %v", path, enum)
		}
	}
	if obj, ok := v.(map[string]any); ok {
		if req, ok := s["required"].([]any); ok {
			for _, r := range req {
				if _, has := obj[r.(string)]; !has {
					return fmt.Errorf("%s lacks %s", path, r)
				}
			}
		}
		if props, ok := s["properties"].(map[string]any); ok {
			keys := make([]string, 0, len(props))
			for k := range props {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				if pv, has := obj[k]; has {
					if ps, ok := props[k].(map[string]any); ok {
						if err := fits(pv, ps, path+"."+k); err != nil {
							return err
						}
					}
				}
			}
		}
	}
	if arr, ok := v.([]any); ok {
		if mi, ok := s["minItems"].(float64); ok && float64(len(arr)) < mi {
			return fmt.Errorf("%s has %d items, fewer than %v", path, len(arr), mi)
		}
		if is, ok := s["items"].(map[string]any); ok {
			for i, e := range arr {
				if err := fits(e, is, fmt.Sprintf("%s[%d]", path, i)); err != nil {
					return err
				}
			}
		}
	}
	return nil
}
