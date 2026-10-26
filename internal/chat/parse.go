package chat

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"unicode"

	"github.com/usunrise88/nanollm/internal/core"
)

// ParseOptions say what the output may contain, which follows from the
// request the way llama.cpp's Gemma 4 parser follows from it.
type ParseOptions struct {
	// Tools enables tool calls: tools were given and tool_choice is not none.
	Tools bool
	// JSON is a response_format request: the answer is a JSON value the
	// grammar wrapped in a ```json fence, and the fence is not content.
	JSON bool
}

// Parser splits a Gemma 4 answer into reasoning, content and tool calls as it
// streams. It is the counterpart of llama.cpp's PEG parser for the format
// (common/parsers/gemma4.cpp): an optional "<|turn>model\n", thought channels
// "<|channel>thought … <channel|>" anywhere before the content, empty
// channels and stray "<channel|>" dropped, and with tools a run of
// "<|tool_call>call:NAME{…}<tool_call|>" after which nothing else is content.
//
// Feed never emits text that might still turn out to be part of a marker; it
// holds back the shortest suffix that could be. Tool calls are emitted whole,
// with arguments converted to JSON; IDs are the caller's to assign.
type Parser struct {
	opts  ParseOptions
	buf   string
	state pstate
	// fenceOpen is set once the ```json fence of a JSON answer is consumed.
	fenceOpen bool
	skipLead  bool   // JSON mode: whitespace after the opening fence
	tail      string // JSON mode: held whitespace and backticks at the end
	// repaired is set when the parser dropped words that the model wrote after its calls.
	repaired bool
}

type pstate int

const (
	psStart pstate = iota
	psContent
	psChannel     // after "<|channel>"
	psThoughtLead // after "<|channel>thought", skipping whitespace
	psReasoning
	psToolCall   // inside "<|tool_call>…"
	psAfterTools // calls made; only more calls follow
)

const genPrompt = "<|turn>model\n"

// NewParser starts parsing one answer.
func NewParser(opts ParseOptions) *Parser { return &Parser{opts: opts} }

// Feed adds decoded text and returns what can be emitted now.
func (p *Parser) Feed(s string) ([]core.Delta, error) {
	p.buf += s
	return p.run(false)
}

// Close flushes everything held back at the end of generation.
func (p *Parser) Close() ([]core.Delta, error) {
	out, err := p.run(true)
	if err != nil {
		return out, err
	}
	if p.opts.JSON {
		// The closing fence and the whitespace around it were held back.
		t := strings.TrimRightFunc(p.tail, unicode.IsSpace)
		t = strings.TrimSuffix(t, "```")
		t = strings.TrimRightFunc(t, unicode.IsSpace)
		if t != "" {
			out = append(out, core.Delta{Content: t})
		}
		p.tail = ""
	}
	return out, nil
}

func (p *Parser) markers() []string {
	if p.opts.Tools {
		return []string{channelOpen, channelClose, callOpen}
	}
	return []string{channelOpen, channelClose}
}

func (p *Parser) run(final bool) ([]core.Delta, error) {
	var out []core.Delta
	content := func(s string) {
		if s == "" {
			return
		}
		if p.opts.JSON {
			s = p.unfence(s)
			if s == "" {
				return
			}
		}
		if n := len(out); n > 0 && out[n-1].Content != "" {
			out[n-1].Content += s
			return
		}
		out = append(out, core.Delta{Content: s})
	}
	reasoning := func(s string) {
		if s == "" {
			return
		}
		if n := len(out); n > 0 && out[n-1].ReasoningContent != "" {
			out[n-1].ReasoningContent += s
			return
		}
		out = append(out, core.Delta{ReasoningContent: s})
	}

	for {
		switch p.state {
		case psStart:
			if strings.HasPrefix(p.buf, genPrompt) {
				p.buf = p.buf[len(genPrompt):]
			} else if !final && strings.HasPrefix(genPrompt, p.buf) {
				return out, nil
			}
			p.state = psContent

		case psContent:
			i, m := firstMarker(p.buf, p.markers())
			if i < 0 {
				keep := 0
				if !final {
					keep = heldPrefix(p.buf, p.markers())
				}
				content(p.buf[:len(p.buf)-keep])
				p.buf = p.buf[len(p.buf)-keep:]
				return out, nil
			}
			content(p.buf[:i])
			p.buf = p.buf[i+len(m):]
			switch m {
			case channelOpen:
				p.state = psChannel
			case channelClose:
				// Gemma 4 sometimes closes a channel it never opened; the
				// token is not content.
			case callOpen:
				p.state = psToolCall
			}

		case psChannel:
			const thought = "thought"
			switch {
			case strings.HasPrefix(p.buf, thought):
				p.buf = p.buf[len(thought):]
				p.state = psThoughtLead
			case !final && strings.HasPrefix(thought, p.buf):
				return out, nil
			default:
				// An empty channel: "<|channel>" not followed by a thought.
				p.state = psContent
			}

		case psThoughtLead:
			trimmed := strings.TrimLeftFunc(p.buf, unicode.IsSpace)
			p.buf = trimmed
			if trimmed == "" && !final {
				return out, nil
			}
			p.state = psReasoning

		case psReasoning:
			i := strings.Index(p.buf, channelClose)
			if i < 0 {
				keep := 0
				if !final {
					keep = heldPrefix(p.buf, []string{channelClose})
				}
				reasoning(p.buf[:len(p.buf)-keep])
				p.buf = p.buf[len(p.buf)-keep:]
				return out, nil
			}
			reasoning(p.buf[:i])
			p.buf = p.buf[i+len(channelClose):]
			p.state = psContent

		case psToolCall:
			i := strings.Index(p.buf, callClose)
			if i < 0 {
				if final {
					return out, fmt.Errorf("gemma4: unterminated tool call %q", truncate(p.buf, 80))
				}
				return out, nil
			}
			call, err := parseCall(p.buf[:i])
			if err != nil {
				return out, err
			}
			out = append(out, core.Delta{ToolCalls: []core.ToolCall{call}})
			p.buf = p.buf[i+len(callClose):]
			p.state = psAfterTools

		case psAfterTools:
			i := strings.Index(p.buf, callOpen)
			if i < 0 {
				// Text after the calls is not part of the answer; keep only
				// what could start another call. Dropping words is a repair.
				keep := heldPrefix(p.buf, []string{callOpen})
				if final {
					keep = 0
				}
				if strings.TrimSpace(p.buf[:len(p.buf)-keep]) != "" {
					p.repaired = true
				}
				p.buf = p.buf[len(p.buf)-keep:]
				return out, nil
			}
			p.buf = p.buf[i+len(callOpen):]
			p.state = psToolCall
		}
	}
}

// unfence strips the ```json fence a JSON answer is wrapped in: the opening
// one at the start, and a closing one, which can only be recognised at the
// end, so trailing whitespace and backticks are held until Close.
func (p *Parser) unfence(s string) string {
	if !p.fenceOpen {
		s = p.tail + s
		p.tail = ""
		t := strings.TrimLeftFunc(s, unicode.IsSpace)
		const fence = "```json"
		switch {
		case strings.HasPrefix(t, fence):
			s = t[len(fence):]
			p.fenceOpen, p.skipLead = true, true
		case strings.HasPrefix(fence, t):
			p.tail = s
			return ""
		default:
			p.fenceOpen = true
		}
	}
	if p.skipLead {
		s = strings.TrimLeftFunc(s, unicode.IsSpace)
		if s == "" {
			return ""
		}
		p.skipLead = false
	}
	s = p.tail + s
	cut := len(strings.TrimRight(s, " \t\r\n`"))
	p.tail = s[cut:]
	return s[:cut]
}

func firstMarker(s string, markers []string) (int, string) {
	best, which := -1, ""
	for _, m := range markers {
		if i := strings.Index(s, m); i >= 0 && (best < 0 || i < best) {
			best, which = i, m
		}
	}
	return best, which
}

// heldPrefix is the length of the longest suffix of s that is a proper prefix
// of one of the markers.
func heldPrefix(s string, markers []string) int {
	keep := 0
	for _, m := range markers {
		for n := min(len(m)-1, len(s)); n > keep; n-- {
			if strings.HasSuffix(s, m[:n]) {
				keep = n
				break
			}
		}
	}
	return keep
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// parseCall reads "call:NAME{…}" into a tool call with JSON arguments.
func parseCall(s string) (core.ToolCall, error) {
	rest, ok := strings.CutPrefix(s, "call:")
	if !ok {
		return core.ToolCall{}, fmt.Errorf("gemma4: tool call without call: prefix: %q", truncate(s, 80))
	}
	brace := strings.IndexByte(rest, '{')
	if brace <= 0 {
		return core.ToolCall{}, fmt.Errorf("gemma4: tool call without arguments: %q", truncate(s, 80))
	}
	name := rest[:brace]
	r := &dslReader{s: rest[brace:]}
	args, err := r.dict()
	if err != nil {
		return core.ToolCall{}, fmt.Errorf("gemma4: tool call %s: %w", name, err)
	}
	r.space()
	if r.i != len(r.s) {
		return core.ToolCall{}, fmt.Errorf("gemma4: tool call %s: unexpected %q after arguments", name, truncate(r.s[r.i:], 40))
	}
	return core.ToolCall{Name: name, Arguments: args}, nil
}

// dslReader converts the model's argument syntax to JSON: strings are quoted
// with <|"|>, keys are bare up to the colon, numbers, booleans and null are
// JSON's own. It follows the grammar rules gemma4-* in llama.cpp.
type dslReader struct {
	s string
	i int
}

func (r *dslReader) space() {
	for r.i < len(r.s) && strings.IndexByte(" \t\r\n", r.s[r.i]) >= 0 {
		r.i++
	}
}

func (r *dslReader) lit(l string) bool {
	if strings.HasPrefix(r.s[r.i:], l) {
		r.i += len(l)
		return true
	}
	return false
}

func (r *dslReader) dict() (string, error) {
	if !r.lit("{") {
		return "", r.errf("expected {")
	}
	r.space()
	if r.lit("}") {
		return "{}", nil
	}
	var sb strings.Builder
	sb.WriteString("{")
	for first := true; ; first = false {
		if !first {
			sb.WriteString(",")
		}
		j := r.i
		for r.i < len(r.s) && r.s[r.i] != ':' && r.s[r.i] != '}' {
			r.i++
		}
		if r.i == j || !r.lit(":") {
			return "", r.errf("expected key:")
		}
		key, _ := json.Marshal(r.s[j : r.i-1])
		sb.Write(key)
		sb.WriteString(":")
		r.space()
		v, err := r.value()
		if err != nil {
			return "", err
		}
		sb.WriteString(v)
		r.space()
		if r.lit(",") {
			r.space()
			continue
		}
		if r.lit("}") {
			sb.WriteString("}")
			return sb.String(), nil
		}
		return "", r.errf("expected , or }")
	}
}

func (r *dslReader) array() (string, error) {
	if !r.lit("[") {
		return "", r.errf("expected [")
	}
	r.space()
	if r.lit("]") {
		return "[]", nil
	}
	var sb strings.Builder
	sb.WriteString("[")
	for first := true; ; first = false {
		if !first {
			sb.WriteString(",")
		}
		v, err := r.value()
		if err != nil {
			return "", err
		}
		sb.WriteString(v)
		r.space()
		if r.lit(",") {
			r.space()
			continue
		}
		if r.lit("]") {
			sb.WriteString("]")
			return sb.String(), nil
		}
		return "", r.errf("expected , or ]")
	}
}

func (r *dslReader) value() (string, error) {
	switch {
	case strings.HasPrefix(r.s[r.i:], q):
		r.i += len(q)
		end := strings.Index(r.s[r.i:], q)
		if end < 0 {
			return "", r.errf("unterminated string")
		}
		b, _ := json.Marshal(r.s[r.i : r.i+end])
		r.i += end + len(q)
		return string(b), nil
	case strings.HasPrefix(r.s[r.i:], "{"):
		return r.dict()
	case strings.HasPrefix(r.s[r.i:], "["):
		return r.array()
	case r.lit("true"):
		return "true", nil
	case r.lit("false"):
		return "false", nil
	case r.lit("null"):
		return "null", nil
	}
	j := r.i
	for r.i < len(r.s) && strings.IndexByte("+-0123456789.eE", r.s[r.i]) >= 0 {
		r.i++
	}
	num := r.s[j:r.i]
	if num == "" || !json.Valid([]byte(num)) {
		r.i = j
		return "", r.errf("expected a value")
	}
	return num, nil
}

func (r *dslReader) errf(msg string) error {
	return fmt.Errorf("%s at offset %d in %s", msg, r.i, strconv.Quote(truncate(r.s, 80)))
}

// InReasoning reports whether the parser is inside a thought, past the
// channel header, so the caller can count reasoning tokens and enforce a
// reasoning budget. Closing the channel any earlier would turn a half-read
// header into content.
func (p *Parser) InReasoning() bool {
	return p.state == psThoughtLead || p.state == psReasoning
}

// Repaired reports whether the parser dropped words that the model wrote after its
// calls. The answer does not hold them, and the parse report says "repaired".
func (p *Parser) Repaired() bool {
	return p.repaired
}
