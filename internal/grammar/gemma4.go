package grammar

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/usunrise88/nanollm/internal/core"
)

// Gemma 4 output grammars, after llama.cpp's PEG parser for the format
// (common/parsers/gemma4.cpp). The grammar sampler compares a token's text,
// and Gemma 4's control tokens render as their own text, so a literal such
// as "<|tool_call>" matches the control token.

// Rule names here carry a prefix no schema-derived rule can take: those are
// named after property paths, and a property called "ws" must not capture the
// fence's whitespace.
const (
	fenceRoot = "nanollm-answer"
	fenceWS   = "nanollm-ws"
)

var rootRule = regexp.MustCompile(`(?m)^root\s*::=\s*`)

// Gemma4JSON wraps a JSON grammar in the ```json fence llama.cpp makes Gemma
// 4 write for response_format, which keeps the model in the shape it was
// trained on; the answer parser strips the fence. The thought channel is not
// allowed before the fence: a constrained answer is generated without
// reasoning.
func Gemma4JSON(inner Spec) Spec {
	g := rootRule.ReplaceAllString(inner.GBNF, fenceRoot+" ::= ")
	g = "root ::= \"```json\" " + fenceWS + " " + fenceRoot + " " + fenceWS + " \"```\"\n" +
		fenceWS + " ::= [ \\t\\n]{0,20}\n" + g
	return Spec{GBNF: g, Root: "root", Warnings: inner.Warnings}
}

// gemma4DSL is the model's argument syntax: strings quoted with <|"|>, bare
// keys up to the colon, JSON numbers, booleans and null — the gemma4-* rules
// of llama.cpp. Arguments are not checked against the tool's schema there
// either (a TODO upstream); the parser converts them to JSON.
const gemma4DSL = `g4-dict ::= "{" g4-ws ( "}" | g4-kv ( "," g4-ws g4-kv )* g4-ws "}" )
g4-kv ::= g4-key ":" g4-ws g4-value
g4-key ::= [^:}<]+
g4-value ::= g4-string | g4-dict | g4-array | g4-number | "true" | "false" | "null"
g4-string ::= "<|\"|>" g4-str-char* "<|\"|>"
g4-str-char ::= [^<] | "<" [^|] | "<|" [^"] | "<|\"" [^|] | "<|\"|" [^>]
g4-array ::= "[" g4-ws ( "]" | g4-value ( "," g4-ws g4-value )* g4-ws "]" )
g4-number ::= "-"? ([0] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
g4-ws ::= [ \t\n]{0,20}
`

var toolName = regexp.MustCompile(`^[A-Za-z0-9_.:-]{1,128}$`)

// Gemma4Tools is the grammar for Gemma 4 tool calls: one call (several with
// parallel) of the declared tools in the model's own syntax. With required
// the answer must be calls, after an optional thought; otherwise the grammar
// is lazy, triggered by "<|tool_call>", so text before a call is free.
func Gemma4Tools(tools []core.Tool, required, parallel bool) (Spec, error) {
	if len(tools) == 0 {
		return Spec{}, fmt.Errorf("no tools to call")
	}
	var names []string
	for _, t := range tools {
		if !toolName.MatchString(t.Name) {
			return Spec{}, fmt.Errorf("tool name %q: use letters, digits and _ . : -", t.Name)
		}
		names = append(names, formatLiteral(t.Name))
	}
	calls := "g4-call"
	if parallel {
		calls = "g4-call+"
	}
	var sb strings.Builder
	if required {
		sb.WriteString("root ::= g4-thought? " + calls + "\n")
		sb.WriteString(`g4-thought ::= "<|channel>thought" g4-thought-char* "<channel|>"` + "\n")
		sb.WriteString(`g4-thought-char ::= [^<] | "<" [^c] | "<c" [^h] | "<ch" [^a]` + "\n")
	} else {
		sb.WriteString("root ::= " + calls + "\n")
	}
	sb.WriteString(`g4-call ::= "<|tool_call>call:" g4-name g4-dict "<tool_call|>"` + "\n")
	sb.WriteString("g4-name ::= " + strings.Join(names, " | ") + "\n")
	sb.WriteString(gemma4DSL)
	spec := Spec{GBNF: sb.String(), Root: "root"}
	if !required {
		spec.Triggers = []string{"<|tool_call>"}
	}
	return spec, nil
}
