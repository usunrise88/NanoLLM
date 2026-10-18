package chat

import (
	"strings"

	"github.com/usunrise88/nanollm/internal/core"
)

// Gemma 4's chat template is written out here in Go rather than interpreted:
// llama_chat_apply_template does not know it, and llama.cpp's Jinja engine
// lives in common, outside the library NanoLLM binds. The port follows the
// template shipped in the model's GGUF statement by statement, and is held to
// it byte for byte by testdata rendered with jinja2 (tools/chatfixtures).
//
// Where the template branches on Python types, this file branches on JSON
// kinds: tool parameters and tool-call arguments are parsed into ordered
// values first, so keys sort, numbers print and missing keys vanish exactly as
// they do in Jinja.

// Options are the template variables besides messages and tools.
type Options struct {
	EnableThinking bool
	// AddGenerationPrompt opens the model's turn at the end.
	AddGenerationPrompt bool
	// PreserveThinking keeps earlier turns' reasoning when they made tool
	// calls; off, as in the template.
	PreserveThinking bool
}

// Gemma 4 control strings.
const (
	BOS          = "<bos>"
	q            = `<|"|>` // the template's string quote
	turnOpen     = "<|turn>"
	turnClose    = "<turn|>"
	thinkToken   = "<|think|>"
	channelOpen  = "<|channel>"
	channelClose = "<channel|>"
	toolOpen     = "<|tool>"
	toolClose    = "<tool|>"
	callOpen     = "<|tool_call>"
	callClose    = "<tool_call|>"
	respOpen     = "<|tool_response>"
	respClose    = "<tool_response|>"
)

// PromptGemma4 is the prompt for a chat request: tool results folded into
// the model turns that asked for them, the template rendered, and the model's
// turn opened even where the template leaves it closed — the same three steps
// llama.cpp takes (common/parsers/gemma4.cpp), so the model sees what it sees
// under llama-server.
func PromptGemma4(messages []core.Message, tools []core.Tool, opts Options) string {
	out := renderGemma4(foldToolResponses(messages), tools, opts)
	if opts.AddGenerationPrompt && strings.HasSuffix(out, turnClose+"\n") {
		// Content after tool results closes the turn, and the template then
		// adds no generation prompt; without one the model does not open its
		// reasoning channel properly.
		out += turnOpen + "model\n"
	}
	return out
}

// RenderGemma4 renders messages through the template as they are, without
// folding tool results. It is the template's own behaviour, kept for the
// conformance test; requests go through PromptGemma4.
func RenderGemma4(messages []core.Message, tools []core.Tool, opts Options) string {
	turns := make([]turn, len(messages))
	for i, m := range messages {
		turns[i] = turn{Message: m}
	}
	return renderGemma4(turns, tools, opts)
}

// turn is a message as the template sees it: possibly a model turn carrying
// the results of its own tool calls (the template's tool_responses).
type turn struct {
	core.Message
	responses []toolResult
}

type toolResult struct {
	name     string
	response value
}

// foldToolResponses turns assistant(tool_calls) → tool+ → assistant(content)
// into one model turn holding calls, results and the final content, as
// llama.cpp's convert_tool_responses_gemma4 does. A result is matched to its
// call by id, then by position (llama.cpp uses position only, which misnames
// results that arrive out of order), then named by the call id. Its content
// is parsed as JSON when it is JSON, so an object result renders in the
// model's own syntax. The content of the assistant message that made the
// calls is dropped, as there.
func foldToolResponses(messages []core.Message) []turn {
	var out []turn
	for i := 0; i < len(messages); {
		m := messages[i]
		if m.Role != core.RoleAssistant || len(m.ToolCalls) == 0 {
			out = append(out, turn{Message: m})
			i++
			continue
		}
		t := turn{Message: core.Message{Role: core.RoleAssistant, ToolCalls: m.ToolCalls, ReasoningContent: m.ReasoningContent}}
		i++
		for ; i < len(messages) && messages[i].Role == core.RoleTool; i++ {
			name := ""
			for _, c := range m.ToolCalls {
				if c.ID != "" && c.ID == messages[i].ToolCallID {
					name = c.Name
				}
			}
			if idx := len(t.responses); name == "" && idx < len(m.ToolCalls) {
				name = m.ToolCalls[idx].Name
			}
			if name == "" {
				name = messages[i].ToolCallID
			}
			resp, err := parseValue([]byte(messages[i].Content))
			if err != nil {
				resp = str(messages[i].Content)
			}
			t.responses = append(t.responses, toolResult{name: name, response: resp})
		}
		if i < len(messages) && messages[i].Role == core.RoleAssistant && len(messages[i].ToolCalls) == 0 && messages[i].Content != "" {
			t.Content = messages[i].Content
			i++
		}
		out = append(out, t)
	}
	return out
}

func renderGemma4(messages []turn, tools []core.Tool, opts Options) string {
	var sb strings.Builder
	var prevType string // the template's ns.prev_message_type
	sb.WriteString(BOS)

	loop := messages
	first := len(messages) > 0 && isSystem(messages[0].Role)
	if opts.EnableThinking || len(tools) > 0 || first {
		sb.WriteString(turnOpen + "system\n")
		if opts.EnableThinking {
			sb.WriteString(thinkToken + "\n")
			prevType = "think"
		}
		if first {
			sb.WriteString(pyStrip(messages[0].Content))
			loop = messages[1:]
		}
		if len(tools) > 0 {
			for _, t := range tools {
				sb.WriteString(toolOpen + pyStrip(declaration(t)) + toolClose)
			}
			prevType = "tool"
		}
		sb.WriteString(turnClose + "\n")
	}

	lastUser := -1
	for i, m := range loop {
		if m.Role == core.RoleUser {
			lastUser = i
		}
	}

	prevNonTool := ""
	for i, m := range loop {
		if m.Role == core.RoleTool {
			continue
		}
		prevType = ""
		role := m.Role
		if role == core.RoleAssistant {
			role = "model"
		}
		if role != "model" || prevNonTool != core.RoleAssistant {
			sb.WriteString(turnOpen + role + "\n")
		}

		if m.ReasoningContent != "" && (i > lastUser || (opts.PreserveThinking && len(m.ToolCalls) > 0)) {
			sb.WriteString(channelOpen + "thought\n" + m.ReasoningContent + "\n" + channelClose)
		}

		for _, c := range m.ToolCalls {
			sb.WriteString(callOpen + "call:" + c.Name + "{")
			sb.WriteString(callArguments(c.Arguments))
			sb.WriteString("}" + callClose)
		}
		if len(m.ToolCalls) > 0 {
			prevType = "tool_call"
		}

		// Tool results follow the call in the same turn: carried on the turn
		// itself, or else the run of tool messages after it, named by the
		// call they answer.
		responded := false
		if len(m.responses) > 0 {
			for _, r := range m.responses {
				name := r.name
				if name == "" {
					name = "unknown"
				}
				sb.WriteString(toolResponse(name, r.response))
				responded = true
				prevType = "tool_response"
			}
		} else if len(m.ToolCalls) > 0 {
			for _, follow := range loop[i+1:] {
				if follow.Role != core.RoleTool {
					break
				}
				name := follow.Name
				if name == "" {
					name = "unknown"
				}
				for _, c := range m.ToolCalls {
					if c.ID == follow.ToolCallID {
						name = c.Name
					}
				}
				sb.WriteString(toolResponse(name, str(follow.Content)))
				responded = true
				prevType = "tool_response"
			}
		}

		var content string
		if role == "model" {
			content = stripThinking(m.Content)
		} else {
			content = pyStrip(m.Content)
		}
		sb.WriteString(content)
		hasContent := pyStrip(content) != ""

		nextRole, nextFound := "", false
		for _, n := range loop[i+1:] {
			if n.Role != core.RoleTool {
				nextRole, nextFound = n.Role, true
				break
			}
		}
		continues := role == "model" && nextRole == core.RoleAssistant && (len(m.ToolCalls) == 0 || responded)

		switch {
		case prevType == "tool_call" && !responded:
			sb.WriteString(respOpen)
		case continues:
		case !responded || hasContent || nextFound:
			sb.WriteString(turnClose + "\n")
		}
		prevNonTool = m.Role
	}

	if opts.AddGenerationPrompt {
		switch {
		case prevType != "tool_response" && prevType != "tool_call":
			sb.WriteString(turnOpen + "model\n")
		case prevType == "tool_response" && opts.EnableThinking:
			sb.WriteString(channelOpen + "thought\n")
		}
	}
	return sb.String()
}

func isSystem(role string) bool { return role == core.RoleSystem || role == "developer" }

// callArguments renders a call's arguments. A JSON object is rendered in the
// template's own syntax; anything else is passed through as the template does
// with a pre-serialized string, with one outer pair of braces dropped.
func callArguments(args string) string {
	if v, err := parseValue([]byte(args)); err == nil && v.isMapping() {
		var parts []string
		for _, k := range v.sortedKeys() {
			parts = append(parts, k+":"+formatArgument(v.obj[k], false))
		}
		return strings.Join(parts, ",")
	}
	s := pyStrip(args)
	if strings.HasPrefix(s, "{") && strings.HasSuffix(s, "}") && len(s) >= 2 {
		return s[1 : len(s)-1]
	}
	return args
}

func toolResponse(name string, resp value) string {
	var sb strings.Builder
	sb.WriteString(respOpen)
	if resp.isMapping() {
		sb.WriteString("response:" + name + "{")
		for i, k := range resp.sortedKeys() {
			if i > 0 {
				sb.WriteString(",")
			}
			sb.WriteString(k + ":" + formatArgument(resp.obj[k], false))
		}
		sb.WriteString("}")
	} else {
		sb.WriteString("response:" + name + "{value:" + formatArgument(resp, false) + "}")
	}
	sb.WriteString(respClose)
	return sb.String()
}

// stripThinking drops thinking channels from an earlier model turn.
func stripThinking(text string) string {
	var sb strings.Builder
	for _, part := range strings.Split(text, channelClose) {
		if before, _, found := strings.Cut(part, channelOpen); found {
			sb.WriteString(before)
		} else {
			sb.WriteString(part)
		}
	}
	return pyStrip(sb.String())
}

func formatArgument(v value, escapeKeys bool) string {
	switch v.kind {
	case kNull:
		return "null"
	case kString:
		return q + v.str + q
	case kBool:
		if v.b {
			return "true"
		}
		return "false"
	case kObject:
		var sb strings.Builder
		sb.WriteString("{")
		for i, k := range v.sortedKeys() {
			if i > 0 {
				sb.WriteString(",")
			}
			if escapeKeys {
				sb.WriteString(q + k + q)
			} else {
				sb.WriteString(k)
			}
			sb.WriteString(":" + formatArgument(v.obj[k], escapeKeys))
		}
		sb.WriteString("}")
		return sb.String()
	case kArray:
		parts := make([]string, len(v.arr))
		for i, e := range v.arr {
			parts[i] = formatArgument(e, escapeKeys)
		}
		return "[" + strings.Join(parts, ",") + "]"
	default:
		return v.pyStr()
	}
}

// iterate is Jinja's for loop over a value: list elements, object keys, or
// the characters of a string.
func iterate(v value) []string {
	var out []string
	switch v.kind {
	case kArray:
		for _, e := range v.arr {
			out = append(out, e.pyStr())
		}
	case kObject:
		out = append(out, v.keys...)
	case kString:
		for _, r := range v.str {
			out = append(out, string(r))
		}
	}
	return out
}

func quotedList(v value) string {
	items := iterate(v)
	for i, it := range items {
		items[i] = q + it + q
	}
	return "[" + strings.Join(items, ",") + "]"
}

var standardKeys = map[string]bool{"description": true, "type": true, "properties": true, "required": true, "nullable": true}

// formatParameters is the template's format_parameters macro: JSON Schema
// properties in the model's declaration syntax. With filterKeys, the standard
// schema keys of an object without "properties" are skipped and the rest are
// treated as properties, as the template does.
func formatParameters(props value, filterKeys bool) string {
	var sb strings.Builder
	found := false
	for _, key := range props.sortedKeys() {
		if filterKeys && standardKeys[key] {
			continue
		}
		v := props.obj[key]
		addComma := false
		comma := func() {
			if addComma {
				sb.WriteString(",")
			}
			addComma = true
		}
		if found {
			sb.WriteString(",")
		}
		found = true
		sb.WriteString(key + ":{")
		if d := v.get("description"); d.truthy() {
			sb.WriteString("description:" + q + d.pyStr() + q)
			addComma = true
		}
		typ := v.get("type").upper()
		switch typ {
		case "STRING":
			if e := v.get("enum"); e.truthy() {
				comma()
				sb.WriteString("enum:" + formatArgument(e, true))
			}
		case "ARRAY":
			if items := v.get("items"); items.isMapping() && items.truthy() {
				comma()
				sb.WriteString("items:{")
				foundItem := false
				for _, ik := range items.sortedKeys() {
					iv := items.obj[ik]
					if iv.kind == kNull {
						continue
					}
					if foundItem {
						sb.WriteString(",")
					}
					foundItem = true
					switch ik {
					case "properties":
						sb.WriteString("properties:{")
						if iv.isMapping() {
							sb.WriteString(formatParameters(iv, false))
						}
						sb.WriteString("}")
					case "required":
						sb.WriteString("required:" + quotedList(iv))
					case "type":
						if iv.isString() {
							sb.WriteString("type:" + formatArgument(str(iv.upper()), true))
						} else {
							up := value{kind: kArray}
							for _, e := range iv.arr {
								up.arr = append(up.arr, str(e.upper()))
							}
							sb.WriteString("type:" + formatArgument(up, true))
						}
					default:
						sb.WriteString(ik + ":" + formatArgument(iv, true))
					}
				}
				sb.WriteString("}")
			}
		}
		if v.get("nullable").truthy() {
			comma()
			sb.WriteString("nullable:true")
		}
		if typ == "OBJECT" {
			if p := v.get("properties"); p.defined() && p.isMapping() {
				comma()
				sb.WriteString("properties:{" + formatParameters(p, false) + "}")
			} else if v.isMapping() {
				comma()
				sb.WriteString("properties:{" + formatParameters(v, true) + "}")
			}
			if r := v.get("required"); r.truthy() {
				comma()
				sb.WriteString("required:" + quotedList(r))
			}
		}
		comma()
		sb.WriteString("type:" + q + typ + q + "}")
	}
	return sb.String()
}

// declaration is format_function_declaration for one tool.
func declaration(t core.Tool) string {
	var sb strings.Builder
	sb.WriteString("declaration:" + t.Name + "{description:" + q + t.Description + q)
	var params value
	if len(t.Parameters) > 0 {
		params, _ = parseValue(t.Parameters)
	}
	if params.truthy() {
		sb.WriteString(",parameters:{")
		if p := params.get("properties"); p.truthy() {
			sb.WriteString("properties:{" + formatParameters(p, false) + "},")
		}
		if r := params.get("required"); r.truthy() {
			sb.WriteString("required:" + quotedList(r) + ",")
		}
		if ty := params.get("type"); ty.truthy() {
			sb.WriteString("type:" + q + ty.upper() + q + "}")
		}
	}
	sb.WriteString("}")
	return sb.String()
}
