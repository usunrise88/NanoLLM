package fake

import (
	"errors"
	"regexp"
	"strconv"
	"strings"

	"github.com/usunrise88/nanollm/internal/backend"
)

// A prompt picks what the fake model answers with a directive anywhere in it,
// usually in the last user message; the last directive wins:
//
//	[[fake:text]]          a fixed two-sentence reply (the default)
//	[[fake:echo]]          the last user message, verbatim
//	[[fake:think]]         a thinking channel, then a reply
//	[[fake:tool NAME ARGS]] one Gemma 4 tool call, ARGS in the model's own syntax
//	[[fake:json]]          a small JSON object
//	[[fake:raw TEXT]]      TEXT verbatim, control tokens included
//	[[fake:long N]]        N bytes of text
//	[[fake:loop]]          never stops on its own (max_tokens and cancellation)
//	[[fake:error]]         the step fails
//
// Every reply but loop ends with <turn|>, the end-of-turn token, as the real
// model's does.
var directive = regexp.MustCompile(`\[\[fake:([a-z]+)(?: ([^\]]*))?\]\]`)

// DefaultReply is what a prompt without a directive gets.
const DefaultReply = "Hello from the fake backend. Привет из фейкового бэкенда."

// errScenario is the failure [[fake:error]] asks for.
var errScenario = errors.New("fake: scripted step failure")

// script is a scenario resolved against a prompt: the output bytes, and
// whether the model stops after them.
type script struct {
	out     []byte
	endless bool
	fail    bool
}

func scenario(prompt string) script {
	m := directive.FindAllStringSubmatch(prompt, -1)
	if len(m) == 0 {
		return reply(DefaultReply)
	}
	name, arg := m[len(m)-1][1], m[len(m)-1][2]
	switch name {
	case "echo":
		return reply(lastUser(prompt))
	case "think":
		return reply("<|channel>thought\nThe user wants an answer; give a short one.<channel|>Here is a short answer.")
	case "tool":
		fn, args, _ := strings.Cut(arg, " ")
		if args == "" {
			args = "{}"
		}
		return reply("<|tool_call>call:" + fn + args + "<tool_call|>")
	case "json":
		return reply(`{"ok":true,"items":[1,2,3],"name":"fake"}`)
	case "raw":
		return reply(arg)
	case "long":
		n, _ := strconv.Atoi(arg)
		return reply(strings.Repeat("lorem ipsum ", n/12+1)[:max(n, 0)])
	case "loop":
		return script{out: []byte("la "), endless: true}
	case "error":
		return script{fail: true}
	default:
		return reply(DefaultReply)
	}
}

func reply(s string) script { return script{out: []byte(s + "<turn|>")} }

// lastUser is the text of the last user turn in a Gemma 4 prompt, without the
// directive asking for it.
func lastUser(prompt string) string {
	const open = "<|turn>user\n"
	i := strings.LastIndex(prompt, open)
	if i < 0 {
		return strings.TrimSpace(directive.ReplaceAllString(prompt, ""))
	}
	s := prompt[i+len(open):]
	if j := strings.Index(s, "<turn|>"); j >= 0 {
		s = s[:j]
	}
	return strings.TrimSpace(directive.ReplaceAllString(s, ""))
}

// tokens is the script as the model emits it: control tokens in the output
// are tokenized as such, so a tool call arrives as the real model's would.
func (s script) tokens() []backend.Token { return tokenize(string(s.out), false, true) }

// resume is the scenario for a prompt that may already hold the start of the
// answer after the last "<|turn>model\n": a continuation, as when the
// reasoning budget closes the thought channel and generation goes on. The
// script picks up after the part already written, or after its own channel
// close when the prompt closed the channel early; otherwise it starts over.
func resume(prompt string) script {
	const open = "<|turn>model\n"
	i := strings.LastIndex(prompt, open)
	if i < 0 {
		return scenario(prompt)
	}
	sc := scenario(prompt[:i+len(open)])
	partial := prompt[i+len(open):]
	if partial == "" || sc.endless || sc.fail {
		return sc
	}
	text := string(sc.out)
	switch {
	case strings.HasPrefix(text, partial):
		sc.out = []byte(text[len(partial):])
	case strings.HasSuffix(partial, "<channel|>") && strings.Contains(text, "<channel|>"):
		j := strings.Index(text, "<channel|>")
		sc.out = []byte(text[j+len("<channel|>"):])
	}
	return sc
}
