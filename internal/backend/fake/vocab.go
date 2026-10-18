package fake

import (
	"fmt"
	"strings"

	"github.com/usunrise88/nanollm/internal/backend"
)

// Specials are the Gemma 4 control tokens, in id order. Every other token is
// one byte, at id len(Specials)+b, so any text round-trips and a chat template
// rendered for the real model tokenizes here into the same control structure.
var Specials = []string{
	"<pad>", "<eos>", "<bos>", "<unk>",
	"<|turn>", "<turn|>",
	"<|think|>",
	"<|channel>", "<channel|>",
	"<|tool>", "<tool|>",
	"<|tool_call>", "<tool_call|>",
	"<|tool_response>", "<tool_response|>",
	`<|"|>`,
	"<|image|>", "<|audio|>", "<|video|>",
}

// Token ids the backend itself needs.
var (
	TokEOS     = special("<eos>")
	TokBOS     = special("<bos>")
	TokEndTurn = special("<turn|>")
)

// NVocab is the vocabulary size: the specials and 256 bytes.
var NVocab = len(Specials) + 256

func special(s string) backend.Token {
	for i, sp := range Specials {
		if sp == s {
			return backend.Token(i)
		}
	}
	panic("fake: no special token " + s)
}

func byteToken(b byte) backend.Token { return backend.Token(len(Specials) + int(b)) }

// tokenize splits text into bytes, matching control tokens first when
// parseSpecial is set, as llama_tokenize does.
func tokenize(text string, addSpecial, parseSpecial bool) []backend.Token {
	out := make([]backend.Token, 0, len(text)+1)
	if addSpecial {
		out = append(out, TokBOS)
	}
	for i := 0; i < len(text); {
		if parseSpecial && text[i] == '<' {
			if id, n := matchSpecial(text[i:]); n > 0 {
				out = append(out, id)
				i += n
				continue
			}
		}
		out = append(out, byteToken(text[i]))
		i++
	}
	return out
}

func matchSpecial(s string) (backend.Token, int) {
	best, bestLen := backend.Token(-1), 0
	for i, sp := range Specials {
		if len(sp) > bestLen && strings.HasPrefix(s, sp) {
			best, bestLen = backend.Token(i), len(sp)
		}
	}
	return best, bestLen
}

func piece(t backend.Token) (string, error) {
	switch {
	case t < 0 || int(t) >= NVocab:
		return "", fmt.Errorf("fake: token %d out of vocabulary (%d)", t, NVocab)
	case int(t) < len(Specials):
		return Specials[t], nil
	default:
		return string([]byte{byte(int(t) - len(Specials))}), nil
	}
}

func detokenize(tokens []backend.Token) (string, error) {
	var sb strings.Builder
	for _, t := range tokens {
		p, err := piece(t)
		if err != nil {
			return "", err
		}
		sb.WriteString(p)
	}
	return sb.String(), nil
}
