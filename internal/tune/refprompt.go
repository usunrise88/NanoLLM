package tune

import (
	_ "embed"
	"errors"
	"fmt"
	"strings"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/chat"
	"github.com/usunrise88/nanollm/internal/core"
)

// SuiteVersion changes with the reference request and the probe logic: a
// profile measured by another version is stale (docs/OPTIMIZE.md, §3.6).
const SuiteVersion = 1

// refText is the text of the reference request: a prose of our own, English
// and Russian, long enough for a prompt of RefPromptTokens. The question stands
// before the text, so any prefix of the prompt is a prompt the model can read.
//
//go:embed refprompt.txt
var refText string

// RefTextSHA256 is the hash of refprompt.txt with its line ends as LF. The text
// is frozen: a change moves every pp and tg the autotuner has recorded, so it
// comes with a new SuiteVersion and new baselines in matrix.yaml.
const RefTextSHA256 = "df40d0885957f2c374becb8413383871d7f4d83fc9e48c2b3f267cdbc69a2bed"

// refBody is the reference text with LF line ends, whatever the checkout did to
// the file.
func refBody() string { return strings.TrimSpace(strings.ReplaceAll(refText, "\r\n", "\n")) }

// refQuestion opens the reference prompt. The answer is a retelling, as in
// real use; a repeated phrase would let the drafter guess nearly every token
// and overstate what MTP gives.
const refQuestion = "Read the text below carefully, then retell its main ideas in your own words, chapter by chapter, and say what the two stories have in common."

const bodyMark = "@@BODY@@"

// RefPrompt builds the reference prompt for a model's tokenizer: the chat
// markup, the question and as much of the text as makes n tokens in all, the
// generation prompt last. It is the same for every model that tokenizes the
// same way, and the first k tokens of it are a prefix of the whole.
func RefPrompt(tok interface {
	Tokenize(text string, addSpecial, parseSpecial bool) ([]backend.Token, error)
}, n int) ([]backend.Token, error) {
	marked := chat.PromptGemma4([]core.Message{{Role: core.RoleUser, Content: refQuestion + "\n\n" + bodyMark}}, nil, chat.Options{AddGenerationPrompt: true})
	pre, post, ok := strings.Cut(marked, bodyMark)
	if !ok {
		return nil, errors.New("tune: the chat template swallowed the reference text")
	}
	head, err := tok.Tokenize(pre, false, true)
	if err != nil {
		return nil, err
	}
	tail, err := tok.Tokenize(post, false, true)
	if err != nil {
		return nil, err
	}
	body, err := tok.Tokenize(refBody(), false, false)
	if err != nil {
		return nil, err
	}
	room := n - len(head) - len(tail)
	if room <= 0 {
		return nil, fmt.Errorf("tune: a prompt of %d tokens leaves no room for the text", n)
	}
	if len(body) < room {
		return nil, fmt.Errorf("tune: the reference text has %d tokens, %d are needed", len(body), room)
	}
	out := make([]backend.Token, 0, n)
	out = append(out, head...)
	out = append(out, body[:room]...)
	out = append(out, tail...)
	return out, nil
}

// ShortPrompt is a short chat prompt (about 64 tokens with a real tokenizer),
// for the check that drafting leaves the answer unchanged: it asks for a
// retelling of the first lines of the reference text.
func ShortPrompt(tok interface {
	Tokenize(text string, addSpecial, parseSpecial bool) ([]backend.Token, error)
}) ([]backend.Token, error) {
	text := refBody()
	if i := strings.Index(text, "\n\n"); i > 0 {
		text = text[:i]
	}
	if len(text) > 200 {
		text = text[:200]
	}
	prompt := chat.PromptGemma4([]core.Message{{Role: core.RoleUser, Content: "Retell this: " + text}}, nil, chat.Options{AddGenerationPrompt: true})
	return tok.Tokenize(prompt, false, true)
}
