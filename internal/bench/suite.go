// Package bench is `nanollm bench`: the preset benchmark of a model on this
// machine, and the public benchmark's client (docs/BENCHMARKS.md).
//
// The orchestrator plans every preset the suite asks for, runs each distinct
// plan once in a child process of its own (`nanollm bench worker`), and
// assembles a nanollm.bench/v1 result. A child per preset gives each one its
// own peak memory, its own budget enforcement, and a crash that ends one
// preset rather than the run: GGML_ABORT and a C++ exception end the whole
// process. Submission is opt-in, after the person has seen the exact JSON.
package bench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

// Suites.
const (
	SuiteStandard = "standard"
	SuiteDev      = "dev"
)

// SuiteVersion is the version of the standard suite: its prompts, its
// generation parameters and its scenarios. Results of different versions are
// never compared, so any change here is a new version.
const SuiteVersion = "v1"

// Generation parameters every measurement uses: greedy, a fixed seed, and
// end-of-generation tokens banned so every run decodes the same length.
const (
	PPTokens = 512
	TGTokens = 128
	// SlowTGTokens is the decode length of a Slow run (WorkerSpec.Slow).
	SlowTGTokens = 12
	DepthN       = 4096
	ChatTokens   = 128
	Seed         = 42
	Reps         = 3
	// DevReps is the default for the dev suite: enough repetitions for a
	// spread that `bench compare` can judge a change against.
	DevReps = 5
)

// ppText is the prefill text, repeated to PPTokens; chatPrompt is the
// chat-ru scenario. Both are fixed by the suite version.
const (
	ppText = "Утром над рекой поднимался туман, и рыбаки не спешили выходить на лодках. " +
		"The old library smelled of paper and dust, and the clock on the wall was five minutes slow. "
	chatPrompt = "Объясни простыми словами, как работает солнечная батарея, и приведи два примера, где её используют."
	chatSystem = "Ты — вежливый ассистент. Отвечай по-русски, понятно и по делу."
)

// PromptsSHA256 fingerprints what the suite version fixes, so a client that
// drifted cannot pass as running it.
func PromptsSHA256() string {
	b, _ := json.Marshal([]any{SuiteVersion, ppText, chatSystem, chatPrompt, PPTokens, TGTokens, DepthN, ChatTokens, Seed, Reps})
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// planHash identifies a placement: two presets whose plans load the same way
// are measured once.
func planHash(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf("unhashable:%v", err)
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:8])
}
