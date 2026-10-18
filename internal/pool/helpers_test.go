package pool

import (
	"github.com/usunrise88/nanollm/internal/sched"
)

func sched0(inst *Instance) sched.Request {
	toks, _ := inst.Model.Tokenize("[[fake:long 20]]", true, true)
	return sched.Request{Prompt: toks, Logprobs: -1}
}
