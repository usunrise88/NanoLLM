// Package engine implements core.Service: the fixed request pipeline of SPEC
// §2 — template, tokenize, cache, prefill, decode, post — over the model pool
// and its scheduling loops.
//
// It is the only place that knows how a chat request becomes tokens and how
// tokens become an answer: the chat template and the answer parser (package
// chat), the grammar (package grammar), stop sequences, the reasoning budget,
// usage and stats. The dialects above it translate shapes; the loop below it
// only schedules tokens.
package engine

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"math"
	"strings"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/chat"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/grammar"
	"github.com/usunrise88/nanollm/internal/pool"
	"github.com/usunrise88/nanollm/internal/registry"
	"github.com/usunrise88/nanollm/internal/sched"
	"github.com/usunrise88/nanollm/internal/stats"
)

// Context overflow policies (config.Generation.ContextOverflow).
const (
	OverflowError          = "error"
	OverflowTruncateMiddle = "truncate_middle"
)

// Config is the part of the configuration the pipeline reads.
type Config struct {
	DefaultModel    string
	EnableThinking  bool
	MaxTokensCap    int
	ContextOverflow string
	// PromptCache is cache.prompt; a request can opt out, not in.
	PromptCache bool
}

// Target is where a request runs: the pool key and the catalog entry behind
// it, with the placement for stats.
type Target struct {
	Key      pool.Key
	Manifest registry.Manifest
}

// Resolver maps a request's model and device to a target. It is where the
// catalog, the store, the planner and the device policy meet; the engine does
// not know about any of them.
type Resolver func(ctx context.Context, model, device string) (Target, error)

// Service is the core.Service implementation.
type Service struct {
	cfg     Config
	pool    *pool.Pool
	resolve Resolver
	// Metrics counts finished requests; it is never nil.
	Metrics *stats.Metrics

	// model and thinking start as cfg.DefaultModel and cfg.EnableThinking and
	// change while the server runs (SetDefaultModel, SetThinking).
	model    atomic.Pointer[string]
	thinking atomic.Bool
}

// New returns the service.
func New(cfg Config, p *pool.Pool, resolve Resolver) *Service {
	if cfg.ContextOverflow == "" {
		cfg.ContextOverflow = OverflowError
	}
	s := &Service{cfg: cfg, pool: p, resolve: resolve, Metrics: stats.New()}
	s.SetDefaultModel(cfg.DefaultModel)
	s.SetThinking(cfg.EnableThinking)
	return s
}

// DefaultModel is the model a request that names none runs.
func (s *Service) DefaultModel() string { return *s.model.Load() }

// SetDefaultModel changes the model a request that names none runs, from the
// next request on.
func (s *Service) SetDefaultModel(id string) { s.model.Store(&id) }

// SetThinking changes whether a request that does not say thinks, from the
// next request on.
func (s *Service) SetThinking(on bool) { s.thinking.Store(on) }

var _ core.Service = (*Service)(nil)

// timer records stage durations.
type timer struct {
	stages map[core.Stage]float64
	last   time.Time
	start  time.Time // when the request began, for the metrics
}

func newTimer() *timer {
	now := time.Now()
	return &timer{stages: map[core.Stage]float64{}, last: now, start: now}
}

func (t *timer) mark(s core.Stage) {
	now := time.Now()
	t.stages[s] += float64(now.Sub(t.last).Microseconds()) / 1000
	t.last = now
}

// record counts a finished generation in the server's metrics. A request the
// client ended (context cancelled or past its deadline) is not an error.
func (s *Service) record(kind string, start time.Time, out *genOut, err error) {
	r := stats.Request{Kind: kind, Outcome: stats.OutcomeOK, TotalMS: float64(time.Since(start).Microseconds()) / 1000}
	switch {
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		r.Outcome = stats.OutcomeCancelled
	case err != nil:
		r.Outcome = stats.OutcomeError
	}
	if out != nil {
		r.Prompt, r.Cached = out.usage.PromptTokens, out.usage.CachedTokens
		r.Completion, r.Reasoning = out.usage.CompletionTokens, out.usage.ReasoningTokens
		r.Drafted, r.Accepted = out.drafted, out.accepted
		r.TTFTMS = out.stats.TTFTMS
		r.QueueMS = float64(out.queued.Microseconds()) / 1000
		r.DecodeMS = out.stats.StagesMS[core.StageDecode]
	}
	s.Metrics.Observe(r)
}

// acquire resolves and loads the model of a request.
func (s *Service) acquire(ctx context.Context, model, device string) (*pool.Instance, Target, func(), error) {
	if model == "" {
		model = s.DefaultModel()
	}
	t, err := s.resolve(ctx, model, device)
	if err != nil {
		return nil, Target{}, nil, err
	}
	inst, release, err := s.pool.Acquire(ctx, t.Key)
	if err != nil {
		switch {
		case errors.Is(err, pool.ErrBusy):
			return nil, t, nil, core.Errorf(core.CodeModelUnavailable, "the server is busy with other models; retry later").WithCause(err)
		case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
			return nil, t, nil, err
		}
		var ce *core.Error
		if errors.As(err, &ce) {
			return nil, t, nil, err
		}
		return nil, t, nil, core.Errorf(core.CodeModelUnavailable, "model %s could not be loaded", model).WithCause(err)
	}
	return inst, t, release, nil
}

func (s *Service) Chat(ctx context.Context, req core.ChatRequest, sink core.Sink) (*core.ChatResult, error) {
	switch req.ToolFormat {
	case "", core.ToolFormatStrict, core.ToolFormatRescue:
	default:
		return nil, core.Errorf(core.CodeInvalidRequest, "tool_format must be strict or rescue, not %q", req.ToolFormat).WithParam("tool_format")
	}
	if len(req.Messages) == 0 {
		return nil, core.Errorf(core.CodeInvalidRequest, "messages must not be empty").WithParam("messages")
	}
	for i, m := range req.Messages {
		switch m.Role {
		case core.RoleSystem, core.RoleUser, core.RoleAssistant, core.RoleTool, "developer":
		default:
			return nil, core.Errorf(core.CodeInvalidRequest, "messages[%d]: unknown role %q", i, m.Role).WithParam("messages")
		}
	}
	tm := newTimer()
	inst, target, release, err := s.acquire(ctx, req.Model, req.Device)
	if err != nil {
		return nil, err
	}
	defer release()
	tm.last = time.Now() // loading is not a pipeline stage

	thinking := s.thinking.Load()
	if req.EnableThinking != nil {
		thinking = *req.EnableThinking
	}
	tools := req.Tools
	choice := core.ToolChoiceAuto
	if req.ToolChoice != nil && req.ToolChoice.Mode != "" {
		choice = req.ToolChoice.Mode
	}
	if req.ToolChoice != nil && req.ToolChoice.Function != "" {
		var only []core.Tool
		for _, t := range tools {
			if t.Name == req.ToolChoice.Function {
				only = append(only, t)
			}
		}
		if len(only) == 0 {
			return nil, core.Errorf(core.CodeInvalidRequest, "tool_choice names %q, which is not among the tools", req.ToolChoice.Function).WithParam("tool_choice")
		}
		tools, choice = only, core.ToolChoiceRequired
	}
	if choice == core.ToolChoiceNone {
		tools = nil
	}
	if req.ToolFormat == core.ToolFormatRescue && choice == core.ToolChoiceRequired && len(tools) > 0 {
		return nil, core.Errorf(core.CodeInvalidRequest, "tool_choice required needs the tool grammar, which tool_format rescue does not apply").WithParam("tool_choice")
	}
	prompt := chat.PromptGemma4(req.Messages, tools, chat.Options{EnableThinking: thinking, AddGenerationPrompt: true, AssistantPrefix: req.AssistantPrefix})
	tm.mark(core.StageTemplate)

	g := gen{
		svc: s, inst: inst, target: target, tm: tm, sink: sink,
		parse:       &chat.ParseOptions{Tools: len(tools) > 0},
		rescue:      req.ToolFormat == core.ToolFormatRescue,
		maxReason:   req.MaxReasoningTokens,
		promptText:  prompt,
		addSpecial:  false,
		interactive: core.Interactive(ctx),
	}
	if err := g.constrain(req.Generation, tools, choice); err != nil {
		return nil, err
	}
	out, err := g.run(ctx, req.Generation)
	s.record("chat", tm.start, out, err)
	if err != nil {
		return nil, err
	}
	res := &core.ChatResult{
		Message: core.Message{
			Role: core.RoleAssistant, Content: out.content, ReasoningContent: out.reasoning, ToolCalls: out.calls,
		},
		FinishReason: out.finish,
		Logprobs:     out.logprobs,
		Usage:        out.usage,
		Stats:        out.stats,
		Warnings:     out.warnings,
		Parse:        out.parse,
	}
	return res, nil
}

func (s *Service) Complete(ctx context.Context, req core.CompletionRequest, sink core.Sink) (*core.CompletionResult, error) {
	if req.Prompt == "" {
		return nil, core.Errorf(core.CodeInvalidRequest, "prompt must not be empty").WithParam("prompt")
	}
	tm := newTimer()
	inst, target, release, err := s.acquire(ctx, req.Model, req.Device)
	if err != nil {
		return nil, err
	}
	defer release()
	tm.last = time.Now()
	tm.mark(core.StageTemplate)
	g := gen{
		svc: s, inst: inst, target: target, tm: tm, sink: sink,
		promptText: req.Prompt, addSpecial: true, interactive: core.Interactive(ctx),
	}
	if err := g.constrain(req.Generation, nil, core.ToolChoiceNone); err != nil {
		return nil, err
	}
	out, err := g.run(ctx, req.Generation)
	s.record("complete", tm.start, out, err)
	if err != nil {
		return nil, err
	}
	return &core.CompletionResult{
		Text: out.content, FinishReason: out.finish, Logprobs: out.logprobs,
		Usage: out.usage, Stats: out.stats, Warnings: out.warnings,
	}, nil
}

func (s *Service) Tokenize(ctx context.Context, req core.TokenizeRequest) (*core.TokenizeResult, error) {
	if (req.Text == "") == (len(req.Messages) == 0) {
		return nil, core.Errorf(core.CodeInvalidRequest, "give exactly one of text and messages")
	}
	inst, _, release, err := s.acquire(ctx, req.Model, "")
	if err != nil {
		return nil, err
	}
	defer release()
	var toks []backend.Token
	if req.Text != "" {
		toks, err = inst.Model.Tokenize(req.Text, false, false)
	} else {
		toks, err = inst.Model.Tokenize(chat.PromptGemma4(req.Messages, req.Tools, chat.Options{AddGenerationPrompt: true}), false, true)
	}
	if err != nil {
		return nil, core.Errorf(core.CodeInternal, "tokenize").WithCause(err)
	}
	out := make([]int, len(toks))
	for i, t := range toks {
		out[i] = int(t)
	}
	return &core.TokenizeResult{Tokens: out, Count: len(out)}, nil
}

func (s *Service) Detokenize(ctx context.Context, req core.DetokenizeRequest) (*core.DetokenizeResult, error) {
	inst, _, release, err := s.acquire(ctx, req.Model, "")
	if err != nil {
		return nil, err
	}
	defer release()
	nv := inst.Model.Capabilities().NVocab
	toks := make([]backend.Token, len(req.Tokens))
	for i, t := range req.Tokens {
		if t < 0 || (nv > 0 && t >= nv) {
			return nil, core.Errorf(core.CodeInvalidRequest, "tokens[%d] = %d is outside the vocabulary", i, t).WithParam("tokens")
		}
		toks[i] = backend.Token(t)
	}
	text, err := inst.Model.Detokenize(toks)
	if err != nil {
		return nil, core.Errorf(core.CodeInternal, "detokenize").WithCause(err)
	}
	return &core.DetokenizeResult{Text: text}, nil
}

// gen is one generation through the pipeline.
type gen struct {
	svc         *Service
	inst        *pool.Instance
	target      Target
	tm          *timer
	sink        core.Sink
	parse       *chat.ParseOptions // nil for raw completion
	rescue      bool               // tool_format rescue: no tool grammar
	sampler     backend.SamplerSpec
	noDraft     bool
	maxReason   int
	promptText  string
	addSpecial  bool
	interactive bool
	warnings    []core.Warning
}

type genOut struct {
	content, reasoning string
	calls              []core.ToolCall
	finish             string
	logprobs           []core.Logprob
	usage              core.Usage
	stats              core.Stats
	warnings           []core.Warning
	parse              *core.ParseReport
	// queued is the time the request waited for a slot, drafted and accepted
	// the MTP tokens, for the server's metrics.
	queued            time.Duration
	drafted, accepted int
}

// constrain builds the sampler: request values over the model card's, and
// the grammar the request implies.
func (g *gen) constrain(req core.Generation, tools []core.Tool, choice string) error {
	g.sampler = samplerSpec(req.Sampling, g.target.Manifest.SamplingDefaults)
	var spec grammar.Spec
	var err error
	switch {
	case req.Grammar != "":
		spec = grammar.Spec{GBNF: req.Grammar, Root: "root"}
	case req.ResponseFormat != nil && req.ResponseFormat.Type == core.FormatJSONObject:
		spec = grammar.JSONObject()
		if g.parse != nil {
			spec = grammar.Gemma4JSON(spec)
		}
	case req.ResponseFormat != nil && req.ResponseFormat.Type == core.FormatJSONSchema:
		spec, err = grammar.FromSchema(req.ResponseFormat.Schema)
		if err == nil && g.parse != nil {
			spec = grammar.Gemma4JSON(spec)
		}
	case len(tools) > 0 && choice == core.ToolChoiceRequired:
		spec, err = grammar.Gemma4Tools(tools, true, true)
	case len(tools) > 0 && g.rescue:
		// rescue applies no grammar: the parser reads the calls from the text.
	case len(tools) > 0:
		// tool_choice auto: the lazy grammar is an improvement, not a
		// requirement; the parser reads calls without it.
		if s, e := grammar.Gemma4Tools(tools, false, true); e == nil {
			spec = s
		}
	case req.ResponseFormat != nil && req.ResponseFormat.Type != core.FormatText && req.ResponseFormat.Type != "":
		return core.Errorf(core.CodeInvalidRequest, "unknown response_format type %q", req.ResponseFormat.Type).WithParam("response_format")
	}
	if err != nil {
		param := "response_format"
		if len(tools) > 0 {
			param = "tools"
		}
		return core.Errorf(core.CodeInvalidRequest, "%v", err).WithParam(param)
	}
	for _, w := range spec.Warnings {
		g.warnings = append(g.warnings, core.Warning{Code: "grammar_relaxed", Message: w})
	}
	if req.ResponseFormat != nil && req.ResponseFormat.Type != core.FormatText && g.parse != nil {
		g.parse.JSON = true
	}
	g.sampler.Grammar, g.sampler.GrammarRoot, g.sampler.GrammarTriggers = spec.GBNF, spec.Root, spec.Triggers
	// A rejected draft would have advanced the grammar past what the target
	// accepted; constrained requests do not draft.
	g.noDraft = spec.GBNF != ""
	return nil
}

func samplerSpec(s core.Sampling, d *registry.SamplingDefaults) backend.SamplerSpec {
	if d == nil {
		d = &registry.SamplingDefaults{}
	}
	pickF := func(v, def *float64, fallback float64) float64 {
		if v != nil {
			return *v
		}
		if def != nil {
			return *def
		}
		return fallback
	}
	out := backend.SamplerSpec{
		Temperature:      pickF(s.Temperature, d.Temperature, 1),
		TopP:             pickF(s.TopP, d.TopP, 1),
		MinP:             pickF(s.MinP, d.MinP, 0),
		RepeatPenalty:    pickF(s.RepeatPenalty, d.RepeatPenalty, 1),
		PresencePenalty:  pickF(s.PresencePenalty, nil, 0),
		FrequencyPenalty: pickF(s.FrequencyPenalty, nil, 0),
		Seed:             -1,
	}
	switch {
	case s.TopK != nil:
		out.TopK = *s.TopK
	case d.TopK != nil:
		out.TopK = *d.TopK
	}
	if s.Seed != nil {
		out.Seed = *s.Seed
	}
	if len(s.LogitBias) > 0 {
		out.LogitBias = map[backend.Token]float64{}
		for t, b := range s.LogitBias {
			out.LogitBias[backend.Token(t)] = b
		}
	}
	return out
}

func (g *gen) run(ctx context.Context, req core.Generation) (*genOut, error) {
	m := g.inst.Model
	caps := m.Capabilities()
	out := &genOut{warnings: g.warnings}

	toks, err := m.Tokenize(g.promptText, g.addSpecial, true)
	if err != nil {
		return nil, core.Errorf(core.CodeInternal, "tokenize the prompt").WithCause(err)
	}
	g.tm.mark(core.StageTokenize)

	maxTokens := req.MaxTokens
	if limit := g.svc.cfg.MaxTokensCap; limit > 0 && (maxTokens <= 0 || maxTokens > limit) {
		maxTokens = limit
	}
	if len(toks) >= caps.NCtx {
		if g.svc.cfg.ContextOverflow != OverflowTruncateMiddle {
			return nil, core.Errorf(core.CodeContextOverflow,
				"the prompt is %d tokens and the context holds %d", len(toks), caps.NCtx).WithParam("messages")
		}
		toks = truncateMiddle(toks, caps.NCtx-min(max(maxTokens, 1), caps.NCtx/4))
		out.warnings = append(out.warnings, core.Warning{Code: "context_truncated",
			Message: "the prompt did not fit the context; its middle was dropped"})
	}
	if req.Logprobs && req.TopLogprobs > 20 {
		return nil, core.Errorf(core.CodeInvalidRequest, "top_logprobs is at most 20").WithParam("top_logprobs")
	}
	lp := -1
	if req.Logprobs {
		lp = req.TopLogprobs
	}
	noCache := !g.svc.cfg.PromptCache || (req.CachePrompt != nil && !*req.CachePrompt)

	// The reasoning budget is enforced by ending the request at the budget
	// and continuing it with the channel closed; the continuation reuses the
	// whole cache, so it costs one step.
	var generated []backend.Token
	reasoningToks := 0
	budgetHit := false

	dec := &utf8Stream{}
	var parser *chat.Parser
	var raw strings.Builder
	var parseErr error
	if g.parse != nil {
		parser = chat.NewParser(*g.parse)
	}
	stop := newStopper(req.Stop)
	stopped := false
	var final *sched.Event
	var firstTok time.Time
	var drafted, accepted int
	var queued time.Duration
	cached := 0
	slot := 0
	var prefillDur time.Duration

	emit := func(d core.Delta) error {
		if d.Content != "" {
			out.content += d.Content
		}
		out.reasoning += d.ReasoningContent
		for i := range d.ToolCalls {
			d.ToolCalls[i].ID = callID()
		}
		out.calls = append(out.calls, d.ToolCalls...)
		if g.sink != nil && (d.Content != "" || d.ReasoningContent != "" || len(d.ToolCalls) > 0 || len(d.Logprobs) > 0) {
			return g.sink(d)
		}
		return nil
	}
	// feed routes decoded text through the parser and the stop sequences.
	feed := func(text string, lps []core.Logprob) error {
		raw.WriteString(text)
		var deltas []core.Delta
		switch {
		case parser != nil && parseErr != nil:
			// rescue stopped the parser; the report keeps the whole text.
		case parser != nil:
			ds, err := parser.Feed(text)
			if err != nil {
				if !g.rescue {
					return err
				}
				parseErr = err
			}
			deltas = ds
		case text != "":
			deltas = []core.Delta{{Content: text}}
		}
		for i, d := range deltas {
			if d.Content != "" {
				var hit bool
				d.Content, hit = stop.feed(d.Content)
				if hit {
					stopped = true
				}
			}
			if i == len(deltas)-1 {
				d.Logprobs = lps
				lps = nil
			}
			if err := emit(d); err != nil {
				return err
			}
			if stopped {
				return nil
			}
		}
		if len(lps) > 0 {
			return emit(core.Delta{Logprobs: lps})
		}
		return nil
	}

	prompt := toks
	for {
		remaining := 0
		if maxTokens > 0 {
			remaining = maxTokens - len(generated)
			if remaining <= 0 {
				final = &sched.Event{Done: true, Finish: sched.FinishLength}
				break
			}
		}
		gctx, cancel := context.WithCancel(ctx)
		h, err := g.inst.Loop.Submit(gctx, sched.Request{
			Prompt: prompt, Sampler: g.sampler, MaxTokens: remaining, Logprobs: lp,
			SessionID: req.SessionID, NoCache: noCache, Interactive: g.interactive, NoDraft: g.noDraft,
		})
		if err != nil {
			cancel()
			var ce sched.ContextError
			if errors.As(err, &ce) {
				return nil, core.Errorf(core.CodeContextOverflow, "the prompt is %d tokens and the context holds %d", ce.Prompt, ce.NCtx).WithParam("messages")
			}
			if errors.Is(err, sched.ErrOverloaded) {
				return nil, core.Errorf(core.CodeQueueFull, "the server is at its limit: too many requests are waiting for this model; retry shortly").WithCause(err)
			}
			return nil, core.Errorf(core.CodeModelUnavailable, "the model is unloading; retry").WithCause(err)
		}
		restart := false
		for {
			ev, err := h.Next(ctx)
			if err != nil {
				cancel()
				return nil, err
			}
			if firstTok.IsZero() && len(ev.Tokens) > 0 {
				firstTok = time.Now()
			}
			for i, t := range ev.Tokens {
				if parser != nil && parser.InReasoning() {
					reasoningToks++
				}
				generated = append(generated, t)
				piece, err := m.TokenPiece(t)
				if err != nil {
					cancel()
					return nil, core.Errorf(core.CodeInternal, "detokenize").WithCause(err)
				}
				var lps []core.Logprob
				if lp >= 0 && i < len(ev.Logprobs) {
					lps = []core.Logprob{g.logprob(ev.Logprobs[i])}
					out.logprobs = append(out.logprobs, lps...)
				}
				if err := feed(dec.write(piece), lps); err != nil {
					cancel()
					return nil, err
				}
				if stopped {
					break
				}
				if g.maxReason > 0 && parser != nil && parser.InReasoning() && reasoningToks >= g.maxReason {
					budgetHit = true
					break
				}
			}
			if stopped || budgetHit {
				cancel()
				// Wait for the loop to let go of the slot, keeping its stats.
				for !ev.Done {
					ev, _ = h.Next(context.WithoutCancel(ctx))
				}
			}
			if ev.Done {
				if ev.Stats != nil {
					queued += ev.Stats.Queued
					drafted += ev.Stats.Drafted
					accepted += ev.Stats.Accepted
					slot = ev.Stats.Slot
					if prefillDur == 0 {
						cached = ev.Stats.CachedTokens
						prefillDur = ev.Stats.Prefill
					}
				}
				if budgetHit {
					budgetHit = false
					g.maxReason = 0
					closeTok, err := m.Tokenize("<channel|>", false, true)
					if err != nil {
						cancel()
						return nil, core.Errorf(core.CodeInternal, "tokenize").WithCause(err)
					}
					prompt = append(append(append([]backend.Token(nil), toks...), generated...), closeTok...)
					if err := feed(dec.write("<channel|>"), nil); err != nil {
						cancel()
						return nil, err
					}
					restart = true
				} else {
					final = &ev
				}
				break
			}
		}
		cancel()
		if !restart {
			break
		}
	}
	if final.Err != nil && !stopped {
		if errors.Is(final.Err, context.Canceled) || errors.Is(final.Err, context.DeadlineExceeded) {
			return nil, final.Err
		}
		return nil, core.Errorf(core.CodeInternal, "generation failed").WithCause(final.Err)
	}

	// Flush what was held back.
	if !stopped {
		if err := feed(dec.flush(), nil); err != nil {
			return nil, err
		}
	}
	if parser != nil && !stopped && parseErr == nil {
		ds, err := parser.Close()
		if err != nil {
			out.warnings = append(out.warnings, core.Warning{Code: "parse_error", Message: err.Error()})
			parseErr = err
		}
		for _, d := range ds {
			if d.Content != "" {
				d.Content, stopped = stop.feed(d.Content)
			}
			if err := emit(d); err != nil {
				return nil, err
			}
		}
	}
	if !stopped {
		if rest := stop.flush(); rest != "" {
			if err := emit(core.Delta{Content: rest}); err != nil {
				return nil, err
			}
		}
	}
	end := time.Now()
	if !firstTok.IsZero() {
		g.tm.stages[core.StagePrefill] = float64(prefillDur.Microseconds()) / 1000
		g.tm.stages[core.StageDecode] = float64(end.Sub(firstTok).Microseconds()) / 1000
	}
	g.tm.last = end

	if parser != nil {
		out.parse = parseReport(parser, parseErr, raw.String())
		if parseErr != nil && g.rescue {
			out.content, out.reasoning, out.calls = raw.String(), "", nil
		}
	}
	switch {
	case len(out.calls) > 0:
		out.finish = core.FinishToolCalls
	case stopped:
		out.finish = core.FinishStop
	case final.Finish == sched.FinishLength:
		out.finish = core.FinishLength
	default:
		out.finish = core.FinishStop
	}
	out.usage = core.Usage{
		PromptTokens: len(toks), CachedTokens: cached,
		CompletionTokens: len(generated), ReasoningTokens: reasoningToks,
	}
	g.tm.mark(core.StagePost)
	out.stats = core.Stats{
		StagesMS:     g.tm.stages,
		PromptTokens: len(toks), CachedTokens: cached,
		CompletionTokens: len(generated), ReasoningTokens: reasoningToks,
		Backend: g.target.Key.Backend, Device: g.target.Key.Device,
		Preset: g.target.Key.Preset, Mode: g.target.Key.Mode, Slot: slot,
	}
	if ms := g.tm.stages[core.StagePrefill]; ms > 0 {
		out.stats.TTFTMS = ms
		out.stats.PrefillTokS = float64(len(toks)-cached) / (ms / 1000)
	}
	if ms := g.tm.stages[core.StageDecode]; ms > 0 && len(generated) > 1 {
		out.stats.DecodeTokS = float64(len(generated)-1) / (ms / 1000)
	}
	out.queued, out.drafted, out.accepted = queued, drafted, accepted
	if drafted > 0 {
		a := float64(accepted) / float64(drafted)
		out.stats.DraftAcceptance = &a
	}
	return out, nil
}

func (g *gen) logprob(t backend.TokenLogprobs) core.Logprob {
	piece := func(tok backend.Token) string {
		p, _ := g.inst.Model.TokenPiece(tok)
		return p
	}
	out := core.Logprob{Token: piece(t.Token), Logprob: finite(t.Logprob)}
	for _, a := range t.Top {
		out.Top = append(out.Top, core.TopToken{Token: piece(a.Token), Logprob: finite(a.Logprob)})
	}
	return out
}

// finite keeps -Inf (a token the grammar masked) out of JSON, which cannot
// carry it.
func finite(x float64) float64 {
	if math.IsInf(x, -1) || math.IsNaN(x) {
		return -9999
	}
	return x
}

// truncateMiddle keeps the start and the end of a prompt within n tokens: the
// system prompt and the latest turns matter more than the middle.
func truncateMiddle(toks []backend.Token, n int) []backend.Token {
	if len(toks) <= n {
		return toks
	}
	head := n / 2
	tail := n - head
	out := append([]backend.Token(nil), toks[:head]...)
	return append(out, toks[len(toks)-tail:]...)
}

func callID() string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return "call_" + hex.EncodeToString(b[:])
}

// utf8Stream joins token pieces into text without splitting a character: a
// multi-byte character can span tokens, and half of one is not text.
type utf8Stream struct{ pending []byte }

func (u *utf8Stream) write(piece string) string {
	u.pending = append(u.pending, piece...)
	n := len(u.pending)
	// Hold back an incomplete sequence at the end, at most 3 bytes.
	for back := 1; back <= 3 && back <= n; back++ {
		b := u.pending[n-back]
		if b < 0x80 {
			break
		}
		if b >= 0xC0 { // a lead byte: is its sequence complete?
			if !utf8.FullRune(u.pending[n-back:]) {
				n -= back
			}
			break
		}
	}
	out := string(u.pending[:n])
	u.pending = append(u.pending[:0], u.pending[n:]...)
	return out
}

func (u *utf8Stream) flush() string {
	out := strings.ToValidUTF8(string(u.pending), "�")
	u.pending = nil
	return out
}

// stopper cuts the content at the first stop sequence, holding back text that
// could still turn into one.
type stopper struct {
	stops []string
	held  string
}

func newStopper(stops []string) *stopper {
	var s []string
	for _, x := range stops {
		if x != "" {
			s = append(s, x)
		}
	}
	return &stopper{stops: s}
}

// feed returns the text that is safe to emit, and whether a stop sequence was
// found (the returned text then ends where it began).
func (s *stopper) feed(text string) (string, bool) {
	if len(s.stops) == 0 {
		return text, false
	}
	buf := s.held + text
	best := -1
	for _, st := range s.stops {
		if i := strings.Index(buf, st); i >= 0 && (best < 0 || i < best) {
			best = i
		}
	}
	if best >= 0 {
		s.held = ""
		return buf[:best], true
	}
	keep := 0
	for _, st := range s.stops {
		for n := min(len(st)-1, len(buf)); n > keep; n-- {
			if strings.HasSuffix(buf, st[:n]) {
				keep = n
				break
			}
		}
	}
	s.held = buf[len(buf)-keep:]
	return buf[:len(buf)-keep], false
}

func (s *stopper) flush() string {
	out := s.held
	s.held = ""
	return out
}

// parseReport says how the reply was read: ok, repaired, or failed with the reply
// text kept whole (RFC R4).
func parseReport(p *chat.Parser, err error, raw string) *core.ParseReport {
	switch {
	case err != nil:
		return &core.ParseReport{Status: core.ParseFailed, Raw: raw, Error: err.Error()}
	case p.Repaired():
		return &core.ParseReport{Status: core.ParseRepaired, Raw: raw}
	}
	return &core.ParseReport{Status: core.ParseOK}
}
