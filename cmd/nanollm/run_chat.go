package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/registry"
	"github.com/usunrise88/nanollm/internal/server"
)

// chatOptions is what `run` passes to the service path.
type chatOptions struct {
	cfgPath, model, prompt, preset, device, system string
	think                                          bool
	maxTokens                                      int
	sampling                                       core.Sampling
	verbose                                        bool
}

// runChat is `nanollm run` on a catalog model: the same core.Service the
// server answers with, in process, so the terminal shows what the API would
// return. With a prompt it answers once; without, it is a chat REPL.
func runChat(o chatOptions, in io.Reader, out, errw io.Writer) error {
	cfg, err := config.Load(o.cfgPath)
	if err != nil {
		return err
	}
	model := o.model
	if model == "" {
		model = cfg.Models.Default
	}
	if o.preset != "" {
		if cfg.Models.Overrides == nil {
			cfg.Models.Overrides = map[string]config.ModelOverride{}
		}
		ov := cfg.Models.Overrides[model]
		ov.Preset = o.preset
		cfg.Models.Overrides[model] = ov
	}
	if o.device != "" {
		cfg.Devices.Default = o.device
	}
	if err := cfg.Validate(); err != nil {
		return err
	}
	level := "warn"
	if o.verbose {
		level = "debug"
	}
	log := newLogger(config.Log{Level: level, Format: "text"}, errw)
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM)
	defer stop()
	catalog, err := registry.Builtin()
	if err != nil {
		return err
	}
	backends := openBackends(cfg, log)
	defer func() {
		for _, b := range backends {
			_ = b.Close()
		}
	}()
	rt := server.New(ctx, server.Options{Config: cfg, Catalog: catalog, Store: newStore(cfg), Backends: backends, Log: log})
	defer rt.Close()

	s := &chatSession{svc: rt.Engine, model: model, opts: o, out: out, errw: errw}
	if o.system != "" {
		s.history = append(s.history, core.Message{Role: core.RoleSystem, Content: o.system})
	}
	if o.prompt != "" {
		return s.turn(ctx, o.prompt)
	}
	return s.repl(ctx, in)
}

type chatSession struct {
	svc     core.Service
	model   string
	opts    chatOptions
	history []core.Message
	out     io.Writer
	errw    io.Writer
}

const replHelp = `/think on|off  thinking for the next turns
/reset         forget the conversation
/exit          leave (or Ctrl-D)
`

// repl reads one user turn per line. Ctrl-C stops the answer in progress and
// keeps the session; Ctrl-D or /exit leaves.
func (s *chatSession) repl(ctx context.Context, in io.Reader) error {
	fmt.Fprintf(s.errw, "%s — /help for commands, Ctrl-D to leave\n", s.model)
	sc := bufio.NewScanner(in)
	sc.Buffer(make([]byte, 0, 64<<10), 1<<20)
	for {
		fmt.Fprint(s.errw, "> ")
		if !sc.Scan() {
			fmt.Fprintln(s.errw)
			return sc.Err()
		}
		line := strings.TrimSpace(sc.Text())
		switch {
		case line == "":
			continue
		case line == "/exit" || line == "/quit":
			return nil
		case line == "/help":
			fmt.Fprint(s.errw, replHelp)
			continue
		case line == "/reset":
			s.history = s.history[:0]
			if s.opts.system != "" {
				s.history = append(s.history, core.Message{Role: core.RoleSystem, Content: s.opts.system})
			}
			fmt.Fprintln(s.errw, "conversation cleared")
			continue
		case strings.HasPrefix(line, "/think"):
			switch strings.TrimSpace(strings.TrimPrefix(line, "/think")) {
			case "on":
				s.opts.think = true
			case "off":
				s.opts.think = false
			default:
				fmt.Fprintln(s.errw, "usage: /think on|off")
				continue
			}
			fmt.Fprintf(s.errw, "thinking %s\n", map[bool]string{true: "on", false: "off"}[s.opts.think])
			continue
		case strings.HasPrefix(line, "/"):
			fmt.Fprintf(s.errw, "unknown command %s; /help lists them\n", line)
			continue
		}
		// Ctrl-C interrupts this answer only.
		turnCtx, cancel := signal.NotifyContext(ctx, os.Interrupt)
		err := s.turn(turnCtx, line)
		interrupted := turnCtx.Err() != nil && ctx.Err() == nil
		cancel()
		switch {
		case interrupted:
			fmt.Fprintln(s.errw, "\n(interrupted)")
		case ctx.Err() != nil:
			return ctx.Err()
		case err != nil:
			fmt.Fprintln(s.errw, "error:", err)
		}
	}
}

// turn sends the history plus one user message and streams the answer:
// reasoning to errw, dimmed when it is a terminal, the answer to out. The
// turn joins the history only when it completes.
func (s *chatSession) turn(ctx context.Context, text string) error {
	think := s.opts.think
	req := core.ChatRequest{
		Generation:     core.Generation{Model: s.model, MaxTokens: s.opts.maxTokens, Sampling: s.opts.sampling},
		Messages:       append(append([]core.Message(nil), s.history...), core.Message{Role: core.RoleUser, Content: text}),
		EnableThinking: &think,
	}
	dim, reset := "", ""
	if isTerminal(s.errw) {
		dim, reset = "\x1b[2m", "\x1b[0m"
	}
	reasoning := false
	res, err := s.svc.Chat(core.WithInteractive(ctx, true), req, func(d core.Delta) error {
		if d.ReasoningContent != "" {
			if !reasoning {
				fmt.Fprint(s.errw, dim)
				reasoning = true
			}
			fmt.Fprint(s.errw, d.ReasoningContent)
		}
		if d.Content != "" {
			if reasoning {
				fmt.Fprint(s.errw, reset+"\n")
				reasoning = false
			}
			fmt.Fprint(s.out, d.Content)
		}
		return nil
	})
	if reasoning {
		fmt.Fprint(s.errw, reset)
	}
	fmt.Fprintln(s.out)
	if err != nil {
		return err
	}
	for _, w := range res.Warnings {
		fmt.Fprintln(s.errw, "warning:", w.Message)
	}
	st := res.Stats
	fmt.Fprintf(s.errw, "%s[%d+%d tokens, ttft %.0f ms, %.1f tok/s", dim, res.Usage.PromptTokens, res.Usage.CompletionTokens, st.TTFTMS, st.DecodeTokS)
	if st.DraftAcceptance != nil {
		fmt.Fprintf(s.errw, ", MTP %.0f%%", 100**st.DraftAcceptance)
	}
	where := st.Device
	if st.Preset != "" {
		where += ", " + st.Preset
	}
	if st.Mode != "" {
		where += " " + st.Mode
	}
	fmt.Fprintf(s.errw, ", %s]%s\n", where, reset)
	s.history = append(req.Messages, res.Message)
	return nil
}

func isTerminal(w io.Writer) bool {
	f, ok := w.(*os.File)
	if !ok {
		return false
	}
	fi, err := f.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}
