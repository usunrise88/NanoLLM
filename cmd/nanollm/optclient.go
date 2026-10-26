package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/tune"
)

// optClient talks to a running NanoLLM service over its HTTP API
// (docs/OPTIMIZE.md, §4.2). When a service is up the measurement happens inside
// it: the same account, the same devices and the same GPU that it works with.
type optClient struct {
	base string
	key  string
	http *http.Client
}

// serviceAddr is where to reach the server of a configuration from the same
// machine: a wildcard bind is reached through the loopback.
func serviceAddr(addr string) string {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return addr
	}
	switch host {
	case "", "0.0.0.0", "::", "[::]":
		host = "127.0.0.1"
	}
	return net.JoinHostPort(host, port)
}

// detectService finds out whether the configuration's server answers: a GET of
// /healthz within a second.
func detectService(cfg config.Config, apiKey string) (*optClient, bool) {
	c := &optClient{base: "http://" + serviceAddr(cfg.Server.Addr), http: &http.Client{}}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, c.base+"/healthz", nil)
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, false
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, false
	}
	c.key = apiKey
	return c, true
}

// adminKey picks the credential for the API: the flag or the environment, else
// the first admin key of the configuration that is stored in the clear. In open
// mode none is needed.
func adminKey(cfg config.Config, flagKey string) (string, error) {
	if flagKey != "" {
		return flagKey, nil
	}
	if k := os.Getenv("NANOLLM_API_KEY"); k != "" {
		return k, nil
	}
	if cfg.Auth.Mode == "open" {
		return "", nil
	}
	hashed := false
	for _, k := range cfg.Auth.Keys {
		if !k.Admin {
			continue
		}
		if strings.HasPrefix(k.Secret, "sha256:") {
			hashed = true
			continue
		}
		return k.Secret, nil
	}
	if hashed {
		return "", errors.New("the server is running and its admin keys are stored as hashes: pass one with --api-key (or NANOLLM_API_KEY), or stop the server to measure without it")
	}
	return "", errors.New("the server is running and the configuration has no admin key to talk to it with")
}

func (c *optClient) do(ctx context.Context, method, path string, body any) (*http.Response, error) {
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.base+path, rd)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.key != "" {
		req.Header.Set("Authorization", "Bearer "+c.key)
	}
	return c.http.Do(req)
}

// call does a request and decodes a JSON answer; a problem answer becomes an
// error carrying its detail.
func (c *optClient) call(ctx context.Context, method, path string, body, out any) error {
	resp, err := c.do(ctx, method, path, body)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode/100 != 2 {
		var p struct {
			Detail string `json:"detail"`
			Code   string `json:"code"`
		}
		if json.Unmarshal(b, &p) == nil && p.Detail != "" {
			return fmt.Errorf("%s (%s, HTTP %d)", p.Detail, p.Code, resp.StatusCode)
		}
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	if out == nil {
		return nil
	}
	return json.Unmarshal(b, out)
}

// stream reads the server-sent events of a run and calls fn for each; it
// returns when the run's done or error event arrives.
func (c *optClient) stream(ctx context.Context, id string, fn func(name string, data json.RawMessage)) error {
	resp, err := c.do(ctx, http.MethodGet, "/api/v1/optimize/"+id+"/events", nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("events: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	return parseSSE(resp.Body, func(name string, data []byte) bool {
		fn(name, data)
		return name != "done" && name != "error"
	})
}

// parseSSE reads server-sent events (event: and data: lines, a blank line
// ending each) and calls fn; fn returning false stops the read.
func parseSSE(r io.Reader, fn func(name string, data []byte) bool) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64<<10), 8<<20)
	var name string
	var data []byte
	flush := func() bool {
		if name == "" && len(data) == 0 {
			return true
		}
		cont := fn(name, data)
		name, data = "", nil
		return cont
	}
	for sc.Scan() {
		line := sc.Text()
		switch {
		case line == "":
			if !flush() {
				return nil
			}
		case strings.HasPrefix(line, ":"):
			// a comment: the keep-alive
		case strings.HasPrefix(line, "event:"):
			name = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			if len(data) > 0 {
				data = append(data, '\n')
			}
			data = append(data, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " ")...)
		}
	}
	flush()
	return sc.Err()
}

// optimizeViaService runs the autotuner in the running service and prints what
// the standalone form prints.
func optimizeViaService(ctx context.Context, c *optClient, mode tune.Mode, model string, dry, asJSON, yes, interactive bool) error {
	var started struct {
		ID        string `json:"id"`
		EstimateS int    `json:"estimate_s"`
	}
	if err := c.call(ctx, http.MethodPost, "/api/v1/optimize", map[string]any{"mode": mode, "model": model, "dry_run": dry}, &started); err != nil {
		return fmt.Errorf("optimize: %w", err)
	}
	if !asJSON {
		fmt.Fprintf(os.Stderr, "Measuring inside the running service (about %d s)...\n", started.EstimateS)
	}
	var failure string
	err := c.stream(ctx, started.ID, func(name string, data json.RawMessage) {
		switch name {
		case "probe_start", "probe", "decision":
			var e tune.Event
			if json.Unmarshal(data, &e) == nil && !asJSON {
				fmt.Fprintf(os.Stderr, "[%3.0f s] %s\n", e.ElapsedS, e.Message)
			}
		case "error":
			var p struct {
				Detail string `json:"detail"`
			}
			_ = json.Unmarshal(data, &p)
			failure = p.Detail
		}
	})
	if err != nil {
		return err
	}
	if failure != "" {
		return errors.New("optimize: " + failure)
	}
	var res struct {
		Profile *tune.Profile `json:"profile"`
		Report  *tune.Report  `json:"report"`
	}
	if err := c.call(ctx, http.MethodGet, "/api/v1/optimize/"+started.ID, nil, &res); err != nil {
		return err
	}
	if dry {
		if asJSON {
			return writeJSONTo(os.Stdout, res.Report)
		}
		if res.Report != nil {
			res.Report.WriteText(os.Stdout)
		}
		return nil
	}
	if res.Profile == nil {
		return errors.New("optimize: the service returned no profile")
	}
	if !asJSON {
		res.Profile.WriteText(os.Stdout)
	}
	apply := yes
	if !apply && interactive {
		apply = askKey(os.Stdin, os.Stderr, "\nApply? [Y/n] ", true)
	}
	if !apply {
		if asJSON {
			return writeJSONTo(os.Stdout, res.Profile)
		}
		fmt.Fprintln(os.Stderr, "\nNothing was applied. Run again with --yes to apply.")
		return nil
	}
	var ar struct {
		Profile         string   `json:"profile"`
		ConfigWritten   bool     `json:"config_written"`
		Reloaded        bool     `json:"reloaded"`
		RestartRequired []string `json:"restart_required"`
		Note            string   `json:"note"`
	}
	if err := c.call(ctx, http.MethodPost, "/api/v1/optimize/"+started.ID+"/apply", nil, &ar); err != nil {
		return fmt.Errorf("optimize: apply: %w", err)
	}
	if asJSON {
		return writeJSONTo(os.Stdout, map[string]any{"profile": res.Profile, "apply": ar})
	}
	fmt.Fprintf(os.Stderr, "\nProfile %s applied.", ar.Profile)
	if ar.Reloaded {
		fmt.Fprint(os.Stderr, " The model was reloaded by it.")
	}
	fmt.Fprintln(os.Stderr)
	if ar.Note != "" {
		fmt.Fprintln(os.Stderr, ar.Note)
	}
	for _, r := range ar.RestartRequired {
		fmt.Fprintf(os.Stderr, "Restart the server to apply %s.\n", r)
	}
	return nil
}
