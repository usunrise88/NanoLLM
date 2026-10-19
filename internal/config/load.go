package config

import (
	"bytes"
	"encoding"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"reflect"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/usunrise88/nanollm/internal/plan"
)

// EnvPrefix starts every environment override. The rest of the name is the
// YAML path in upper case joined by underscores: server.addr is
// NANOLLM_SERVER_ADDR.
const EnvPrefix = "NANOLLM_"

// Load builds the effective configuration: defaults, then the YAML file if the
// path is non-empty, then NANOLLM_* environment overrides, then Validate.
// Flags are applied by the caller after Load, being highest priority.
func Load(path string) (Config, error) {
	cfg := Default()

	if path != "" {
		b, err := os.ReadFile(path)
		if err != nil {
			return cfg, fmt.Errorf("read config %s: %w", path, err)
		}
		if err := decode(b, &cfg); err != nil {
			return cfg, fmt.Errorf("parse config %s: %w", path, err)
		}
	}

	if err := applyEnv(&cfg, os.LookupEnv); err != nil {
		return cfg, err
	}
	if err := cfg.Validate(); err != nil {
		return cfg, err
	}
	return cfg, nil
}

// decode merges a YAML document over cfg. KnownFields makes a typo a startup
// failure rather than a silently ignored setting (SPEC §8).
func decode(b []byte, cfg *Config) error {
	dec := yaml.NewDecoder(bytes.NewReader(b))
	dec.KnownFields(true)
	if err := dec.Decode(cfg); err != nil && !errors.Is(err, io.EOF) {
		return err
	}
	return nil
}

// applyEnv lets every scalar and list key be overridden from the environment,
// for Docker (SPEC §8). The names are derived from the YAML tags rather than
// listed by hand, so a key added to the schema is overridable the day it is
// added, and the spelling cannot drift from the file's.
//
// Two shapes are not reachable this way. models.overrides is a map keyed by
// model id, which has no stable variable name. auth.keys is a list of records,
// so NANOLLM_AUTH_KEYS takes a comma-separated list of secrets instead, each
// becoming an unnamed, non-administrative key: granting admin rights should take
// a deliberate edit to a file, not a variable copied between deployments.
func applyEnv(c *Config, lookup func(string) (string, bool)) error {
	if v, ok := lookup(EnvPrefix + "AUTH_KEYS"); ok && v != "" {
		c.Auth.Keys = nil
		for i, secret := range strings.Split(v, ",") {
			if s := strings.TrimSpace(secret); s != "" {
				c.Auth.Keys = append(c.Auth.Keys, APIKey{Name: fmt.Sprintf("env-%d", i+1), Secret: s})
			}
		}
	}
	return envStruct(reflect.ValueOf(c).Elem(), strings.TrimSuffix(EnvPrefix, "_"), lookup)
}

var textUnmarshaler = reflect.TypeFor[encoding.TextUnmarshaler]()

func envStruct(v reflect.Value, prefix string, lookup func(string) (string, bool)) error {
	t := v.Type()
	for i := range t.NumField() {
		f := t.Field(i)
		tag, _, _ := strings.Cut(f.Tag.Get("yaml"), ",")
		if tag == "" || tag == "-" {
			continue
		}
		name := prefix + "_" + strings.ToUpper(tag)
		fv := v.Field(i)

		// A type that parses itself (Auto) is a leaf even though it is a struct.
		if reflect.PointerTo(f.Type).Implements(textUnmarshaler) {
			if s, ok := lookup(name); ok {
				if err := fv.Addr().Interface().(encoding.TextUnmarshaler).UnmarshalText([]byte(s)); err != nil {
					return fmt.Errorf("%s: %w", name, err)
				}
			}
			continue
		}
		if f.Type.Kind() == reflect.Struct {
			if err := envStruct(fv, name, lookup); err != nil {
				return err
			}
			continue
		}

		s, ok := lookup(name)
		if !ok {
			continue
		}
		if err := setScalar(fv, s); err != nil {
			return fmt.Errorf("%s=%q: %w", name, s, err)
		}
	}
	return nil
}

func setScalar(v reflect.Value, s string) error {
	switch v.Kind() {
	case reflect.String:
		v.SetString(s)
	case reflect.Bool:
		b, err := strconv.ParseBool(s)
		if err != nil {
			return errors.New("not a boolean")
		}
		v.SetBool(b)
	case reflect.Int, reflect.Int64:
		n, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return errors.New("not an integer")
		}
		v.SetInt(n)
	case reflect.Float64:
		x, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return errors.New("not a number")
		}
		v.SetFloat(x)
	case reflect.Slice:
		if v.Type().Elem().Kind() != reflect.String {
			// auth.keys: handled before the walk.
			return nil
		}
		var items []string
		for _, item := range strings.Split(s, ",") {
			if item = strings.TrimSpace(item); item != "" {
				items = append(items, item)
			}
		}
		v.Set(reflect.ValueOf(items))
	case reflect.Map:
		// models.overrides: no stable variable name, see applyEnv.
	default:
		return fmt.Errorf("cannot be set from the environment (%s)", v.Kind())
	}
	return nil
}

// Validate rejects configurations that would fail confusingly at request time.
func (c *Config) Validate() error {
	if err := c.validateAuth(); err != nil {
		return err
	}

	if len(c.API.Dialects) == 0 && !c.UI.Enabled {
		return fmt.Errorf("no api dialects enabled and ui disabled: server would serve nothing")
	}
	for i, d := range c.API.Dialects {
		switch strings.TrimSpace(d) {
		case DialectOpenAI, DialectNative, DialectOllama:
		default:
			return fmt.Errorf("api.dialects[%d]: unknown dialect %q; use %s, %s or %s",
				i, d, DialectOpenAI, DialectNative, DialectOllama)
		}
	}

	if len(c.Backends.Enabled) == 0 {
		return fmt.Errorf("backends.enabled is empty: nothing could run a model")
	}
	for i, b := range c.Backends.Enabled {
		if !knownBackend(b) {
			return fmt.Errorf("backends.enabled[%d]: unknown backend %q; use %s, %s or %s",
				i, b, BackendLlamaCpp, BackendORTGenAI, BackendMLX)
		}
	}
	switch c.Bench.Submit {
	case "ask", "never":
	default:
		return fmt.Errorf("bench.submit must be ask or never, got %q", c.Bench.Submit)
	}
	if u, err := url.Parse(c.Bench.SubmitURL); c.Bench.Submit != "never" && (err != nil || u.Scheme != "https" || u.Host == "") {
		return fmt.Errorf("bench.submit_url %q must be an https URL", c.Bench.SubmitURL)
	}
	if c.Server.MaxBody <= 0 {
		return fmt.Errorf("server.max_body must be positive")
	}
	for i, m := range c.Registry.Mirrors {
		if u, err := url.Parse(m); err != nil || u.Scheme != "https" || u.Host == "" {
			return fmt.Errorf("registry.mirrors[%d]: %q must be an https URL", i, m)
		}
	}
	if c.Backends.LibDir == "" {
		return fmt.Errorf("backends.lib_dir must be set")
	}

	if err := ValidDevice(c.Devices.Default); err != nil {
		return fmt.Errorf("devices.default: %w", err)
	}
	if err := ValidNUMA(c.Devices.NUMA); err != nil {
		return fmt.Errorf("devices.numa: %w", err)
	}
	if c.Devices.CPUThreadsBatch < 0 {
		return fmt.Errorf("devices.cpu_threads_batch must not be negative, got %d", c.Devices.CPUThreadsBatch)
	}
	if c.Devices.CPUThreads < 0 {
		return fmt.Errorf("devices.cpu_threads must not be negative, got %d", c.Devices.CPUThreads)
	}

	if strings.TrimSpace(c.Models.Default) == "" {
		return fmt.Errorf("models.default must name a model")
	}
	for i, id := range c.Models.Preload {
		if strings.TrimSpace(id) == "" {
			return fmt.Errorf("models.preload[%d] is empty", i)
		}
	}
	for id, o := range c.Models.Overrides {
		if err := o.validate(); err != nil {
			return fmt.Errorf("models.overrides.%s: %w", id, err)
		}
	}
	if c.Pool.MaxPending < 0 {
		return fmt.Errorf("pool.max_pending must not be negative, got %d", c.Pool.MaxPending)
	}
	if c.Pool.MaxResidentModels < 1 {
		return fmt.Errorf("pool.max_resident_models must be at least 1, got %d", c.Pool.MaxResidentModels)
	}
	if len(c.Models.Preload) > c.Pool.MaxResidentModels {
		// The pool would evict the first preload to make room for the last,
		// and startup would report every one of them as loaded.
		return fmt.Errorf("models.preload names %d models but pool.max_resident_models is %d",
			len(c.Models.Preload), c.Pool.MaxResidentModels)
	}
	if c.Cache.MaxSnapshots < 0 {
		return fmt.Errorf("cache.max_snapshots must not be negative, got %d", c.Cache.MaxSnapshots)
	}
	if c.Cache.PromptMaxBytes < 0 {
		return fmt.Errorf("cache.prompt_max_bytes must not be negative, got %d", c.Cache.PromptMaxBytes)
	}
	if c.Cache.PromptMinTokens < 0 {
		return fmt.Errorf("cache.prompt_min_tokens must not be negative, got %d", c.Cache.PromptMinTokens)
	}
	if c.Cache.PromptDisk && c.Cache.PromptMaxBytes == 0 {
		return fmt.Errorf("cache.prompt_disk is on but cache.prompt_max_bytes is 0: nothing could be kept")
	}
	if c.Generation.MaxTokensCap <= 0 {
		return fmt.Errorf("generation.max_tokens_cap must be positive, got %d", c.Generation.MaxTokensCap)
	}
	switch c.Generation.ContextOverflow {
	case OverflowError, OverflowTruncateMiddle:
	default:
		return fmt.Errorf("generation.context_overflow must be %s or %s, got %q",
			OverflowError, OverflowTruncateMiddle, c.Generation.ContextOverflow)
	}
	if c.Queue.Enabled && c.Queue.Concurrency <= 0 {
		return fmt.Errorf("queue.concurrency must be positive when the queue is enabled, got %d", c.Queue.Concurrency)
	}
	if c.Queue.Enabled && c.Queue.MaxPending <= 0 {
		return fmt.Errorf("queue.max_pending must be positive when the queue is enabled, got %d",
			c.Queue.MaxPending)
	}

	switch c.Log.Level {
	case "debug", "info", "warn", "error":
	default:
		return fmt.Errorf("log.level must be debug, info, warn or error, got %q", c.Log.Level)
	}
	switch c.Log.Format {
	case "json", "text":
	default:
		return fmt.Errorf("log.format must be json or text, got %q", c.Log.Format)
	}
	return nil
}

func (c *Config) validateAuth() error {
	switch c.Auth.Mode {
	case "apikey":
		// A server in apikey mode with no keys answers 401 to everything: it
		// is running, it looks configured, and it is useless. Say so at
		// startup rather than at the first request.
		if len(c.Auth.Keys) == 0 {
			return fmt.Errorf("auth.mode=apikey but no keys are configured; " +
				"set auth.keys or NANOLLM_AUTH_KEYS, or use auth.mode=open on a loopback address")
		}
		names := map[string]bool{}
		for i, k := range c.Auth.Keys {
			if strings.TrimSpace(k.Secret) == "" {
				return fmt.Errorf("auth.keys[%d] (%q) has no secret", i, k.Name)
			}
			if k.Name != "" {
				if names[k.Name] {
					return fmt.Errorf("auth.keys[%d]: the name %q is used twice", i, k.Name)
				}
				names[k.Name] = true
			}
			if k.RPS < 0 {
				return fmt.Errorf("auth.keys[%d] (%q): rps must not be negative", i, k.Name)
			}
		}
	case "open":
		// An unauthenticated listener on a routable address is not a
		// configuration mistake we are willing to boot with.
		if !IsLoopback(c.Server.Addr) {
			return fmt.Errorf("auth.mode=open requires a loopback listen address, got %q", c.Server.Addr)
		}
	default:
		return fmt.Errorf("auth.mode must be apikey or open, got %q", c.Auth.Mode)
	}
	return nil
}

func (o ModelOverride) validate() error {
	if o.Preset != "" {
		if _, err := plan.LookupPreset(o.Preset); err != nil {
			return fmt.Errorf("preset: %w", err)
		}
	}
	if _, err := plan.ParseMode(o.Mode); err != nil {
		return fmt.Errorf("mode: %w", err)
	}
	switch o.KVType {
	case "", KVF16, KVQ8_0:
	default:
		return fmt.Errorf("kv_type must be %s or %s, got %q", KVF16, KVQ8_0, o.KVType)
	}
	switch o.Speculative.Type {
	case "", SpecMTP, SpecNone:
	default:
		return fmt.Errorf("speculative.type must be %s or %s, got %q", SpecMTP, SpecNone, o.Speculative.Type)
	}
	if o.Speculative.NMax < 0 {
		return fmt.Errorf("speculative.n_max must not be negative, got %d", o.Speculative.NMax)
	}
	if p := o.Speculative.PMin; p < 0 || p >= 1 {
		return fmt.Errorf("speculative.p_min must be in [0, 1), got %v", p)
	}
	if o.Stream.Ahead < 0 || o.Stream.Keep < 0 {
		return fmt.Errorf("stream.ahead and stream.keep must not be negative, got %d and %d", o.Stream.Ahead, o.Stream.Keep)
	}
	return nil
}

// ValidDevice checks a device selector: auto, cpu, gpu, npu, or
// "<backend>:<device>" naming one device of one backend.
func ValidDevice(s string) error {
	switch s {
	case DeviceAuto, DeviceCPU, DeviceGPU, DeviceNPU:
		return nil
	}
	backend, dev, ok := strings.Cut(s, ":")
	if !ok || dev == "" {
		return fmt.Errorf("%q is not auto, cpu, gpu, npu or <backend>:<device>", s)
	}
	if !knownBackend(backend) {
		return fmt.Errorf("%q names unknown backend %q", s, backend)
	}
	return nil
}

func knownBackend(name string) bool {
	switch name {
	case BackendLlamaCpp, BackendORTGenAI, BackendMLX, BackendFake:
		return true
	}
	return false
}

// IsLoopback answers whether addr binds only this machine. It fails closed: an
// empty host (":8080") is what http.Server listens on every interface with, and
// so are the unspecified addresses.
func IsLoopback(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		host = addr
	}
	if host == "" {
		return false
	}
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(strings.Trim(host, "[]"))
	return ip != nil && ip.IsLoopback()
}

// ValidNUMA accepts devices.numa: empty or auto, off, interleave, node:N.
func ValidNUMA(s string) error {
	switch {
	case s == "", s == "auto", s == "off", s == "interleave":
		return nil
	case strings.HasPrefix(s, "node:"):
		if n, err := strconv.Atoi(strings.TrimPrefix(s, "node:")); err == nil && n >= 0 {
			return nil
		}
	}
	return fmt.Errorf("%q is not auto, off, interleave or node:N", s)
}
