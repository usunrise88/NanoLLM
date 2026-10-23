package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"path"
	"strings"
	"time"

	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/httpx"
	"github.com/usunrise88/nanollm/internal/registry"
)

// neverExpires is expires_at of a resident model: Ollama unloads an idle model
// after keep_alive, this server only when another model needs the memory.
var neverExpires = time.Date(2299, time.December, 31, 0, 0, 0, 0, time.UTC)

// capabilities is what /api/show says the model can do. There is no "vision"
// (images are refused) and no "embedding".
var capabilities = []string{"completion", "tools", "thinking"}

// local is a model that can be used now: complete on disk, or resident.
type local struct {
	m      registry.Manifest
	size   int64
	loaded []instance
}

// instance is the part of a resident model's pool status that /api/ps reports.
// core.ModelInfo carries it as an opaque value, so it is read back through its
// JSON rather than by importing the pool.
type instance struct {
	Key struct {
		NCtx int `json:"n_ctx"`
	} `json:"key"`
	Memory struct {
		Devices []struct {
			Device  string `json:"device"`
			Weights int64  `json:"weights"`
			KV      int64  `json:"kv"`
			Compute int64  `json:"compute"`
		} `json:"devices"`
	} `json:"memory"`
}

func (h *handler) locals(ctx context.Context) ([]local, error) {
	if h.deps.Admin == nil {
		// Nothing says what is on disk, so every catalog model is offered.
		out := make([]local, 0, len(h.deps.Catalog))
		for _, m := range h.deps.Catalog {
			l := local{m: m}
			for _, f := range m.Files {
				l.size += f.Size
			}
			out = append(out, l)
		}
		return out, nil
	}
	infos, err := h.deps.Admin.Models(ctx)
	if err != nil {
		return nil, err
	}
	out := []local{}
	for _, info := range infos {
		if info.Disk != registry.StateComplete && len(info.Loaded) == 0 {
			continue
		}
		m, ok := registry.Find(h.deps.Catalog, info.ID)
		if !ok {
			m = registry.Manifest{ID: info.ID, Family: info.Family, Params: info.Params}
		}
		l := local{m: m, size: info.SizeBytes}
		for _, v := range info.Loaded {
			b, err := json.Marshal(v)
			if err != nil {
				continue
			}
			var in instance
			if json.Unmarshal(b, &in) == nil {
				l.loaded = append(l.loaded, in)
			}
		}
		out = append(out, l)
	}
	return out, nil
}

// find is the local model a client's name stands for.
func (h *handler) find(ctx context.Context, name string) (local, error) {
	m, err := h.resolve(name)
	if err != nil {
		return local{}, err
	}
	all, err := h.locals(ctx)
	if err != nil {
		return local{}, err
	}
	for _, l := range all {
		if l.m.ID == m.ID {
			return l, nil
		}
	}
	return local{}, core.Errorf(core.CodeModelNotFound, "model '%s' not found, try pulling it first", name).WithParam("model")
}

type modelDetails struct {
	ParentModel       string   `json:"parent_model"`
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

func detailsOf(m registry.Manifest) modelDetails {
	family := m.Family
	if family == "" {
		family = m.Arch
	}
	format := m.Format
	if format == "" {
		format = registry.FormatGGUF
	}
	return modelDetails{Format: format, Family: family, Families: []string{family},
		ParameterSize: m.Params, QuantizationLevel: Quantization}
}

type tagEntry struct {
	Name       string       `json:"name"`
	Model      string       `json:"model"`
	ModifiedAt time.Time    `json:"modified_at"`
	Size       int64        `json:"size"`
	Digest     string       `json:"digest"`
	Details    modelDetails `json:"details"`
}

func (h *handler) tags(w http.ResponseWriter, r *http.Request) {
	all, err := h.locals(r.Context())
	if err != nil {
		writeError(w, err)
		return
	}
	models := make([]tagEntry, 0, len(all))
	for _, l := range all {
		models = append(models, tagEntry{Name: tagged(l.m.ID), Model: tagged(l.m.ID), ModifiedAt: h.started,
			Size: l.size, Digest: digestOf(l.m), Details: detailsOf(l.m)})
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]any{"models": models})
}

type runningEntry struct {
	Name          string       `json:"name"`
	Model         string       `json:"model"`
	Size          int64        `json:"size"`
	Digest        string       `json:"digest"`
	Details       modelDetails `json:"details"`
	ExpiresAt     time.Time    `json:"expires_at"`
	SizeVRAM      int64        `json:"size_vram"`
	ContextLength int          `json:"context_length,omitempty"`
}

func (h *handler) ps(w http.ResponseWriter, r *http.Request) {
	models := []runningEntry{}
	if h.deps.Admin != nil {
		all, err := h.locals(r.Context())
		if err != nil {
			writeError(w, err)
			return
		}
		for _, l := range all {
			if len(l.loaded) == 0 {
				continue
			}
			e := runningEntry{Name: tagged(l.m.ID), Model: tagged(l.m.ID), Digest: digestOf(l.m),
				Details: detailsOf(l.m), ExpiresAt: neverExpires}
			// Size is what the instances hold in memory, size_vram the part of
			// it that is not on the CPU.
			for _, in := range l.loaded {
				e.ContextLength = max(e.ContextLength, in.Key.NCtx)
				for _, d := range in.Memory.Devices {
					n := d.Weights + d.KV + d.Compute
					e.Size += n
					if d.Device != "cpu" {
						e.SizeVRAM += n
					}
				}
			}
			if e.Size == 0 {
				e.Size = l.size
			}
			models = append(models, e)
		}
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]any{"models": models})
}

func (h *handler) show(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model string `json:"model"`
		Name  string `json:"name"` // the field older clients use
	}
	if err := decode(r, &req); err != nil {
		writeError(w, err)
		return
	}
	name := req.Model
	if name == "" {
		name = req.Name
	}
	l, err := h.find(r.Context(), name)
	if err != nil {
		writeError(w, err)
		return
	}
	m := l.m
	info := map[string]any{}
	if m.Arch != "" {
		info["general.architecture"] = m.Arch
		if m.ContextMax > 0 {
			info[m.Arch+".context_length"] = m.ContextMax
		}
	}
	// The model card's sampling, which a request that sets nothing gets.
	var params []string
	if d := m.SamplingDefaults; d != nil {
		add := func(name string, v any) { params = append(params, fmt.Sprintf("%-30s %v", name, v)) }
		if d.Temperature != nil {
			add("temperature", *d.Temperature)
		}
		if d.TopK != nil {
			add("top_k", *d.TopK)
		}
		if d.TopP != nil {
			add("top_p", *d.TopP)
		}
		if d.MinP != nil {
			add("min_p", *d.MinP)
		}
		if d.RepeatPenalty != nil {
			add("repeat_penalty", *d.RepeatPenalty)
		}
	}
	modelfile := "# Modelfile generated by nanollm\nFROM " + m.ID + "\n"
	for _, p := range params {
		modelfile += "PARAMETER " + p + "\n"
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]any{
		"license":      m.License,
		"modelfile":    modelfile,
		"parameters":   strings.Join(params, "\n"),
		"template":     "",
		"details":      detailsOf(m),
		"model_info":   info,
		"capabilities": capabilities,
		"modified_at":  h.started,
	})
}

// --- /api/pull ---

type pullLine struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty"`
	Completed int64  `json:"completed,omitempty"`
}

func (h *handler) pull(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model  string `json:"model"`
		Name   string `json:"name"`
		Stream *bool  `json:"stream"`
	}
	if err := decode(r, &req); err != nil {
		writeError(w, err)
		return
	}
	name := req.Model
	if name == "" {
		name = req.Name
	}
	m, err := h.resolve(name)
	if err != nil {
		writeError(w, err)
		return
	}
	if h.deps.Admin == nil {
		writeError(w, core.Errorf(core.CodeNotImplemented, "model management is not available in this configuration"))
		return
	}
	if !h.mayManage(r) {
		writeError(w, core.Errorf(core.CodeForbidden, "pulling a model needs an administrative key"))
		return
	}
	if req.Stream != nil && !*req.Stream {
		if err := h.deps.Admin.Download(r.Context(), m.ID, nil); err != nil {
			writeError(w, err)
			return
		}
		httpx.WriteJSON(w, http.StatusOK, pullLine{Status: "success"})
		return
	}

	st, err := httpx.NewNDJSON(w)
	if err != nil {
		writeError(w, err)
		return
	}
	defer st.Close()
	digests := map[string]string{}
	for _, f := range m.Files {
		digests[path.Base(f.Path)] = f.SHA256
	}
	progress := make(chan core.DownloadProgress, 16)
	done := make(chan error, 1)
	go func() {
		done <- h.deps.Admin.Download(r.Context(), m.ID, func(p core.DownloadProgress) {
			select {
			case progress <- p:
			default: // a slow reader misses an update, nothing more
			}
		})
	}()
	last := pullLine{Status: "pulling manifest"}
	if st.Line(last) != nil {
		return
	}
	var sent time.Time
	// emit reports one update; a line per read would be thousands a second,
	// so only the last of a burst goes out, and the end of a file always does.
	emit := func(p core.DownloadProgress) bool {
		if p.Downloaded < p.Total && time.Since(sent) < 250*time.Millisecond {
			return true
		}
		sent = time.Now()
		digest := digests[p.File]
		last = pullLine{Status: "pulling " + digest[:min(12, len(digest))], Total: p.Total, Completed: p.Downloaded}
		if digest != "" {
			last.Digest = "sha256:" + digest
		}
		return st.Line(last) == nil
	}
	for {
		select {
		case p := <-progress:
			if !emit(p) {
				return
			}
		case err := <-done:
			if err != nil {
				if !errors.Is(err, context.Canceled) {
					_ = st.Line(errorBody{Error: core.AsError(err).Message})
				}
				return
			}
			// What the download reported before it returned.
			for drained := false; !drained; {
				select {
				case p := <-progress:
					if !emit(p) {
						return
					}
				default:
					drained = true
				}
			}
			for _, status := range []string{"verifying sha256 digest", "writing manifest", "success"} {
				if st.Line(pullLine{Status: status}) != nil {
					return
				}
			}
			return
		case <-st.Heartbeat():
			// Quiet while a file is verified: say what the last line said.
			if st.Line(last) != nil {
				return
			}
		}
	}
}
