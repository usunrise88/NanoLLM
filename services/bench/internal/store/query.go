package store

import (
	"context"
	"database/sql"
	"errors"
	"sort"
	"strings"
)

// Filter selects aggregate groups. Empty fields match everything; CPU, GPU
// and System are slugs.
type Filter struct {
	Model, Suite, Preset, CPU, GPU, System, OS, FormFactor string
}

// Stat is one metric of a group.
type Stat struct {
	N      int     `json:"n"`
	Median float64 `json:"median"`
	P25    float64 `json:"p25"`
	P75    float64 `json:"p75"`
}

// Group is one row of the results: a model, a suite, a preset and a
// machine configuration, with the medians over its units.
type Group struct {
	Model      string          `json:"model"`
	Suite      string          `json:"suite"`
	Preset     string          `json:"preset"`
	CPU        string          `json:"cpu"`
	CPUSlug    string          `json:"cpu_slug"`
	GPU        string          `json:"gpu"`
	GPUSlug    string          `json:"gpu_slug"`
	System     string          `json:"system"`
	SystemSlug string          `json:"system_slug"`
	OS         string          `json:"os"`
	FormFactor string          `json:"form_factor"`
	Verified   bool            `json:"vendor_verified"`
	Installs   int             `json:"installs"`
	Metrics    map[string]Stat `json:"metrics"`
}

// Results reads the aggregate groups a filter selects, fastest tg first.
func Results(ctx context.Context, q Querier, f Filter) ([]Group, error) {
	where := []string{"1 = 1"}
	var args []any
	for col, v := range map[string]string{
		"model_id": f.Model, "suite": f.Suite, "preset": f.Preset, "cpu_slug": f.CPU, "gpu_slug": f.GPU,
		"system_slug": f.System, "os": f.OS, "form_factor": f.FormFactor,
	} {
		if v != "" {
			where = append(where, col+" = ?")
			args = append(args, v)
		}
	}
	rows, err := q.QueryContext(ctx, `SELECT model_id, suite, preset, cpu_slug, gpu_slug, system_slug, os, form_factor, cpu, gpu, system, verified,
		metric, n, median, p25, p75 FROM aggregates WHERE `+strings.Join(where, " AND ")+` ORDER BY model_id, suite, preset, system_slug, cpu_slug, gpu_slug, os, form_factor`, args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []Group
	idx := map[string]int{}
	for rows.Next() {
		var g Group
		var metric string
		var s Stat
		if err := rows.Scan(&g.Model, &g.Suite, &g.Preset, &g.CPUSlug, &g.GPUSlug, &g.SystemSlug, &g.OS, &g.FormFactor, &g.CPU, &g.GPU, &g.System, &g.Verified,
			&metric, &s.N, &s.Median, &s.P25, &s.P75); err != nil {
			return nil, err
		}
		key := strings.Join([]string{g.Model, g.Suite, g.Preset, g.CPUSlug, g.GPUSlug, g.SystemSlug, g.OS, g.FormFactor}, "\x00")
		i, ok := idx[key]
		if !ok {
			g.Metrics = map[string]Stat{}
			out = append(out, g)
			i = len(out) - 1
			idx[key] = i
		}
		out[i].Metrics[metric] = s
		if s.N > out[i].Installs {
			out[i].Installs = s.N
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Metrics["tg_d0"].Median > out[j].Metrics["tg_d0"].Median })
	return out, nil
}

// Facet is one value of a filter with how many installs stand behind it.
type Facet struct {
	Value    string `json:"value"`
	Label    string `json:"label"`
	Installs int    `json:"installs"`
}

// Facets are the values each filter can take.
type Facets struct {
	Suites      []Facet `json:"suites"`
	Models      []Facet `json:"models"`
	Presets     []Facet `json:"presets"`
	CPUs        []Facet `json:"cpus"`
	GPUs        []Facet `json:"gpus"`
	Systems     []Facet `json:"systems"`
	OS          []Facet `json:"os"`
	FormFactors []Facet `json:"form_factors"`
}

// ReadFacets lists the filter values within a suite.
func ReadFacets(ctx context.Context, q Querier, suite string) (Facets, error) {
	fc := Facets{Suites: []Facet{}}
	rows, err := q.QueryContext(ctx, `SELECT suite, SUM(n) FROM aggregates WHERE metric = 'tg_d0' GROUP BY suite ORDER BY suite`)
	if err != nil {
		return fc, err
	}
	for rows.Next() {
		var f Facet
		if err := rows.Scan(&f.Value, &f.Installs); err != nil {
			_ = rows.Close()
			return fc, err
		}
		f.Label = f.Value
		fc.Suites = append(fc.Suites, f)
	}
	_ = rows.Close()
	if err := rows.Err(); err != nil {
		return fc, err
	}
	groups, err := Results(ctx, q, Filter{Suite: suite})
	if err != nil {
		return fc, err
	}
	count := func(value, label func(Group) string) []Facet {
		m := map[string]*Facet{}
		var order []string
		for _, g := range groups {
			v := value(g)
			f, ok := m[v]
			if !ok {
				f = &Facet{Value: v, Label: label(g)}
				m[v] = f
				order = append(order, v)
			}
			f.Installs += g.Installs
		}
		out := make([]Facet, 0, len(order))
		for _, v := range order {
			out = append(out, *m[v])
		}
		sort.Slice(out, func(i, j int) bool { return strings.ToLower(out[i].Label) < strings.ToLower(out[j].Label) })
		return out
	}
	same := func(f func(Group) string) (func(Group) string, func(Group) string) { return f, f }
	fc.Models = count(same(func(g Group) string { return g.Model }))
	fc.Presets = count(same(func(g Group) string { return g.Preset }))
	fc.CPUs = count(func(g Group) string { return g.CPUSlug }, func(g Group) string { return g.CPU })
	fc.GPUs = count(func(g Group) string { return g.GPUSlug }, func(g Group) string { return g.GPU })
	fc.Systems = count(func(g Group) string { return g.SystemSlug }, func(g Group) string { return g.System })
	fc.OS = count(same(func(g Group) string { return g.OS }))
	fc.FormFactors = count(same(func(g Group) string { return g.FormFactor }))
	return fc, nil
}

// RunView is a normalized run as the pages and the API show it.
type RunView struct {
	Preset        string  `json:"preset"`
	Status        string  `json:"status"`
	Reason        string  `json:"reason,omitempty"`
	Mode          string  `json:"mode,omitempty"`
	Backend       string  `json:"backend,omitempty"`
	GPULayers     int     `json:"gpu_layers"`
	MTP           bool    `json:"mtp"`
	LoadMS        float64 `json:"load_ms,omitempty"`
	PPTokS        float64 `json:"pp_tok_s,omitempty"`
	TGD0          float64 `json:"tg_d0,omitempty"`
	TGDN          float64 `json:"tg_dn,omitempty"`
	TTFTMS        float64 `json:"ttft_ms,omitempty"`
	ChatTokS      float64 `json:"chat_tok_s,omitempty"`
	MTPSpeedup    float64 `json:"mtp_speedup,omitempty"`
	MTPAcceptance float64 `json:"mtp_acceptance,omitempty"`
	Plausibility  string  `json:"plausibility"`
	Ceiling       float64 `json:"ceiling_tok_s,omitempty"`
	OutputTokens  int     `json:"output_tokens,omitempty"`
	OutputMatch   int     `json:"output_match"`
	Outlier       bool    `json:"outlier"`
}

// SubmissionView is a normalized submission.
type SubmissionView struct {
	ID          string    `json:"id"`
	Date        string    `json:"date"`
	ReceivedOn  string    `json:"received_on"`
	State       string    `json:"state"`
	Suite       string    `json:"suite"`
	Model       string    `json:"model"`
	ModelSHA256 string    `json:"model_sha256"`
	NanoLLM     string    `json:"nanollm"`
	LlamaTag    string    `json:"llama_cpp_tag"`
	CPUVariant  string    `json:"ggml_cpu_variant,omitempty"`
	CPU         string    `json:"cpu"`
	CPUSlug     string    `json:"cpu_slug"`
	GPU         string    `json:"gpu"`
	GPUSlug     string    `json:"gpu_slug"`
	System      string    `json:"system"`
	SystemSlug  string    `json:"system_slug"`
	FormFactor  string    `json:"form_factor"`
	OS          string    `json:"os"`
	Verified    bool      `json:"vendor_verified"`
	UserEntered bool      `json:"user_entered_system,omitempty"`
	Runs        []RunView `json:"runs"`
}

const viewCols = `n.id, n.date, s.received_on, m.state, n.suite, n.model_id, n.model_sha256, n.nanollm, n.llama_tag, n.cpu_variant,
	n.cpu, n.cpu_slug, n.gpu, n.gpu_slug, n.system, n.system_slug, n.form_factor, n.os, n.verified, n.user_entered`

func scanView(sc interface{ Scan(...any) error }) (SubmissionView, error) {
	var v SubmissionView
	err := sc.Scan(&v.ID, &v.Date, &v.ReceivedOn, &v.State, &v.Suite, &v.Model, &v.ModelSHA256, &v.NanoLLM, &v.LlamaTag, &v.CPUVariant,
		&v.CPU, &v.CPUSlug, &v.GPU, &v.GPUSlug, &v.System, &v.SystemSlug, &v.FormFactor, &v.OS, &v.Verified, &v.UserEntered)
	return v, err
}

// GetView reads one normalized submission with its runs.
func GetView(ctx context.Context, q Querier, id string) (SubmissionView, error) {
	v, err := scanView(q.QueryRowContext(ctx, `SELECT `+viewCols+` FROM norm_submissions n JOIN submissions s ON s.id = n.id
		JOIN moderation m ON m.submission_id = n.id WHERE n.id = ?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return v, ErrNotFound
	}
	if err != nil {
		return v, err
	}
	runs, err := runViews(ctx, q, `WHERE submission_id = ?`, id)
	if err != nil {
		return v, err
	}
	v.Runs = runs[id]
	return v, nil
}

func runViews(ctx context.Context, q Querier, where string, args ...any) (map[string][]RunView, error) {
	rows, err := q.QueryContext(ctx, `SELECT submission_id, preset, status, reason, mode, backend, gpu_layers, mtp, load_ms, pp_tok_s, tg_d0, tg_dn,
		ttft_ms, chat_tok_s, mtp_speedup, mtp_acceptance, plausibility, ceiling_tok_s, output_tokens, output_match, outlier
		FROM norm_runs `+where+` ORDER BY submission_id, rowid`, args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	out := map[string][]RunView{}
	for rows.Next() {
		var id string
		var r RunView
		if err := rows.Scan(&id, &r.Preset, &r.Status, &r.Reason, &r.Mode, &r.Backend, &r.GPULayers, &r.MTP, &r.LoadMS, &r.PPTokS, &r.TGD0, &r.TGDN,
			&r.TTFTMS, &r.ChatTokS, &r.MTPSpeedup, &r.MTPAcceptance, &r.Plausibility, &r.Ceiling, &r.OutputTokens, &r.OutputMatch, &r.Outlier); err != nil {
			return nil, err
		}
		out[id] = append(out[id], r)
	}
	return out, rows.Err()
}

// PublishedViews lists the published submissions of a suite (all suites
// when empty), newest first, for the export and the recent list.
func PublishedViews(ctx context.Context, q Querier, suite string, limit int) ([]SubmissionView, error) {
	rows, err := q.QueryContext(ctx, `SELECT `+viewCols+` FROM norm_submissions n JOIN submissions s ON s.id = n.id
		JOIN moderation m ON m.submission_id = n.id WHERE m.state = 'published' AND (? = '' OR n.suite = ?)
		ORDER BY s.received_on DESC, s.rowid DESC LIMIT ?`, suite, suite, limit)
	if err != nil {
		return nil, err
	}
	var out []SubmissionView
	for rows.Next() {
		v, err := scanView(rows)
		if err != nil {
			_ = rows.Close()
			return nil, err
		}
		out = append(out, v)
	}
	_ = rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}
	runs, err := runViews(ctx, q, `WHERE submission_id IN (SELECT submission_id FROM moderation WHERE state = 'published')`)
	if err != nil {
		return nil, err
	}
	for i := range out {
		out[i].Runs = runs[out[i].ID]
	}
	return out, nil
}
