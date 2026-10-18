package web

import (
	"encoding/csv"
	"fmt"
	"net/http"
	"strconv"

	"github.com/usunrise88/nanollm/services/bench/internal/store"
)

// export serves the open data: the aggregate groups (what=results) or every
// published run (what=runs), as CSV or JSON, under the data licence. Neither
// carries install keys or fingerprints.
func (s *Server) export(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	format, what := q.Get("format"), q.Get("what")
	if format == "" {
		format = "csv"
	}
	if what == "" {
		what = "results"
	}
	if (format != "csv" && format != "json") || (what != "results" && what != "runs") {
		writeErr(w, http.StatusBadRequest, "format is csv or json, what is results or runs")
		return
	}
	f := s.filter(r)
	var header []string
	var rows [][]string
	var objects []any
	switch what {
	case "results":
		groups, err := store.Results(r.Context(), s.st.DB, f)
		if err != nil {
			s.fail(w, r, err)
			return
		}
		header = []string{"model", "suite", "preset", "system", "system_slug", "cpu", "cpu_slug", "gpu", "gpu_slug", "os", "form_factor", "vendor_verified", "installs"}
		for _, m := range store.AggMetrics {
			header = append(header, m+"_median", m+"_p25", m+"_p75", m+"_n")
		}
		for _, g := range groups {
			row := []string{g.Model, g.Suite, g.Preset, g.System, g.SystemSlug, g.CPU, g.CPUSlug, g.GPU, g.GPUSlug, g.OS, g.FormFactor, strconv.FormatBool(g.Verified), strconv.Itoa(g.Installs)}
			for _, m := range store.AggMetrics {
				st, ok := g.Metrics[m]
				if !ok {
					row = append(row, "", "", "", "0")
					continue
				}
				row = append(row, num(st.Median), num(st.P25), num(st.P75), strconv.Itoa(st.N))
			}
			rows = append(rows, row)
			objects = append(objects, g)
		}
	case "runs":
		views, err := store.PublishedViews(r.Context(), s.st.DB, f.Suite, 100000)
		if err != nil {
			s.fail(w, r, err)
			return
		}
		header = []string{"id", "date", "suite", "model", "model_sha256", "nanollm", "llama_cpp_tag", "ggml_cpu_variant", "system", "cpu", "gpu", "os", "form_factor", "vendor_verified",
			"preset", "status", "mode", "backend", "gpu_layers", "mtp", "load_ms", "pp_tok_s", "tg_d0", "tg_dn", "ttft_ms", "chat_tok_s", "mtp_speedup", "mtp_acceptance", "plausibility"}
		for _, v := range views {
			objects = append(objects, v)
			for _, run := range v.Runs {
				rows = append(rows, []string{v.ID, v.Date, v.Suite, v.Model, v.ModelSHA256, v.NanoLLM, v.LlamaTag, v.CPUVariant, v.System, v.CPU, v.GPU, v.OS, v.FormFactor,
					strconv.FormatBool(v.Verified), run.Preset, run.Status, run.Mode, run.Backend, strconv.Itoa(run.GPULayers), strconv.FormatBool(run.MTP),
					num(run.LoadMS), num(run.PPTokS), num(run.TGD0), num(run.TGDN), num(run.TTFTMS), num(run.ChatTokS), num(run.MTPSpeedup), num(run.MTPAcceptance), run.Plausibility})
			}
		}
	}
	day := store.Today()
	attribution := fmt.Sprintf("NanoLLM public benchmark, %s/benchmarks", s.cfg.PublicURL)
	w.Header().Set("X-Data-License", s.cfg.DataLicense)
	if s.cfg.DataLicenseURL != "" {
		w.Header().Set("Link", fmt.Sprintf("<%s>; rel=\"license\"", s.cfg.DataLicenseURL))
	}
	name := fmt.Sprintf("nanollm-bench-%s-%s.%s", what, day, format)
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", name))
	if format == "json" {
		if objects == nil {
			objects = []any{}
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"license": s.cfg.DataLicense, "license_url": s.cfg.DataLicenseURL, "attribution": attribution,
			"generated": day, "suite": f.Suite, "what": what, "rows": objects,
		})
		return
	}
	w.Header().Set("Content-Type", "text/csv; charset=utf-8")
	_, _ = fmt.Fprintf(w, "# %s. License: %s %s. Generated %s. Suite %s.\n", attribution, s.cfg.DataLicense, s.cfg.DataLicenseURL, day, f.Suite)
	cw := csv.NewWriter(w)
	_ = cw.Write(header)
	_ = cw.WriteAll(rows)
}

func num(v float64) string {
	if v == 0 {
		return ""
	}
	return strconv.FormatFloat(v, 'f', -1, 64)
}
