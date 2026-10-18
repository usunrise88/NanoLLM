package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"

	"github.com/usunrise88/nanollm/benchfmt"
	"github.com/usunrise88/nanollm/services/bench/internal/normalize"
)

// Plausibility verdicts.
const (
	PlausibleOK      = "ok"
	PlausibleNo      = "implausible"
	PlausibleUnknown = "unknown"
)

// NormRun is a normalized run with what the indexer concluded about it.
type NormRun struct {
	normalize.Run
	Plausibility string
	Ceiling      float64
	Cohort       string
	// OutputMatch is the longest output prefix, in tokens, this run shares
	// with an earlier run of its cohort; -1 when nothing to compare with.
	OutputMatch int
	Outlier     bool
}

// ClearDerived empties the derived tables, for a reindex.
func ClearDerived(ctx context.Context, q Querier) error {
	for _, t := range []string{"aggregates", "norm_runs", "norm_submissions"} {
		if _, err := q.ExecContext(ctx, `DELETE FROM `+t); err != nil {
			return err
		}
	}
	return nil
}

// WriteNorm replaces the derived rows of one submission.
func WriteNorm(ctx context.Context, q Querier, id string, rec normalize.Record, verified bool, runs []NormRun) error {
	if _, err := q.ExecContext(ctx, `DELETE FROM norm_runs WHERE submission_id = ?`, id); err != nil {
		return err
	}
	if _, err := q.ExecContext(ctx, `DELETE FROM norm_submissions WHERE id = ?`, id); err != nil {
		return err
	}
	_, err := q.ExecContext(ctx, `INSERT INTO norm_submissions (id, normalizer, date, suite, model_id, model_sha256, nanollm, llama_tag,
		cpu_variant, repack, cpu, cpu_slug, gpu, gpu_slug, system, system_slug, system_vendor, form_factor, os, install_id, fingerprint, verified, user_entered)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		id, rec.Normalizer, rec.Date, rec.Suite, rec.ModelID, rec.ModelSHA256, rec.NanoLLM, rec.LlamaTag,
		rec.CPUVariant, rec.Repack, rec.CPU, rec.CPUSlug, rec.GPU, rec.GPUSlug, rec.System, rec.SystemSlug, rec.SystemVendor,
		rec.FormFactor, rec.OS, rec.InstallID, rec.Fingerprint, verified, rec.UserEntered)
	if err != nil {
		return err
	}
	for _, r := range runs {
		pfx := ""
		if len(r.Prefixes) > 0 {
			b, _ := json.Marshal(r.Prefixes)
			pfx = string(b)
		}
		_, err := q.ExecContext(ctx, `INSERT INTO norm_runs (submission_id, preset, status, reason, mode, backend, gpu_layers, mtp,
			load_ms, pp_tok_s, tg_d0, tg_dn, ttft_ms, chat_tok_s, mtp_speedup, mtp_acceptance,
			plausibility, ceiling_tok_s, cohort, output_sha256, output_tokens, output_prefixes, output_match, outlier)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT (submission_id, preset) DO NOTHING`,
			id, r.Preset, r.Status, r.Reason, r.Mode, r.Backend, r.GPULayers, r.MTP,
			r.LoadMS, r.PPTokS, r.TGD0, r.TGDN, r.TTFTMS, r.ChatTokS, r.MTPSpeedup, r.MTPAcceptance,
			r.Plausibility, r.Ceiling, r.Cohort, r.OutputSHA256, r.OutputTokens, pfx, r.OutputMatch, r.Outlier)
		if err != nil {
			return err
		}
	}
	return nil
}

// CohortOutput is an earlier run's output fingerprint.
type CohortOutput struct {
	SHA256   string
	Tokens   int
	Prefixes []benchfmt.Prefix
}

// CohortOutputs lists the output fingerprints of a cohort's indexed,
// non-hidden runs other than one submission's.
func CohortOutputs(ctx context.Context, q Querier, cohort, exclude string) ([]CohortOutput, error) {
	rows, err := q.QueryContext(ctx, `SELECT r.output_sha256, r.output_tokens, r.output_prefixes FROM norm_runs r
		JOIN moderation m ON m.submission_id = r.submission_id
		WHERE r.cohort = ? AND r.submission_id != ? AND r.output_sha256 != '' AND m.state != 'hidden' LIMIT 500`, cohort, exclude)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []CohortOutput
	for rows.Next() {
		var c CohortOutput
		var pfx string
		if err := rows.Scan(&c.SHA256, &c.Tokens, &pfx); err != nil {
			return nil, err
		}
		if pfx != "" {
			_ = json.Unmarshal([]byte(pfx), &c.Prefixes)
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

// UnitValue is one run's value of a metric with who produced it.
type UnitValue struct {
	InstallID   string
	Fingerprint string
	Value       float64
}

// HWCohort is the group a run is compared with for MAD outliers: one model
// file, suite, preset, placement and hardware.
type HWCohort struct {
	Suite, ModelSHA256, Preset, Mode, Backend, CPUSlug, GPUSlug string
}

// metricColumns are the run columns a metric name may refer to.
var metricColumns = map[string]string{
	"tg_d0": "tg_d0", "tg_dn": "tg_dn", "pp_tok_s": "pp_tok_s", "chat_tok_s": "chat_tok_s",
	"ttft_ms": "ttft_ms", "load_ms": "load_ms", "mtp_speedup": "mtp_speedup",
}

// CohortValues lists the published, plausible values of a metric in a
// hardware cohort, other than one submission's.
func CohortValues(ctx context.Context, q Querier, c HWCohort, metric, exclude string) ([]UnitValue, error) {
	col, ok := metricColumns[metric]
	if !ok {
		return nil, errors.New("unknown metric " + metric)
	}
	rows, err := q.QueryContext(ctx, `SELECT n.install_id, n.fingerprint, r.`+col+` FROM norm_runs r
		JOIN norm_submissions n ON n.id = r.submission_id JOIN moderation m ON m.submission_id = r.submission_id
		WHERE m.state = 'published' AND r.status = 'ok' AND r.plausibility != 'implausible' AND r.`+col+` > 0
		AND n.suite = ? AND n.model_sha256 = ? AND r.preset = ? AND r.mode = ? AND r.backend = ? AND n.cpu_slug = ? AND n.gpu_slug = ?
		AND r.submission_id != ?`, c.Suite, c.ModelSHA256, c.Preset, c.Mode, c.Backend, c.CPUSlug, c.GPUSlug, exclude)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []UnitValue
	for rows.Next() {
		var u UnitValue
		if err := rows.Scan(&u.InstallID, &u.Fingerprint, &u.Value); err != nil {
			return nil, err
		}
		out = append(out, u)
	}
	return out, rows.Err()
}

// NormalizerVersions reports the oldest normalizer version among the
// derived rows and whether every submission has them.
func NormalizerVersions(ctx context.Context, q Querier) (oldest int, complete bool, err error) {
	var subs, norms int
	var v sql.NullInt64
	if err = q.QueryRowContext(ctx, `SELECT COUNT(*) FROM submissions`).Scan(&subs); err != nil {
		return
	}
	if err = q.QueryRowContext(ctx, `SELECT COUNT(*), MIN(normalizer) FROM norm_submissions`).Scan(&norms, &v); err != nil {
		return
	}
	return int(v.Int64), subs == norms, nil
}

// AggRow is one input row of the aggregates: a published, plausible ok run
// with its group, unit and metric values.
type AggRow struct {
	ModelID, Suite, Preset, CPUSlug, GPUSlug, SystemSlug, OS, FormFactor string
	CPU, GPU, System                                                     string
	Verified                                                             bool
	InstallID, Fingerprint                                               string
	Metrics                                                              map[string]float64
}

// AggMetrics are the metrics aggregated, in display order.
var AggMetrics = []string{"tg_d0", "tg_dn", "pp_tok_s", "chat_tok_s", "ttft_ms", "load_ms", "mtp_speedup"}

// AggInputs reads every run the aggregates count.
func AggInputs(ctx context.Context, q Querier) ([]AggRow, error) {
	rows, err := q.QueryContext(ctx, `SELECT n.model_id, n.suite, r.preset, n.cpu_slug, n.gpu_slug, n.system_slug, n.os, n.form_factor,
		n.cpu, n.gpu, n.system, n.verified, n.install_id, n.fingerprint,
		r.tg_d0, r.tg_dn, r.pp_tok_s, r.chat_tok_s, r.ttft_ms, r.load_ms, r.mtp_speedup
		FROM norm_runs r JOIN norm_submissions n ON n.id = r.submission_id JOIN moderation m ON m.submission_id = r.submission_id
		WHERE m.state = 'published' AND r.status = 'ok' AND r.plausibility != 'implausible'`)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []AggRow
	for rows.Next() {
		var a AggRow
		v := make([]float64, len(AggMetrics))
		if err := rows.Scan(&a.ModelID, &a.Suite, &a.Preset, &a.CPUSlug, &a.GPUSlug, &a.SystemSlug, &a.OS, &a.FormFactor,
			&a.CPU, &a.GPU, &a.System, &a.Verified, &a.InstallID, &a.Fingerprint, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6]); err != nil {
			return nil, err
		}
		a.Metrics = map[string]float64{}
		for i, m := range AggMetrics {
			a.Metrics[m] = v[i]
		}
		out = append(out, a)
	}
	return out, rows.Err()
}

// AggOut is one row of the aggregates table.
type AggOut struct {
	ModelID, Suite, Preset, CPUSlug, GPUSlug, SystemSlug, OS, FormFactor string
	CPU, GPU, System                                                     string
	Verified                                                             bool
	Metric                                                               string
	N                                                                    int
	Median, P25, P75                                                     float64
}

// ReplaceAggregates rewrites the aggregates table.
func ReplaceAggregates(ctx context.Context, q Querier, rows []AggOut) error {
	if _, err := q.ExecContext(ctx, `DELETE FROM aggregates`); err != nil {
		return err
	}
	for _, a := range rows {
		_, err := q.ExecContext(ctx, `INSERT INTO aggregates (model_id, suite, preset, cpu_slug, gpu_slug, system_slug, os, form_factor,
			cpu, gpu, system, verified, metric, n, median, p25, p75) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			a.ModelID, a.Suite, a.Preset, a.CPUSlug, a.GPUSlug, a.SystemSlug, a.OS, a.FormFactor,
			a.CPU, a.GPU, a.System, a.Verified, a.Metric, a.N, a.Median, a.P25, a.P75)
		if err != nil {
			return err
		}
	}
	return nil
}
