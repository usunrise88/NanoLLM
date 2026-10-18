// Package ingest accepts submissions and withdrawals and keeps the derived
// tables in step with the raw ones: the size limit, the format and the
// signature, the suite, the plausibility ceiling, rate limits, the output
// cohort signal, MAD outliers and moderation (docs/BENCHMARKS.md, "Доверие").
// Reindex replays every raw submission through the same indexing.
package ingest

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/base32"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"sync"

	"github.com/usunrise88/nanollm/benchfmt"
	"github.com/usunrise88/nanollm/services/bench/internal/aggregate"
	"github.com/usunrise88/nanollm/services/bench/internal/normalize"
	"github.com/usunrise88/nanollm/services/bench/internal/store"
)

// MaxBody is the largest submission accepted.
const MaxBody = 256 << 10

// Suite is an accepted suite version.
type Suite struct {
	// PromptsSHA256 is what the suite version fixes (internal/bench
	// PromptsSHA256 in NanoLLM); empty skips the check.
	PromptsSHA256 string
}

// StandardV1Prompts is internal/bench.PromptsSHA256() of NanoLLM for
// standard/v1. It changes only with a new suite version.
const StandardV1Prompts = "059d9eb2857d7450cb6f694c672b0732dc977731254196615c809a2c51ef318f"

// DefaultSuites are the suites accepted out of the box.
func DefaultSuites() map[string]Suite {
	return map[string]Suite{"standard/v1": {PromptsSHA256: StandardV1Prompts}}
}

// Moderation policies.
const (
	ModerationAuto   = "auto"   // publish unless a check queues it
	ModerationManual = "manual" // queue everything
)

// Config is the ingest policy.
type Config struct {
	PublicURL string
	// Suites accepted ("standard/v1"); Retired answer 410, any other
	// version 426.
	Suites  map[string]Suite
	Retired map[string]bool
	// CeilingSlack multiplies the plausibility ceiling.
	CeilingSlack float64
	Moderation   string
	// PerIP and PerKey limit submissions; nil means no limit.
	PerIP, PerKey *Limiter
}

// Service is the ingest pipeline over a store.
type Service struct {
	cfg Config
	st  *store.Store
	// mu serializes writes: indexing reads the cohort it joins.
	mu sync.Mutex
}

// New makes the service.
func New(st *store.Store, cfg Config) *Service {
	if cfg.Suites == nil {
		cfg.Suites = DefaultSuites()
	}
	if cfg.CeilingSlack < 1 {
		cfg.CeilingSlack = 1.3
	}
	if cfg.Moderation == "" {
		cfg.Moderation = ModerationAuto
	}
	cfg.PublicURL = strings.TrimSuffix(cfg.PublicURL, "/")
	return &Service{cfg: cfg, st: st}
}

// Error is a refusal with its HTTP status.
type Error struct {
	Status int
	Msg    string
}

func (e *Error) Error() string { return e.Msg }

func refuse(status int, format string, args ...any) error {
	return &Error{Status: status, Msg: fmt.Sprintf(format, args...)}
}

// Receipt is the answer to a submission. id and url are the client's
// contract; the rest is informative.
type Receipt struct {
	ID    string   `json:"id"`
	URL   string   `json:"url"`
	State string   `json:"state"`
	Flags []string `json:"flags,omitempty"`
	// Existing is set when the same bytes were submitted before.
	Existing bool `json:"existing,omitempty"`
}

// URLFor is the public page of a submission.
func (s *Service) URLFor(id string) string { return s.cfg.PublicURL + "/benchmarks/s/" + id }

var idRe = regexp.MustCompile(`^[a-z2-7]{13}$`)

// ValidID reports whether id is shaped like a submission id.
func ValidID(id string) bool { return idRe.MatchString(id) }

func newID() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(err)
	}
	return strings.ToLower(base32.StdEncoding.WithPadding(base32.NoPadding).EncodeToString(b[:]))
}

// Submit runs a submission through every check and stores it. ipKey is the
// limiter's hash of the sender's address ("" skips the per-address limit).
func (s *Service) Submit(ctx context.Context, body []byte, ipKey string) (*Receipt, error) {
	if ipKey != "" && !s.cfg.PerIP.Allow(ipKey) {
		return nil, refuse(http.StatusTooManyRequests, "too many submissions from this address; try again later")
	}
	if len(body) > MaxBody {
		return nil, refuse(http.StatusRequestEntityTooLarge, "a submission is at most %d bytes", MaxBody)
	}
	var head struct {
		Schema string `json:"schema"`
		Suite  struct {
			ID      string `json:"id"`
			Version string `json:"version"`
		} `json:"suite"`
	}
	if err := json.Unmarshal(body, &head); err != nil {
		return nil, refuse(http.StatusUnprocessableEntity, "not a JSON document: %v", err)
	}
	if head.Schema != benchfmt.Schema {
		if strings.HasPrefix(head.Schema, "nanollm.bench/") {
			return nil, refuse(http.StatusUpgradeRequired, "format %s is not accepted; this service reads %s", head.Schema, benchfmt.Schema)
		}
		return nil, refuse(http.StatusUnprocessableEntity, "schema is %q, want %q", head.Schema, benchfmt.Schema)
	}
	if head.Suite.ID != "standard" {
		return nil, refuse(http.StatusUnprocessableEntity, "only the standard suite is published; %q results stay local", head.Suite.ID)
	}
	key := head.Suite.ID + "/" + head.Suite.Version
	if s.cfg.Retired[key] {
		return nil, refuse(http.StatusGone, "suite %s is retired", key)
	}
	suite, ok := s.cfg.Suites[key]
	if !ok {
		return nil, refuse(http.StatusUpgradeRequired, "suite %s is not accepted", key)
	}
	res, err := benchfmt.Parse(body)
	if err != nil {
		return nil, refuse(http.StatusUnprocessableEntity, "%v", err)
	}
	if suite.PromptsSHA256 != "" && res.Suite.PromptsSHA256 != suite.PromptsSHA256 {
		return nil, refuse(http.StatusUnprocessableEntity, "suite.prompts_sha256 does not match %s: the client's prompts or parameters differ from the suite", key)
	}
	if res.InstallID == "" || res.Signature == "" {
		return nil, refuse(http.StatusUnprocessableEntity, "a submission is signed: install_id and signature are required")
	}
	if err := res.Verify(); err != nil {
		return nil, refuse(http.StatusUnprocessableEntity, "%v", err)
	}
	res.InstallID = strings.ToLower(res.InstallID)
	sum := sha256.Sum256(body)
	rawSHA := hex.EncodeToString(sum[:])

	s.mu.Lock()
	defer s.mu.Unlock()
	if id, err := store.SubmissionByRawSHA(ctx, s.st.DB, rawSHA); err == nil {
		_, m, err := store.GetSubmission(ctx, s.st.DB, id)
		if err != nil {
			return nil, err
		}
		return &Receipt{ID: id, URL: s.URLFor(id), State: m.State, Existing: true}, nil
	} else if !errors.Is(err, store.ErrNotFound) {
		return nil, err
	}
	if !s.cfg.PerKey.Allow(res.InstallID) {
		return nil, refuse(http.StatusTooManyRequests, "too many submissions from this installation today")
	}

	id := newID()
	rc := &Receipt{ID: id, URL: s.URLFor(id)}
	err = s.st.Tx(ctx, func(tx *sql.Tx) error {
		al, verified, err := s.inputs(ctx, tx)
		if err != nil {
			return err
		}
		rec, runs, err := s.index(ctx, tx, id, res, al)
		if err != nil {
			return err
		}
		mod := store.Moderation{State: store.StatePublished}
		for i, r := range runs {
			switch {
			case r.Plausibility == store.PlausibleNo:
				return refuse(http.StatusUnprocessableEntity, "runs[%d] (%s): token generation above the plausible ceiling of %.1f tok/s for this hardware and placement",
					i, r.Preset, r.Ceiling)
			case r.Status == benchfmt.StatusOK && r.Plausibility == store.PlausibleUnknown:
				rc.Flags = appendOnce(rc.Flags, "plausibility:unknown")
			}
			if r.Outlier && mod.State == store.StatePublished {
				mod = store.Moderation{State: store.StateQueued, Reason: fmt.Sprintf("outlier in its cohort (%s)", r.Preset)}
			}
			if r.OutputMatch >= 0 && r.OutputTokens > 0 && r.OutputMatch < r.OutputTokens {
				rc.Flags = appendOnce(rc.Flags, "output:differs")
			}
		}
		switch {
		case s.cfg.Moderation == ModerationManual:
			mod = store.Moderation{State: store.StateQueued, Reason: "manual moderation"}
		case rec.UserEntered && mod.State == store.StatePublished:
			mod = store.Moderation{State: store.StateQueued, Reason: "system model entered by hand"}
		}
		if mod.State == store.StateQueued {
			rc.Flags = appendOnce(rc.Flags, "moderation:"+mod.Reason)
		}
		rc.State = mod.State
		sub := store.Submission{ID: id, ReceivedOn: store.Today(), InstallID: res.InstallID, RawSHA256: rawSHA, Raw: body}
		if err := store.InsertSubmission(ctx, tx, sub, mod); err != nil {
			return err
		}
		if err := store.WriteNorm(ctx, tx, id, rec, verified[res.InstallID], runs); err != nil {
			return err
		}
		return aggregate.Rebuild(ctx, tx)
	})
	if err != nil {
		return nil, err
	}
	return rc, nil
}

func appendOnce(xs []string, x string) []string {
	for _, y := range xs {
		if y == x {
			return xs
		}
	}
	return append(xs, x)
}

// inputs reads the admin-edited inputs of the normalizer.
func (s *Service) inputs(ctx context.Context, q store.Querier) (normalize.Aliases, map[string]bool, error) {
	as, err := store.ListAliases(ctx, q)
	if err != nil {
		return nil, nil, err
	}
	al := normalize.Aliases{}
	for _, a := range as {
		al.Set(a.Kind, a.Pattern, a.Name)
	}
	vs, err := store.ListVerified(ctx, q)
	if err != nil {
		return nil, nil, err
	}
	verified := map[string]bool{}
	for _, v := range vs {
		verified[v.InstallID] = true
	}
	return al, verified, nil
}

// cohortKey groups runs whose greedy outputs should agree: one model file,
// backend, ggml CPU variant, repack and llama.cpp tag.
func cohortKey(rec normalize.Record, r normalize.Run) string {
	return strings.Join([]string{rec.ModelSHA256, r.Backend, rec.CPUVariant, rec.Repack, rec.LlamaTag}, "|")
}

// index normalizes a submission and judges its runs against what is
// already indexed.
func (s *Service) index(ctx context.Context, q store.Querier, id string, res *benchfmt.Result, al normalize.Aliases) (normalize.Record, []store.NormRun, error) {
	rec := normalize.Normalize(res, al)
	runs := make([]store.NormRun, len(rec.Runs))
	for i, r := range rec.Runs {
		nr := store.NormRun{Run: r, Cohort: cohortKey(rec, r), OutputMatch: -1}
		nr.Plausibility, nr.Ceiling = Plausibility(res, res.Runs[i], rec.CPU, s.cfg.CeilingSlack)
		if r.Status == benchfmt.StatusOK && r.OutputSHA256 != "" {
			others, err := store.CohortOutputs(ctx, q, nr.Cohort, id)
			if err != nil {
				return rec, nil, err
			}
			nr.OutputMatch = OutputMatch(r, others)
		}
		if r.Status == benchfmt.StatusOK && nr.Plausibility != store.PlausibleNo {
			c := store.HWCohort{Suite: rec.Suite, ModelSHA256: rec.ModelSHA256, Preset: r.Preset, Mode: r.Mode, Backend: r.Backend, CPUSlug: rec.CPUSlug, GPUSlug: rec.GPUSlug}
			for metric, v := range map[string]float64{"tg_d0": r.TGD0, "pp_tok_s": r.PPTokS} {
				if v <= 0 {
					continue
				}
				vs, err := store.CohortValues(ctx, q, c, metric, id)
				if err != nil {
					return rec, nil, err
				}
				if aggregate.Outlier(v, aggregate.Units(vs)) {
					nr.Outlier = true
				}
			}
		}
		runs[i] = nr
	}
	return rec, runs, nil
}

// OutputMatch is the longest output prefix, in tokens, a run shares with
// any of its cohort: the whole output when a full hash agrees, else the
// longest checkpoint whose prefix hashes agree, else 0; -1 with no one to
// compare with. A soft signal only: kernels that differ in the last bits
// drift apart late in a greedy decode.
func OutputMatch(r normalize.Run, others []store.CohortOutput) int {
	if len(others) == 0 {
		return -1
	}
	best := 0
	mine := map[int]string{}
	for _, p := range r.Prefixes {
		mine[p.Tokens] = p.SHA256
	}
	for _, o := range others {
		if o.SHA256 == r.OutputSHA256 && o.Tokens == r.OutputTokens {
			return r.OutputTokens
		}
		for _, p := range o.Prefixes {
			if mine[p.Tokens] == p.SHA256 && p.Tokens > best {
				best = p.Tokens
			}
		}
	}
	return best
}

// Withdraw deletes a submission whose install signs the withdrawal message.
func (s *Service) Withdraw(ctx context.Context, id, installID, signature string) error {
	if !ValidID(id) {
		return refuse(http.StatusNotFound, "no submission %q", id)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	sub, _, err := store.GetSubmission(ctx, s.st.DB, id)
	if errors.Is(err, store.ErrNotFound) {
		if w, _ := store.Withdrawn(ctx, s.st.DB, id); w {
			return nil // already withdrawn: the client's retry is not an error
		}
		return refuse(http.StatusNotFound, "no submission %s", id)
	}
	if err != nil {
		return err
	}
	pub, err1 := hex.DecodeString(installID)
	sig, err2 := hex.DecodeString(signature)
	if err1 != nil || err2 != nil || len(pub) != ed25519.PublicKeySize || !strings.EqualFold(installID, sub.InstallID) ||
		!ed25519.Verify(pub, benchfmt.WithdrawalMessage(id), sig) {
		return refuse(http.StatusForbidden, "the withdrawal is not signed by the installation that submitted %s", id)
	}
	return s.st.Tx(ctx, func(tx *sql.Tx) error {
		if err := s.st.Withdraw(ctx, tx, id); err != nil {
			return err
		}
		return aggregate.Rebuild(ctx, tx)
	})
}

// Moderate records an admin decision and updates the aggregates.
func (s *Service) Moderate(ctx context.Context, id, state, reason string) error {
	switch state {
	case store.StatePublished, store.StateHidden, store.StateQueued:
	default:
		return refuse(http.StatusBadRequest, "state must be published, queued or hidden")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.st.Tx(ctx, func(tx *sql.Tx) error {
		if err := store.SetModeration(ctx, tx, id, store.Moderation{State: state, Reason: reason, DecidedBy: "admin"}); err != nil {
			return err
		}
		return aggregate.Rebuild(ctx, tx)
	})
}

// Reindex rebuilds every derived table from the raw submissions with the
// current normalizer, aliases and verified installs. Moderation decisions
// are kept. It returns how many submissions were indexed.
func (s *Service) Reindex(ctx context.Context) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	err := s.st.Tx(ctx, func(tx *sql.Tx) error {
		if err := store.ClearDerived(ctx, tx); err != nil {
			return err
		}
		al, verified, err := s.inputs(ctx, tx)
		if err != nil {
			return err
		}
		err = store.EachSubmission(ctx, tx, func(sub store.Submission, _ store.Moderation) error {
			res, err := benchfmt.Parse(sub.Raw)
			if err != nil {
				// It parsed when it was accepted and the format only
				// grows: this is a reader bug, which must not silently
				// drop a result from the pages.
				return fmt.Errorf("submission %s no longer parses: %w", sub.ID, err)
			}
			res.InstallID = strings.ToLower(res.InstallID)
			rec, runs, err := s.index(ctx, tx, sub.ID, res, al)
			if err != nil {
				return err
			}
			n++
			return store.WriteNorm(ctx, tx, sub.ID, rec, verified[res.InstallID], runs)
		})
		if err != nil {
			return err
		}
		return aggregate.Rebuild(ctx, tx)
	})
	return n, err
}

// NeedsReindex reports whether the derived tables are missing rows or were
// made by an older normalizer.
func (s *Service) NeedsReindex(ctx context.Context) (bool, error) {
	oldest, complete, err := store.NormalizerVersions(ctx, s.st.DB)
	if err != nil {
		return false, err
	}
	return !complete || (oldest != 0 && oldest < normalize.Version), nil
}
