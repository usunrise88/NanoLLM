// Command benchd is the NanoLLM public benchmark service
// (docs/BENCHMARKS.md, "Сервис services/bench").
//
//	benchd [serve]     the HTTP service
//	benchd reindex     rebuild the derived tables from the raw submissions
//	benchd backup      snapshot the database and prune old snapshots
//	benchd migrate     apply migrations and exit
//	benchd healthcheck GET /healthz (for container health checks)
//
// Every flag also reads an environment variable (BENCHD_*); see -h.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/usunrise88/nanollm/services/bench/internal/admin"
	"github.com/usunrise88/nanollm/services/bench/internal/ingest"
	"github.com/usunrise88/nanollm/services/bench/internal/store"
	"github.com/usunrise88/nanollm/services/bench/internal/web"
)

var version = "dev"

func main() {
	cmd, args := "serve", os.Args[1:]
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		cmd, args = args[0], args[1:]
	}
	var err error
	switch cmd {
	case "serve":
		err = serve(args)
	case "reindex":
		err = reindex(args)
	case "backup":
		err = backup(args)
	case "migrate":
		err = migrate(args)
	case "healthcheck":
		err = healthcheck(args)
	case "version":
		fmt.Println("benchd", version)
	default:
		err = fmt.Errorf("unknown command %q: serve, reindex, backup, migrate, healthcheck, version", cmd)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "benchd:", err)
		os.Exit(1)
	}
}

// config is the whole configuration; flags default to BENCHD_* variables.
type config struct {
	addr, db, publicURL, adminToken string
	license, licenseURL             string
	trustProxy, acceptFake          bool
	ratePerIP, ratePerKey           int
	retired, moderation             string
	slack                           float64
	backupDir                       string
	backupKeep, backupEvery         time.Duration
	tlsCert, tlsKey                 string
}

func env(name, def string) string {
	if v, ok := os.LookupEnv(name); ok {
		return v
	}
	return def
}

func envInt(name string, def int) int {
	if v, err := strconv.Atoi(env(name, "")); err == nil {
		return v
	}
	return def
}

func envDur(name string, def time.Duration) time.Duration {
	if v, err := time.ParseDuration(env(name, "")); err == nil {
		return v
	}
	return def
}

func envBool(name string) bool {
	v, _ := strconv.ParseBool(env(name, "false"))
	return v
}

func envFloat(name string, def float64) float64 {
	if v, err := strconv.ParseFloat(env(name, ""), 64); err == nil {
		return v
	}
	return def
}

func flags(name string, args []string) (*config, error) {
	c := &config{}
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.StringVar(&c.addr, "addr", env("BENCHD_ADDR", ":8080"), "listen address (BENCHD_ADDR)")
	fs.StringVar(&c.db, "db", env("BENCHD_DB", "bench.db"), "SQLite database path (BENCHD_DB)")
	fs.StringVar(&c.publicURL, "public-url", env("BENCHD_PUBLIC_URL", "https://nano.llmto.day"), "public base URL for receipts and pages (BENCHD_PUBLIC_URL)")
	fs.StringVar(&c.license, "data-license", env("BENCHD_DATA_LICENSE", "CC-BY-4.0"), "SPDX id of the open data licence: the owner's decision (BENCHD_DATA_LICENSE)")
	fs.StringVar(&c.licenseURL, "data-license-url", env("BENCHD_DATA_LICENSE_URL", ""), "licence text URL; known SPDX ids fill it in (BENCHD_DATA_LICENSE_URL)")
	fs.BoolVar(&c.trustProxy, "trust-proxy", envBool("BENCHD_TRUST_PROXY"), "take the client address from X-Forwarded-For (BENCHD_TRUST_PROXY)")
	fs.IntVar(&c.ratePerIP, "rate-ip", envInt("BENCHD_RATE_IP", 20), "submissions per address per hour, 0 = no limit (BENCHD_RATE_IP)")
	fs.IntVar(&c.ratePerKey, "rate-key", envInt("BENCHD_RATE_KEY", 20), "submissions per install key per day, 0 = no limit (BENCHD_RATE_KEY)")
	fs.StringVar(&c.retired, "retired-suites", env("BENCHD_RETIRED_SUITES", ""), "comma-separated retired suites answered with 410, e.g. standard/v0 (BENCHD_RETIRED_SUITES)")
	fs.StringVar(&c.moderation, "moderation", env("BENCHD_MODERATION", ingest.ModerationAuto), "auto (publish unless a check queues it) or manual (BENCHD_MODERATION)")
	fs.BoolVar(&c.acceptFake, "accept-fake", envBool("BENCHD_ACCEPT_FAKE"), "test mode: accept results from NanoLLM's fake backend, without the plausibility ceiling (BENCHD_ACCEPT_FAKE)")
	fs.Float64Var(&c.slack, "ceiling-slack", envFloat("BENCHD_CEILING_SLACK", 1.3), "multiplier on the tg plausibility ceiling (BENCHD_CEILING_SLACK)")
	fs.StringVar(&c.backupDir, "backup-dir", env("BENCHD_BACKUP_DIR", ""), "backup directory (BENCHD_BACKUP_DIR)")
	fs.DurationVar(&c.backupKeep, "backup-keep", envDur("BENCHD_BACKUP_KEEP", 30*24*time.Hour), "backup retention (BENCHD_BACKUP_KEEP)")
	fs.DurationVar(&c.backupEvery, "backup-every", envDur("BENCHD_BACKUP_EVERY", 0), "serve: take a backup this often, 0 = never (BENCHD_BACKUP_EVERY)")
	fs.StringVar(&c.tlsCert, "tls-cert", env("BENCHD_TLS_CERT", ""), "serve HTTPS with this certificate (normally Caddy terminates TLS) (BENCHD_TLS_CERT)")
	fs.StringVar(&c.tlsKey, "tls-key", env("BENCHD_TLS_KEY", ""), "the certificate's key (BENCHD_TLS_KEY)")
	if err := fs.Parse(args); err != nil {
		return nil, err
	}
	// The admin token is read from the environment only, so it never shows
	// in a process list.
	c.adminToken = os.Getenv("BENCHD_ADMIN_TOKEN")
	switch c.moderation {
	case ingest.ModerationAuto, ingest.ModerationManual:
	default:
		return nil, fmt.Errorf("-moderation must be auto or manual")
	}
	if c.backupKeep <= 0 {
		return nil, fmt.Errorf("-backup-keep must be positive: retention is what makes a withdrawal reach the backups")
	}
	return c, nil
}

func (c *config) retiredSet() map[string]bool {
	m := map[string]bool{}
	for _, s := range strings.Split(c.retired, ",") {
		if s = strings.TrimSpace(s); s != "" {
			m[s] = true
		}
	}
	return m
}

func logger() *slog.Logger { return slog.New(slog.NewJSONHandler(os.Stderr, nil)) }

func open(ctx context.Context, c *config) (*store.Store, *ingest.Service, *ingest.Limiter, error) {
	st, err := store.Open(ctx, c.db)
	if err != nil {
		return nil, nil, nil, err
	}
	perIP := ingest.NewLimiter(c.ratePerIP, time.Hour)
	ing := ingest.New(st, ingest.Config{
		PublicURL: c.publicURL, Retired: c.retiredSet(), CeilingSlack: c.slack, Moderation: c.moderation, AcceptFake: c.acceptFake,
		PerIP: perIP, PerKey: ingest.NewLimiter(c.ratePerKey, 24*time.Hour),
	})
	return st, ing, perIP, nil
}

func serve(args []string) error {
	c, err := flags("serve", args)
	if err != nil {
		return err
	}
	log := logger()
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	st, ing, perIP, err := open(ctx, c)
	if err != nil {
		return err
	}
	defer func() { _ = st.Close() }()
	if need, err := ing.NeedsReindex(ctx); err != nil {
		return err
	} else if need {
		n, err := ing.Reindex(ctx)
		if err != nil {
			return fmt.Errorf("reindex: %w", err)
		}
		log.Info("reindexed", "submissions", n)
	}
	retentionDays := int(c.backupKeep / (24 * time.Hour))
	ws, err := web.New(web.Config{
		PublicURL: c.publicURL, DataLicense: c.license, DataLicenseURL: c.licenseURL, TrustProxy: c.trustProxy,
		BackupRetentionDays: retentionDays,
	}, st, ing, perIP, log)
	if err != nil {
		return err
	}
	mux := http.NewServeMux()
	ws.Routes(mux)
	admin.New(c.adminToken, st, ing, log).Routes(mux)
	srv := &http.Server{
		Addr: c.addr, Handler: web.Headers(mux),
		ReadHeaderTimeout: 10 * time.Second, ReadTimeout: 30 * time.Second, WriteTimeout: 60 * time.Second, IdleTimeout: 120 * time.Second,
		MaxHeaderBytes: 32 << 10,
		// No access log: the service never records client addresses.
		ErrorLog: slog.NewLogLogger(log.Handler(), slog.LevelWarn),
	}
	if c.backupEvery > 0 {
		if c.backupDir == "" {
			return errors.New("-backup-every needs -backup-dir")
		}
		go backupLoop(ctx, st, c, log)
	}
	errc := make(chan error, 1)
	go func() {
		log.Info("serving", "addr", c.addr, "version", version, "admin", c.adminToken != "", "license", c.license)
		if c.tlsCert != "" {
			errc <- srv.ListenAndServeTLS(c.tlsCert, c.tlsKey)
		} else {
			errc <- srv.ListenAndServe()
		}
	}()
	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
	}
	shut, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return srv.Shutdown(shut)
}

func backupLoop(ctx context.Context, st *store.Store, c *config, log *slog.Logger) {
	t := time.NewTicker(c.backupEvery)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			path, removed, err := st.Backup(ctx, c.backupDir, c.backupKeep, time.Now())
			if err != nil {
				log.Error("backup failed", "err", err)
				continue
			}
			log.Info("backup", "path", path, "pruned", len(removed))
		}
	}
}

func reindex(args []string) error {
	c, err := flags("reindex", args)
	if err != nil {
		return err
	}
	ctx := context.Background()
	st, ing, _, err := open(ctx, c)
	if err != nil {
		return err
	}
	defer func() { _ = st.Close() }()
	n, err := ing.Reindex(ctx)
	if err != nil {
		return err
	}
	fmt.Printf("reindexed %d submissions\n", n)
	return nil
}

func backup(args []string) error {
	c, err := flags("backup", args)
	if err != nil {
		return err
	}
	if c.backupDir == "" {
		return errors.New("-backup-dir (BENCHD_BACKUP_DIR) is required")
	}
	ctx := context.Background()
	st, err := store.Open(ctx, c.db)
	if err != nil {
		return err
	}
	defer func() { _ = st.Close() }()
	path, removed, err := st.Backup(ctx, c.backupDir, c.backupKeep, time.Now())
	if err != nil {
		return err
	}
	fmt.Println("backup:", path)
	for _, r := range removed {
		fmt.Println("pruned:", r)
	}
	return nil
}

func migrate(args []string) error {
	c, err := flags("migrate", args)
	if err != nil {
		return err
	}
	ctx := context.Background()
	st, err := store.Open(ctx, c.db)
	if err != nil {
		return err
	}
	defer func() { _ = st.Close() }()
	v, err := st.SchemaVersion(ctx)
	if err != nil {
		return err
	}
	fmt.Println("schema version", v)
	return nil
}

func healthcheck(args []string) error {
	c, err := flags("healthcheck", args)
	if err != nil {
		return err
	}
	addr := c.addr
	if strings.HasPrefix(addr, ":") {
		addr = "127.0.0.1" + addr
	}
	scheme := "http"
	if c.tlsCert != "" {
		scheme = "https"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, scheme+"://"+addr+"/healthz", nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("healthz: %s", resp.Status)
	}
	return nil
}
