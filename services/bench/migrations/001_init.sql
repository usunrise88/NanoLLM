-- 001: the first schema.
--
-- SQLite's ALTER TABLE cannot change a column or a constraint, so a later
-- migration that needs to rebuilds the table: CREATE TABLE x_new, INSERT
-- INTO x_new SELECT … FROM x, DROP TABLE x, ALTER TABLE x_new RENAME TO x,
-- and recreate its indexes and triggers. Derived tables (norm_*, aggregates)
-- may simply be dropped and recreated: `benchd reindex` refills them.

-- Raw submissions are the source of truth: the exact bytes received, never
-- updated. received_on is a day, not a time.
CREATE TABLE submissions (
    id          TEXT PRIMARY KEY,
    received_on TEXT NOT NULL,
    install_id  TEXT NOT NULL,
    raw_sha256  TEXT NOT NULL UNIQUE,
    raw         BLOB NOT NULL
);
CREATE INDEX submissions_install ON submissions (install_id);

CREATE TRIGGER submissions_immutable BEFORE UPDATE ON submissions
BEGIN
    SELECT RAISE(ABORT, 'submissions are immutable');
END;

-- A withdrawn submission is deleted; only its id and the day remain, so its
-- page answers 410 and a restored backup can be cleaned again.
CREATE TABLE withdrawals (
    id           TEXT PRIMARY KEY,
    withdrawn_on TEXT NOT NULL
);

-- Moderation is state about a submission, kept apart from its bytes.
CREATE TABLE moderation (
    submission_id TEXT PRIMARY KEY REFERENCES submissions (id) ON DELETE CASCADE,
    state         TEXT NOT NULL CHECK (state IN ('published', 'queued', 'hidden')),
    reason        TEXT NOT NULL DEFAULT '',
    decided_by    TEXT NOT NULL DEFAULT 'auto' CHECK (decided_by IN ('auto', 'admin')),
    updated_on    TEXT NOT NULL
);
CREATE INDEX moderation_state ON moderation (state);

-- Inputs of the normalizer that the admin edits.
CREATE TABLE aliases (
    kind    TEXT NOT NULL CHECK (kind IN ('cpu', 'gpu', 'vendor', 'system')),
    pattern TEXT NOT NULL,
    name    TEXT NOT NULL,
    PRIMARY KEY (kind, pattern)
);

-- Installs whose submissions carry the vendor-verified badge (the owner's
-- machines).
CREATE TABLE verified_installs (
    install_id TEXT PRIMARY KEY,
    label      TEXT NOT NULL DEFAULT ''
);

-- Derived: rebuilt by `benchd reindex` from submissions, aliases and
-- verified_installs. Never edited by hand.
CREATE TABLE norm_submissions (
    id            TEXT PRIMARY KEY REFERENCES submissions (id) ON DELETE CASCADE,
    normalizer    INTEGER NOT NULL,
    date          TEXT NOT NULL,
    suite         TEXT NOT NULL,
    model_id      TEXT NOT NULL,
    model_sha256  TEXT NOT NULL,
    nanollm       TEXT NOT NULL,
    llama_tag     TEXT NOT NULL,
    cpu_variant   TEXT NOT NULL,
    repack        TEXT NOT NULL,
    cpu           TEXT NOT NULL,
    cpu_slug      TEXT NOT NULL,
    gpu           TEXT NOT NULL,
    gpu_slug      TEXT NOT NULL,
    system        TEXT NOT NULL,
    system_slug   TEXT NOT NULL,
    system_vendor TEXT NOT NULL,
    form_factor   TEXT NOT NULL,
    os            TEXT NOT NULL,
    install_id    TEXT NOT NULL,
    fingerprint   TEXT NOT NULL,
    verified      INTEGER NOT NULL DEFAULT 0,
    user_entered  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX norm_submissions_cpu ON norm_submissions (cpu_slug);
CREATE INDEX norm_submissions_gpu ON norm_submissions (gpu_slug);
CREATE INDEX norm_submissions_system ON norm_submissions (system_slug);

CREATE TABLE norm_runs (
    submission_id  TEXT NOT NULL REFERENCES submissions (id) ON DELETE CASCADE,
    preset         TEXT NOT NULL,
    status         TEXT NOT NULL,
    reason         TEXT NOT NULL DEFAULT '',
    mode           TEXT NOT NULL DEFAULT '',
    backend        TEXT NOT NULL DEFAULT '',
    gpu_layers     INTEGER NOT NULL DEFAULT 0,
    mtp            INTEGER NOT NULL DEFAULT 0,
    load_ms        REAL NOT NULL DEFAULT 0,
    pp_tok_s       REAL NOT NULL DEFAULT 0,
    tg_d0          REAL NOT NULL DEFAULT 0,
    tg_dn          REAL NOT NULL DEFAULT 0,
    ttft_ms        REAL NOT NULL DEFAULT 0,
    chat_tok_s     REAL NOT NULL DEFAULT 0,
    mtp_speedup    REAL NOT NULL DEFAULT 0,
    mtp_acceptance REAL NOT NULL DEFAULT 0,
    -- ok | implausible | unknown: the tg ceiling from plan placement and
    -- memory bandwidth; ceiling_tok_s is 0 when unknown.
    plausibility   TEXT NOT NULL DEFAULT 'unknown',
    ceiling_tok_s  REAL NOT NULL DEFAULT 0,
    -- The output-hash cohort and how this run compares within it (a soft
    -- signal, never a reason to refuse).
    cohort         TEXT NOT NULL DEFAULT '',
    output_sha256  TEXT NOT NULL DEFAULT '',
    output_tokens  INTEGER NOT NULL DEFAULT 0,
    output_prefixes TEXT NOT NULL DEFAULT '',
    output_match   INTEGER NOT NULL DEFAULT -1,
    -- MAD outlier within the hardware cohort at the time of indexing.
    outlier        INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (submission_id, preset)
);
CREATE INDEX norm_runs_cohort ON norm_runs (cohort);

-- One row per group and metric: the median, p25 and p75 over units (each
-- install once, and each hardware fingerprint once).
CREATE TABLE aggregates (
    model_id    TEXT NOT NULL,
    suite       TEXT NOT NULL,
    preset      TEXT NOT NULL,
    cpu_slug    TEXT NOT NULL,
    gpu_slug    TEXT NOT NULL,
    system_slug TEXT NOT NULL,
    os          TEXT NOT NULL,
    form_factor TEXT NOT NULL,
    cpu         TEXT NOT NULL,
    gpu         TEXT NOT NULL,
    system      TEXT NOT NULL,
    verified    INTEGER NOT NULL DEFAULT 0,
    metric      TEXT NOT NULL,
    n           INTEGER NOT NULL,
    median      REAL NOT NULL,
    p25         REAL NOT NULL,
    p75         REAL NOT NULL,
    PRIMARY KEY (model_id, suite, preset, cpu_slug, gpu_slug, system_slug, os, form_factor, metric)
);
