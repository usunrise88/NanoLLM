// Package migrations holds the numbered SQL migrations, embedded in the
// binary. A file is NNN_name.sql; the store applies those above the version
// recorded in schema_version, each in its own transaction.
package migrations

import "embed"

//go:embed *.sql
var FS embed.FS
