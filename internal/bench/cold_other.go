//go:build !linux

package bench

// evictFile does nothing off Linux: other systems have no per-file page cache eviction we can ask for.
func evictFile(string) {}
