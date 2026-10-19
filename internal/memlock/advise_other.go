//go:build !linux

package memlock

// AdviseRandom is a no-op off Linux (macOS readahead on a mapped file is not
// tunable the same way, Windows has its own prefetcher).
func AdviseRandom(string, []Range) (int64, error) { return 0, nil }
