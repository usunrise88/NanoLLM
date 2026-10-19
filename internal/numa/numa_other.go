//go:build !linux

package numa

// Default is true where no policy can be read.
func Default() bool { return true }

// Enter does nothing off Linux: memory placement is not ours to set there.
func Enter(Spec, []int) error { return nil }
