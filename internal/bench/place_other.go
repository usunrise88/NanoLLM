//go:build !linux

package bench

import "errors"

// Resolve: pinning and memory policy are Linux-only; elsewhere only the zero
// placement is accepted.
func (p Placement) Resolve() (Placement, int, error) {
	if !p.IsZero() {
		return p, 0, errors.New("bench: CPU pinning and NUMA policy are Linux-only")
	}
	return p, 0, nil
}
