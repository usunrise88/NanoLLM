//go:build !linux && !darwin && !windows

package memlock

type pinSys struct{}

func (p *Pin) lock() error {
	if Total(p.ranges) > 0 {
		return errUnsupported
	}
	return nil
}

func (p *Pin) unlock() error { return nil }

func Resident([]byte) (int64, error) { return 0, errUnsupported }

func evict([]byte, uintptr, int64) error { return errUnsupported }
