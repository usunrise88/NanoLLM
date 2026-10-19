//go:build !linux

package bench

// cgroupWatch has nothing to watch off Linux.
type cgroupWatch struct{}

func watchCgroup() *cgroupWatch { return &cgroupWatch{} }

func (*cgroupWatch) finish() (peak, pageCache int64) { return 0, 0 }
