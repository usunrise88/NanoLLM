//go:build linux || darwin

package memlock

import (
	"fmt"

	"golang.org/x/sys/unix"
)

type pinSys struct {
	maps [][]byte
}

// raiseLimit lifts RLIMIT_MEMLOCK towards need: the soft limit up to the
// hard one for anybody, the hard one too for a process allowed to (root,
// CAP_SYS_RESOURCE). It never refuses on its own: a process with
// CAP_IPC_LOCK locks past any limit, so only the lock itself can say no. It
// returns the limit in force, for the error message.
func raiseLimit(need int64) int64 {
	var lim unix.Rlimit
	if err := unix.Getrlimit(unix.RLIMIT_MEMLOCK, &lim); err != nil {
		return -1
	}
	if lim.Cur != unix.RLIM_INFINITY && int64(lim.Cur) < need {
		want := uint64(need) + 64<<20
		if lim.Max == unix.RLIM_INFINITY || lim.Max >= want {
			_ = unix.Setrlimit(unix.RLIMIT_MEMLOCK, &unix.Rlimit{Cur: want, Max: lim.Max})
		} else if unix.Setrlimit(unix.RLIMIT_MEMLOCK, &unix.Rlimit{Cur: want, Max: want}) != nil {
			_ = unix.Setrlimit(unix.RLIMIT_MEMLOCK, &unix.Rlimit{Cur: lim.Max, Max: lim.Max})
		}
		_ = unix.Getrlimit(unix.RLIMIT_MEMLOCK, &lim)
	}
	if lim.Cur == unix.RLIM_INFINITY {
		return -1
	}
	return int64(lim.Cur)
}

func (p *Pin) lock() error {
	need := Total(p.ranges)
	if need == 0 {
		return nil
	}
	limit := raiseLimit(need)
	for _, r := range p.ranges {
		b, err := unix.Mmap(int(p.f.Fd()), r.Off, int(r.Len), unix.PROT_READ, unix.MAP_SHARED)
		if err != nil {
			return fmt.Errorf("memlock: map %d bytes at %d: %w", r.Len, r.Off, err)
		}
		p.sys.maps = append(p.sys.maps, b)
		_ = unix.Madvise(b, unix.MADV_WILLNEED)
		if err := unix.Mlock(b); err != nil {
			if err == unix.ENOMEM || err == unix.EPERM || err == unix.EAGAIN {
				return limitError(need, limit,
					"raise it with LimitMEMLOCK=infinity in the systemd unit, `ulimit -l`, or memlock in /etc/security/limits.conf")
			}
			return fmt.Errorf("memlock: lock %d bytes at %d: %w", r.Len, r.Off, err)
		}
		p.locked += r.Len
	}
	return nil
}

func (p *Pin) unlock() error {
	var first error
	for _, b := range p.sys.maps {
		_ = unix.Munlock(b)
		if err := unix.Munmap(b); err != nil && first == nil {
			first = err
		}
	}
	p.sys.maps = nil
	p.locked = 0
	return first
}

// Resident counts the bytes of b that are in RAM.
func Resident(b []byte) (int64, error) {
	if len(b) == 0 {
		return 0, nil
	}
	vec := make([]byte, (int64(len(b))+PageSize-1)/PageSize)
	if err := mincore(b, vec); err != nil {
		return 0, err
	}
	var n int64
	for _, v := range vec {
		if v&1 != 0 {
			n += PageSize
		}
	}
	return min(n, int64(len(b))), nil
}
