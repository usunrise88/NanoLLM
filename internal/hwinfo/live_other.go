//go:build !windows && !linux && !darwin

package hwinfo

import (
	"context"
	"time"
)

func memoryNow() (total, available int64) { return 0, 0 }

func cpuBusy(context.Context, time.Duration) float64 { return -1 }
