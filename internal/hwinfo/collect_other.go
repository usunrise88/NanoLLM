//go:build !linux && !windows && !darwin

package hwinfo

import (
	"context"

	"github.com/usunrise88/nanollm/benchfmt"
)

func collect(context.Context) benchfmt.Machine { return benchfmt.Machine{} }

func cpuNameRaw() string { return "" }
