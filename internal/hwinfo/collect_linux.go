//go:build linux

package hwinfo

import (
	"context"
	"os"

	"github.com/usunrise88/nanollm/benchfmt"
)

func collect(ctx context.Context) benchfmt.Machine {
	m := linuxProfile(os.DirFS("/"))
	if m.OS.WSL {
		wslHost(ctx, &m)
	}
	return m
}
