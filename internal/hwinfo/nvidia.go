package hwinfo

import (
	"context"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/usunrise88/nanollm/benchfmt"
)

// enrichGPUs adds what nvidia-smi knows to NVIDIA GPUs: the driver's major
// version and the power limit, which on laptops varies by model far more than
// the GPU name suggests (a "3070 Ti Laptop" runs anywhere from 80 to 150 W).
func enrichGPUs(ctx context.Context, m *benchfmt.Machine) {
	var nv []int
	for i, g := range m.GPUs {
		if strings.Contains(strings.ToLower(g.Name+" "+g.Vendor), "nvidia") || strings.Contains(g.Name, "GeForce") || strings.Contains(g.Name, "RTX") {
			nv = append(nv, i)
		}
	}
	if len(nv) == 0 {
		return
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=name,driver_version,power.limit", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return
	}
	rows := parseNvidiaSMI(string(out))
	for k, i := range nv {
		if k >= len(rows) {
			break
		}
		g := &m.GPUs[i]
		g.Vendor = "NVIDIA"
		if g.DriverMajor == "" {
			g.DriverMajor = majorOf(rows[k].driver)
		}
		if rows[k].powerW > 0 {
			g.PowerLimitW = rows[k].powerW
		}
	}
}

type nvRow struct {
	name, driver string
	powerW       int
}

func parseNvidiaSMI(s string) []nvRow {
	var out []nvRow
	for _, line := range strings.Split(strings.TrimSpace(s), "\n") {
		f := strings.Split(line, ",")
		if len(f) < 3 {
			continue
		}
		r := nvRow{name: strings.TrimSpace(f[0]), driver: strings.TrimSpace(f[1])}
		if w, err := strconv.ParseFloat(strings.TrimSpace(f[2]), 64); err == nil {
			r.powerW = int(w + 0.5)
		}
		out = append(out, r)
	}
	return out
}
