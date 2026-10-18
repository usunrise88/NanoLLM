package plan

import (
	"fmt"
	"math"

	"github.com/usunrise88/nanollm/internal/backend"
)

// DeviceMemory is what a device has free, or its total when it does not say.
func DeviceMemory(d backend.Device) int64 {
	if d.MemoryFree > 0 {
		return d.MemoryFree
	}
	return d.MemoryTotal
}

// DerivedPreset is the budget a device offers when the configuration names
// none: what it has free, rounded down to half a GiB, as a vram-, uma- or
// ram- budget. The host keeps a quarter of its free memory for everything
// else that runs on it.
func DerivedPreset(d backend.Device) (Preset, error) {
	free := DeviceMemory(d)
	if free <= 0 {
		return Preset{}, fmt.Errorf("device %s reports no memory; set models.overrides.<id>.preset", d.Selector())
	}
	kind := "vram"
	switch {
	case d.Unified:
		kind = "uma"
	case d.Type == backend.CPU:
		kind, free = "ram", free*3/4
	}
	gib := math.Floor(float64(free)/float64(GiB)*2) / 2
	if gib < 0.5 {
		return Preset{}, fmt.Errorf("device %s has %d MiB free, too little for any preset", d.Selector(), free>>20)
	}
	return ParsePreset(fmt.Sprintf("%s-%gg", kind, gib))
}
