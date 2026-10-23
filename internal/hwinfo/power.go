package hwinfo

import "strings"

// sameSaverName says a power plan is the power saver: a PC maker's copy of the
// plan has a GUID of its own (the AERO 5 XE's "Power saver" is not
// a1841308-...) and keeps the name Windows gives the standard plan, localized.
func sameSaverName(plan, standard string) bool {
	return plan != "" && strings.EqualFold(strings.TrimSpace(plan), strings.TrimSpace(standard))
}

// PowerMode is how the OS is asked to trade speed for energy.
type PowerMode struct {
	// Saver says the OS is in a mode that lowers CPU and GPU clocks: the
	// Windows power saver plan or the best-power-efficiency overlay, the
	// Linux powersave platform profile or governor.
	Saver bool `json:"saver"`
	// Name is what the OS calls the mode (power plan GUID, platform profile),
	// for the report.
	Name string `json:"name,omitempty"`
}
