package hwinfo

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
