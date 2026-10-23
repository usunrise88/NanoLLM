package hwinfo

// Disk kinds, as the benchmark format spells them.
const (
	DiskNVMe    = "nvme"
	DiskSSD     = "ssd"
	DiskHDD     = "hdd"
	DiskUnknown = "unknown"
)

// DiskKind describes the drive that holds a file: unlike Machine.Disk, which
// is the first drive of the system, this is the one the model is read from.
type DiskKind struct {
	// Type is nvme, ssd, hdd or unknown.
	Type string `json:"type"`
	// Bus is the interface where the OS says (nvme, sata, usb, sas, raid, …).
	Bus string `json:"bus,omitempty"`
	// Removable is a USB stick or a card.
	Removable bool `json:"removable,omitempty"`
}

// busName maps a Windows STORAGE_BUS_TYPE to a name.
func busName(t uint32) string {
	switch t {
	case 0x01:
		return "scsi"
	case 0x02:
		return "atapi"
	case 0x03:
		return "ata"
	case 0x07:
		return "usb"
	case 0x08:
		return "raid"
	case 0x09:
		return "iscsi"
	case 0x0A:
		return "sas"
	case 0x0B:
		return "sata"
	case 0x0C:
		return "sd"
	case 0x0D:
		return "mmc"
	case 0x0E:
		return "virtual"
	case 0x0F:
		return "file-backed-virtual"
	case 0x10:
		return "spaces"
	case 0x11:
		return "nvme"
	}
	return ""
}

// kindOf combines what the drive says: the NVMe bus is NVMe; otherwise a seek
// penalty means a spinning disk and none an SSD; a drive that answers neither
// is unknown.
func kindOf(bus string, seekPenalty, haveSeek bool) string {
	switch {
	case bus == "nvme":
		return DiskNVMe
	case haveSeek && seekPenalty:
		return DiskHDD
	case haveSeek:
		return DiskSSD
	}
	return DiskUnknown
}
