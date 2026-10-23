package hwinfo

import (
	"encoding/binary"
	"strings"
)

// smbiosInfo is what is read from the SMBIOS tables: system (type 1),
// enclosure (type 3) and memory devices (type 17). The serial number, UUID
// and asset tags in those same structures are skipped by offset and never
// decoded.
type smbiosInfo struct {
	vendor, product, version, family string
	chassis                          int
	memMTS                           int
	// memModules counts the memory devices that are populated.
	memModules int
}

// parseSMBIOS reads the raw table GetSystemFirmwareTable('RSMB') returns: an
// 8-byte header (calling method, major, minor, DMI revision, u32 length) and
// the structure table.
func parseSMBIOS(raw []byte) smbiosInfo {
	var out smbiosInfo
	if len(raw) < 8 {
		return out
	}
	n := int(binary.LittleEndian.Uint32(raw[4:8]))
	table := raw[8:]
	if n < len(table) {
		table = table[:n]
	}
	for len(table) >= 4 {
		typ, length := table[0], int(table[1])
		if length < 4 || length > len(table) {
			break
		}
		formatted := table[:length]
		// Strings follow the formatted area, each NUL-terminated, the set
		// ending with an extra NUL.
		rest := table[length:]
		end := 0
		for end+1 < len(rest) && (rest[end] != 0 || rest[end+1] != 0) {
			end++
		}
		var strs []string
		if end > 0 {
			strs = strings.Split(string(rest[:end]), "\x00")
		}
		str := func(off int) string {
			if off >= len(formatted) {
				return ""
			}
			i := int(formatted[off])
			if i == 0 || i > len(strs) {
				return ""
			}
			return strings.TrimSpace(strs[i-1])
		}
		switch typ {
		case 1:
			out.vendor, out.product, out.version = str(4), str(5), str(6)
			out.family = str(0x1A)
		case 3:
			if len(formatted) > 5 && out.chassis == 0 {
				out.chassis = int(formatted[5] & 0x7f)
			}
		case 17:
			// Size (0x0C): 0 is an empty slot, 0xFFFF unknown.
			if len(formatted) >= 0x0E && binary.LittleEndian.Uint16(formatted[0x0C:0x0E]) != 0 {
				out.memModules++
			}
			speed := 0
			if len(formatted) >= 0x22 {
				speed = int(binary.LittleEndian.Uint16(formatted[0x20:0x22])) // configured speed
			}
			if speed == 0 && len(formatted) >= 0x17 {
				speed = int(binary.LittleEndian.Uint16(formatted[0x15:0x17]))
			}
			if speed > out.memMTS && speed < 0xffff {
				out.memMTS = speed
			}
		case 127:
			return out
		}
		next := length + end + 2
		if next > len(table) {
			break
		}
		table = table[next:]
	}
	return out
}
