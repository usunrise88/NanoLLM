//go:build windows

package hwinfo

import (
	"encoding/binary"
	"path/filepath"

	"golang.org/x/sys/windows"
)

// DiskOf asks the volume that holds path what drive it is on, with
// IOCTL_STORAGE_QUERY_PROPERTY. Opening a volume with no access rights is
// allowed to any user, and the two queries (device descriptor, seek penalty)
// need nothing more.
func DiskOf(path string) DiskKind {
	k := DiskKind{Type: DiskUnknown}
	abs, err := filepath.Abs(path)
	if err != nil {
		return k
	}
	vol := filepath.VolumeName(abs)
	if len(vol) != 2 || vol[1] != ':' {
		return k // a UNC share or a path without a drive letter
	}
	p, err := windows.UTF16PtrFromString(`\\.\` + vol)
	if err != nil {
		return k
	}
	h, err := windows.CreateFile(p, 0, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE, nil, windows.OPEN_EXISTING, 0, 0)
	if err != nil {
		return k
	}
	defer windows.CloseHandle(h)

	const ioctlStorageQueryProperty = 0x2D1400
	query := func(propertyID uint32, out []byte) bool {
		// STORAGE_PROPERTY_QUERY: PropertyId, QueryType (0: standard), one
		// byte of additional parameters.
		in := make([]byte, 12)
		binary.LittleEndian.PutUint32(in[0:], propertyID)
		var n uint32
		return windows.DeviceIoControl(h, ioctlStorageQueryProperty, &in[0], uint32(len(in)), &out[0], uint32(len(out)), &n, nil) == nil
	}
	var bus string
	desc := make([]byte, 1024)
	if query(0, desc) { // StorageDeviceProperty
		// STORAGE_DEVICE_DESCRIPTOR: RemovableMedia at 10, BusType at 28.
		k.Removable = desc[10] != 0
		bus = busName(binary.LittleEndian.Uint32(desc[28:]))
		k.Bus = bus
	}
	seek := make([]byte, 16)
	haveSeek := query(7, seek) // StorageDeviceSeekPenaltyProperty
	// DEVICE_SEEK_PENALTY_DESCRIPTOR: Version, Size, IncursSeekPenalty.
	penalty := haveSeek && seek[8] != 0
	k.Type = kindOf(bus, penalty, haveSeek)
	return k
}
