package benchfmt

import "strings"

// placeholders are what firmware vendors leave in SMBIOS fields they did not
// fill in. White-box and server boards mostly do not.
var placeholders = map[string]bool{
	"": true, "to be filled by o.e.m.": true, "to be filled by o.e.m": true, "default string": true,
	"system product name": true, "system version": true, "system manufacturer": true,
	"base board product name": true, "base board manufacturer": true, "not applicable": true,
	"none": true, "o.e.m.": true, "oem": true, "not specified": true, "type1productconfigid": true,
	"invalid": true, "unknown": true, "x.x": true, "0123456789": true, "1234567890": true,
}

// Placeholder reports whether an SMBIOS string carries no information. The
// client and the service share the list, so a machine is named the same way
// where it is measured and where it is shown.
func Placeholder(s string) bool { return placeholders[strings.ToLower(strings.TrimSpace(s))] }
