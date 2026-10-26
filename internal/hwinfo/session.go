package hwinfo

import "os"

// InServiceSession reports whether this process runs in Windows session 0,
// where services live: it has no desktop, and the Intel and AMD graphics
// drivers do not expose their Vulkan device there. False elsewhere.
func InServiceSession() bool { return inServiceSession(os.Getpid()) }
