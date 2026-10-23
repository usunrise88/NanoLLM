package main

import "testing"

func TestFileName(t *testing.T) {
	for want, g := range map[string]Golden{
		"cpu-haswell.json":          {CPUVariant: "haswell"},
		"cpu-mtp-haswell-q8_0.json": {CPUVariant: "haswell", KVType: "q8_0", DrafterFile: "mtp.gguf"},
		"cpu-apple-m1.json":         {CPUName: "Apple M1"},
		"cpu-generic.json":          {},
		"mtl-apple-m1.json":         {Device: "mtl0", DeviceName: "Apple M1", CPUName: "Apple M1"},
		"mtl-mtp-apple-m1.json":     {Device: "mtl0", DeviceName: "Apple M1", DrafterFile: "mtp.gguf"},
		"cuda-nvidia-geforce-rtx-3070-ti-laptop-gpu-q8_0.json": {Device: "cuda0", DeviceName: "NVIDIA GeForce RTX 3070 Ti Laptop GPU", KVType: "q8_0"},
	} {
		if got := FileName(g); got != want {
			t.Errorf("FileName(%+v) = %q, want %q", g, got, want)
		}
	}
}

func TestListedDevice(t *testing.T) {
	m := listedDevice.FindStringSubmatch("  MTL0: Apple M1 (5461 MiB, 5460 MiB free)")
	if m == nil || m[1] != "MTL0" || m[2] != "Apple M1" {
		t.Errorf("parsed %q", m)
	}
	if listedDevice.MatchString("Available devices:") {
		t.Error("the heading matched")
	}
}
