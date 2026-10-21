//go:build !windows

package main

import "errors"

func setAffinity(string) error {
	return errors.New("-mask is Windows only; use taskset or -place (NANOLLM_PLACE) elsewhere")
}
