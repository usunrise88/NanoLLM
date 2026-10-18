package config

import (
	"fmt"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// Auto is a count the operator may leave to the planner: "auto" (or nothing)
// in YAML, a positive integer otherwise. The zero value is auto.
type Auto struct {
	N int
}

// IsAuto reports whether the planner decides.
func (a Auto) IsAuto() bool { return a.N == 0 }

func (a Auto) String() string {
	if a.IsAuto() {
		return "auto"
	}
	return strconv.Itoa(a.N)
}

// UnmarshalText is shared by the YAML decoder and the NANOLLM_* environment
// overrides, so both accept exactly the same spellings.
func (a *Auto) UnmarshalText(b []byte) error {
	s := strings.TrimSpace(string(b))
	if s == "" || s == "auto" {
		a.N = 0
		return nil
	}
	n, err := strconv.Atoi(s)
	if err != nil || n <= 0 {
		return fmt.Errorf("%q is neither auto nor a positive integer", s)
	}
	a.N = n
	return nil
}

func (a *Auto) UnmarshalYAML(n *yaml.Node) error {
	if n.Kind != yaml.ScalarNode {
		return fmt.Errorf("line %d: expected auto or a positive integer", n.Line)
	}
	if err := a.UnmarshalText([]byte(n.Value)); err != nil {
		return fmt.Errorf("line %d: %w", n.Line, err)
	}
	return nil
}

func (a Auto) MarshalYAML() (any, error) {
	if a.IsAuto() {
		return "auto", nil
	}
	return a.N, nil
}
