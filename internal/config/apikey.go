package config

import (
	"crypto/rand"
	"encoding/hex"
)

// KeyPrefix marks a NanoLLM secret, so one pasted into the wrong place is
// recognisable and secret scanners can be taught the pattern.
const KeyPrefix = "sk-nanollm-"

// NewKeySecret returns a fresh API key: the prefix and 128 random bits.
func NewKeySecret() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return KeyPrefix + hex.EncodeToString(b), nil
}
