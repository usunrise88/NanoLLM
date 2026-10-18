package config

import (
	"crypto/rand"
	"crypto/sha256"
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

// HashSecret is the form a key is stored in when its plaintext should not be
// in the configuration file: "sha256:<hex>", which the server accepts in place
// of the secret.
func HashSecret(secret string) string {
	sum := sha256.Sum256([]byte(secret))
	return "sha256:" + hex.EncodeToString(sum[:])
}
