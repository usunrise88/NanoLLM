package benchfmt

import (
	"bytes"
	"crypto/ed25519"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
)

// The signature proves ownership, not honesty: whoever holds an install's key
// can withdraw its submissions and have its repeats counted as one, and that
// is all. Anyone can make a key and sign a fabricated result; the service's
// plausibility checks, not this, deal with that (docs/BENCHMARKS.md).

// Canonical is the form that is signed: the JSON document without its
// "signature" member, keys sorted bytewise at every level, no white space,
// numbers exactly as written, strings escaped as `"`, `\\` and control
// characters only (no HTML escaping). It is computed from the document, not
// from a Go struct: a member that a verifier's struct does not know still
// counts, so a service that is older than its client verifies a result with
// new fields instead of rejecting its signature.
func Canonical(doc []byte) ([]byte, error) {
	dec := json.NewDecoder(bytes.NewReader(doc))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, fmt.Errorf("benchfmt: %w", err)
	}
	if dec.More() {
		return nil, errors.New("benchfmt: data after the JSON document")
	}
	m, ok := v.(map[string]any)
	if !ok {
		return nil, errors.New("benchfmt: a result is a JSON object")
	}
	delete(m, "signature")
	var out bytes.Buffer
	enc := json.NewEncoder(&out)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(m); err != nil { // sorted map keys, json.Number as written
		return nil, err
	}
	return bytes.TrimRight(out.Bytes(), "\n"), nil
}

// signingBytes is Canonical of the struct's own JSON.
func (r Result) signingBytes() ([]byte, error) {
	r.Signature = ""
	doc, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	return Canonical(doc)
}

// Sign sets InstallID to the key's public half and Signature over the rest.
func (r *Result) Sign(key ed25519.PrivateKey) error {
	r.InstallID = hex.EncodeToString(key.Public().(ed25519.PublicKey))
	b, err := r.signingBytes()
	if err != nil {
		return err
	}
	r.Signature = hex.EncodeToString(ed25519.Sign(key, b))
	return nil
}

// Verify checks Signature against InstallID over the struct's own JSON. A
// service verifies the document it received with VerifyJSON instead.
func (r *Result) Verify() error {
	b, err := r.signingBytes()
	if err != nil {
		return err
	}
	return verify(r.InstallID, r.Signature, b)
}

// VerifyJSON checks the signature of a received document over its canonical
// bytes (see Canonical), whatever members it has.
func VerifyJSON(doc []byte) error {
	var head struct {
		InstallID string `json:"install_id"`
		Signature string `json:"signature"`
	}
	if err := json.Unmarshal(doc, &head); err != nil {
		return fmt.Errorf("benchfmt: %w", err)
	}
	b, err := Canonical(doc)
	if err != nil {
		return err
	}
	return verify(head.InstallID, head.Signature, b)
}

func verify(installID, signature string, canonical []byte) error {
	pub, err := hex.DecodeString(installID)
	if err != nil || len(pub) != ed25519.PublicKeySize {
		return errors.New("benchfmt: install_id is not an ed25519 public key")
	}
	sig, err := hex.DecodeString(signature)
	if err != nil || len(sig) != ed25519.SignatureSize {
		return errors.New("benchfmt: signature is malformed")
	}
	if !ed25519.Verify(pub, canonical, sig) {
		return fmt.Errorf("benchfmt: signature does not match install_id %s…", installID[:12])
	}
	return nil
}

// WithdrawalMessage is what an install signs to withdraw a submission.
func WithdrawalMessage(id string) []byte { return []byte("nanollm.bench/v1 withdraw " + id) }
