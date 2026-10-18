package benchfmt

import (
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

// signingBytes is the canonical form that is signed: the document with an
// empty signature, as encoding/json writes it (struct order, no spaces).
func (r Result) signingBytes() ([]byte, error) {
	r.Signature = ""
	return json.Marshal(r)
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

// Verify checks Signature against InstallID.
func (r *Result) Verify() error {
	pub, err := hex.DecodeString(r.InstallID)
	if err != nil || len(pub) != ed25519.PublicKeySize {
		return errors.New("benchfmt: install_id is not an ed25519 public key")
	}
	sig, err := hex.DecodeString(r.Signature)
	if err != nil || len(sig) != ed25519.SignatureSize {
		return errors.New("benchfmt: signature is malformed")
	}
	b, err := r.signingBytes()
	if err != nil {
		return err
	}
	if !ed25519.Verify(pub, b, sig) {
		return fmt.Errorf("benchfmt: signature does not match install_id %s…", r.InstallID[:12])
	}
	return nil
}

// WithdrawalMessage is what an install signs to withdraw a submission.
func WithdrawalMessage(id string) []byte { return []byte("nanollm.bench/v1 withdraw " + id) }
