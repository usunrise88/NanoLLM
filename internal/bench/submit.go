package bench

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/usunrise88/nanollm/benchfmt"
)

// ConsentVersion names the consent text below; a submission carries it, so
// the service knows what its sender agreed to.
const ConsentVersion = "2026-10"

// ConsentText is shown before anything is sent.
const ConsentText = `Отправить результат на публичный бенчмарк NanoLLM?

Уйдёт ровно то, что показано выше: метрики пресетов, версии NanoLLM и
llama.cpp, хэши модели и профиль железа — процессор, память, видеокарта,
производитель и модель компьютера, ОС, источник питания, тип диска.
Не уходят: имя компьютера, имя пользователя, серийные номера, UUID,
MAC-адреса, IP-адрес не хранится. Тексты ответов модели не уходят — только
их хэш. Результат будет опубликован под открытой лицензией. Удалить его
можно командой ` + "`nanollm bench withdraw ID`" + `.

Send this result to the public NanoLLM benchmark? Exactly what is shown
above is sent; no host or user names, serial numbers, UUIDs or MAC
addresses, and the IP address is not stored. It is published under an open
licence and can be withdrawn with ` + "`nanollm bench withdraw ID`" + `.`

// DefaultSubmitURL is the public benchmark service.
const DefaultSubmitURL = "https://nano.llmto.day"

// Identity is this installation's key: it proves ownership of its
// submissions (to withdraw them, to count repeats once), nothing more.
type Identity struct {
	dir string
	key ed25519.PrivateKey
}

// LoadIdentity reads <dataDir>/bench/identity.key, creating it on first use.
func LoadIdentity(dataDir string) (*Identity, error) {
	dir := filepath.Join(dataDir, "bench")
	p := filepath.Join(dir, "identity.key")
	if b, err := os.ReadFile(p); err == nil {
		seed, err := hex.DecodeString(strings.TrimSpace(string(b)))
		if err != nil || len(seed) != ed25519.SeedSize {
			return nil, fmt.Errorf("%s is not an ed25519 seed", p)
		}
		return &Identity{dir: dir, key: ed25519.NewKeyFromSeed(seed)}, nil
	}
	_, key, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	if err := os.WriteFile(p, []byte(hex.EncodeToString(key.Seed())+"\n"), 0o600); err != nil {
		return nil, err
	}
	return &Identity{dir: dir, key: key}, nil
}

// Anonymous is a throwaway identity: the submission cannot be withdrawn and
// is not linked to the installation's others.
func Anonymous() *Identity {
	_, key, _ := ed25519.GenerateKey(rand.Reader)
	return &Identity{key: key}
}

// Receipt is what the service answers to a submission.
type Receipt struct {
	ID  string `json:"id"`
	URL string `json:"url"`
}

// Client talks to the benchmark service.
type Client struct {
	BaseURL string
	HTTP    *http.Client
}

func (c *Client) http() *http.Client {
	if c.HTTP != nil {
		return c.HTTP
	}
	return &http.Client{Timeout: 60 * time.Second}
}

// Submit signs and sends a result. With a persistent identity the receipt is
// remembered, for withdraw.
func (c *Client) Submit(ctx context.Context, res *benchfmt.Result, id *Identity) (*Receipt, error) {
	if res.Client.GGMLCPUVariant == benchfmt.FakeCPUVariant && strings.TrimSuffix(c.BaseURL, "/") == DefaultSubmitURL {
		return nil, errors.New("this result was measured on the fake backend; it is for testing and is not sent to the public benchmark")
	}
	if res.Suite.ID != SuiteStandard {
		return nil, fmt.Errorf("only the %s suite is submitted; %s results stay local", SuiteStandard, res.Suite.ID)
	}
	if err := res.Sign(id.key); err != nil {
		return nil, err
	}
	if err := res.Validate(); err != nil {
		return nil, err
	}
	body, _ := json.Marshal(res)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimSuffix(c.BaseURL, "/")+"/api/v1/submissions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http().Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	switch resp.StatusCode {
	case http.StatusGone, http.StatusUpgradeRequired:
		return nil, fmt.Errorf("the service no longer accepts suite %s/%s; update NanoLLM (%s)", res.Suite.ID, res.Suite.Version, strings.TrimSpace(string(b)))
	}
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("the service refused the result: %s %s", resp.Status, strings.TrimSpace(string(b)))
	}
	var r Receipt
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, fmt.Errorf("unexpected answer from the service: %w", err)
	}
	if id.dir != "" {
		_ = appendReceipt(id.dir, r)
	}
	return &r, nil
}

// Withdraw deletes a submission this installation made.
func (c *Client) Withdraw(ctx context.Context, subID string, id *Identity) error {
	body, _ := json.Marshal(map[string]string{
		"install_id": hex.EncodeToString(id.key.Public().(ed25519.PublicKey)),
		"signature":  hex.EncodeToString(ed25519.Sign(id.key, benchfmt.WithdrawalMessage(subID))),
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, strings.TrimSuffix(c.BaseURL, "/")+"/api/v1/submissions/"+subID, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("withdraw %s: %s %s", subID, resp.Status, strings.TrimSpace(string(b)))
	}
	return nil
}

func appendReceipt(dir string, r Receipt) error {
	p := filepath.Join(dir, "submissions.jsonl")
	f, err := os.OpenFile(p, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	b, _ := json.Marshal(struct {
		Receipt
		Date string `json:"date"`
	}{r, time.Now().UTC().Format(time.RFC3339)})
	_, err = f.Write(append(b, '\n'))
	return err
}

// ErrDeclined is returned when the person says no.
var ErrDeclined = errors.New("not sent")
