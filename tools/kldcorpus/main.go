// Command kldcorpus builds the text corpus the KL-divergence gate measures on
// (docs/STAGE2.md 2.4): Wikipedia articles in Russian and English, each at a
// pinned revision, so that every machine builds the same bytes.
//
//	go run ./tools/kldcorpus -manifest testdata/kld/corpus.json -out .cache/kld/corpus.txt
//	go run ./tools/kldcorpus -manifest testdata/kld/corpus.json -update   # pin revisions and hashes
//
// The manifest lists articles by language and title. -update resolves each to
// its current revision (revid), fetches the rendered text and records the
// revid and the sha256 of the extracted text; a plain run fetches those exact
// revisions and refuses a text whose hash differs, which is how a change in
// Wikipedia's renderer shows up instead of silently changing the corpus.
//
// The texts are CC BY-SA 4.0 (Wikipedia); the corpus is built on the machine
// that measures and is not committed. The manifest is the record of what it
// is made of.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// Article is one corpus entry.
type Article struct {
	Lang   string `json:"lang"`
	Title  string `json:"title"`
	RevID  int64  `json:"revid,omitempty"`
	SHA256 string `json:"sha256,omitempty"` // of the extracted text
	Bytes  int    `json:"bytes,omitempty"`
}

// Manifest is testdata/kld/corpus.json.
type Manifest struct {
	Note    string `json:"note"`
	License string `json:"license"`
	// MaxBytes cuts each article to its first paragraphs that fit: the gate
	// reads every token of the corpus on a CPU, a minute per thousand, and
	// the opening of an article is as varied as the whole.
	MaxBytes int       `json:"max_bytes"`
	Articles []Article `json:"articles"`
}

func main() {
	manifest := flag.String("manifest", "testdata/kld/corpus.json", "the article list")
	out := flag.String("out", ".cache/kld/corpus.txt", "the corpus file to write")
	update := flag.Bool("update", false, "pin each article to its current revision and record the text hash")
	flag.Parse()
	if err := run(context.Background(), *manifest, *out, *update); err != nil {
		fmt.Fprintln(os.Stderr, "kldcorpus:", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, manifestPath, out string, update bool) error {
	b, err := os.ReadFile(manifestPath)
	if err != nil {
		return err
	}
	var m Manifest
	if err := json.Unmarshal(b, &m); err != nil {
		return fmt.Errorf("%s: %w", manifestPath, err)
	}
	if len(m.Articles) == 0 {
		return errors.New("the manifest lists no articles")
	}
	var corpus strings.Builder
	for i := range m.Articles {
		a := &m.Articles[i]
		if update || a.RevID == 0 {
			if !update {
				return fmt.Errorf("%s/%s has no revid; run with -update", a.Lang, a.Title)
			}
			if a.RevID, err = latestRevision(ctx, a.Lang, a.Title); err != nil {
				return fmt.Errorf("%s/%s: %w", a.Lang, a.Title, err)
			}
		}
		text, err := fetchText(ctx, a.Lang, a.RevID)
		if err != nil {
			return fmt.Errorf("%s/%s: %w", a.Lang, a.Title, err)
		}
		text = Truncate(text, m.MaxBytes)
		sum := sha256.Sum256([]byte(text))
		got := hex.EncodeToString(sum[:])
		switch {
		case update:
			a.SHA256, a.Bytes = got, len(text)
		case a.SHA256 != got:
			return fmt.Errorf("%s/%s at revision %d extracts to sha256 %s, the manifest has %s: Wikipedia's rendering changed; review and run -update", a.Lang, a.Title, a.RevID, got, a.SHA256)
		}
		corpus.WriteString(text)
		corpus.WriteString("\n\n")
		fmt.Fprintf(os.Stderr, "%-3s %-40s rev %-11d %7d bytes\n", a.Lang, a.Title, a.RevID, len(text))
	}
	if update {
		enc, _ := json.MarshalIndent(m, "", "  ")
		if err := os.WriteFile(manifestPath, append(enc, '\n'), 0o644); err != nil {
			return err
		}
	}
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(out, []byte(corpus.String()), 0o644); err != nil {
		return err
	}
	sum := sha256.Sum256([]byte(corpus.String()))
	fmt.Fprintf(os.Stderr, "%s: %d articles, %d bytes, sha256 %s\n", out, len(m.Articles), corpus.Len(), hex.EncodeToString(sum[:]))
	return nil
}

var client = &http.Client{Timeout: 60 * time.Second}

// api calls a Wikipedia API with the polite User-Agent its policy asks for.
func api(ctx context.Context, lang string, q url.Values, v any) error {
	q.Set("format", "json")
	q.Set("formatversion", "2")
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://"+lang+".wikipedia.org/w/api.php?"+q.Encode(), nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "NanoLLM-kldcorpus/1 (https://github.com/usunrise88/NanoLLM; quality-gate corpus)")
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: %s", req.URL.Host, resp.Status)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
	if err != nil {
		return err
	}
	return json.Unmarshal(data, v)
}

func latestRevision(ctx context.Context, lang, title string) (int64, error) {
	var r struct {
		Query struct {
			Pages []struct {
				Missing   bool  `json:"missing"`
				LastRevID int64 `json:"lastrevid"`
			} `json:"pages"`
		} `json:"query"`
	}
	if err := api(ctx, lang, url.Values{"action": {"query"}, "prop": {"info"}, "titles": {title}, "redirects": {"1"}}, &r); err != nil {
		return 0, err
	}
	if len(r.Query.Pages) != 1 || r.Query.Pages[0].Missing || r.Query.Pages[0].LastRevID == 0 {
		return 0, errors.New("no such article")
	}
	return r.Query.Pages[0].LastRevID, nil
}

func fetchText(ctx context.Context, lang string, revid int64) (string, error) {
	var r struct {
		Parse struct {
			Text string `json:"text"`
		} `json:"parse"`
		Error struct {
			Info string `json:"info"`
		} `json:"error"`
	}
	q := url.Values{"action": {"parse"}, "oldid": {fmt.Sprint(revid)}, "prop": {"text"}, "disablelimitreport": {"1"}, "disableeditsection": {"1"}, "disabletoc": {"1"}}
	if err := api(ctx, lang, q, &r); err != nil {
		return "", err
	}
	if r.Error.Info != "" {
		return "", errors.New(r.Error.Info)
	}
	text := ProseOf(r.Parse.Text)
	if len(text) < 1000 {
		return "", fmt.Errorf("only %d bytes of prose", len(text))
	}
	return text, nil
}

var (
	reBlock   = regexp.MustCompile(`(?s)<(p|h[2-4]|li)\b[^>]*>(.*?)</(?:p|h[2-4]|li)>`)
	reDrop    = regexp.MustCompile(`(?s)<(sup|style|script|table|figure|math|annotation)\b[^>]*>.*?</(?:sup|style|script|table|figure|math|annotation)>`)
	reTag     = regexp.MustCompile(`(?s)<[^>]*>`)
	reSpace   = regexp.MustCompile(`[ \t\x{a0}\x{2009}]+`)
	reBracket = regexp.MustCompile(`\[[^\]]{0,24}\]`)
)

// ProseOf extracts the running text of a rendered article: paragraphs, list
// items and the headings above them, without references, tables, figures or
// markup.
func ProseOf(htmlText string) string {
	htmlText = reDrop.ReplaceAllString(htmlText, "")
	var b strings.Builder
	for _, m := range reBlock.FindAllStringSubmatch(htmlText, -1) {
		s := reTag.ReplaceAllString(m[2], "")
		s = html.UnescapeString(s)
		s = reBracket.ReplaceAllString(s, "") // [1], [note 2], [citation needed]
		s = reSpace.ReplaceAllString(s, " ")
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		switch m[1] {
		case "li":
			if len([]rune(s)) < 60 { // menus and short lists are not prose
				continue
			}
		case "h2", "h3", "h4":
			b.WriteString("\n")
		}
		b.WriteString(s)
		b.WriteString("\n")
	}
	return strings.TrimSpace(b.String()) + "\n"
}

// Truncate keeps the leading lines of text that fit in max bytes; max <= 0
// keeps everything.
func Truncate(text string, max int) string {
	if max <= 0 || len(text) <= max {
		return text
	}
	var b strings.Builder
	for _, line := range strings.SplitAfter(text, "\n") {
		if b.Len()+len(line) > max && b.Len() > 0 {
			break
		}
		b.WriteString(line)
	}
	return strings.TrimRight(b.String(), "\n") + "\n"
}
