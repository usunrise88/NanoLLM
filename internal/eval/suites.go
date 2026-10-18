package eval

import (
	"bytes"
	"embed"
	"fmt"
	"io/fs"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/usunrise88/nanollm/internal/core"
)

//go:embed suites/*.jsonl
var builtin embed.FS

// Suites lists the built-in suites: the files in suites/ and the generated
// long-context one.
func Suites() []string {
	entries, _ := fs.ReadDir(builtin, "suites")
	var out []string
	for _, e := range entries {
		out = append(out, strings.TrimSuffix(e.Name(), ".jsonl"))
	}
	out = append(out, "long-context", EraSuite)
	sort.Strings(out)
	return out
}

// EraSuite is the call-scenario suite (ROADMAP 3.6). Its items are the
// owner's call scripts and are never built in or committed: the name reads
// the file EraEnv points at. docs/eval/era.example.jsonl shows the shape.
const (
	EraSuite = "era"
	EraEnv   = "NANOLLM_ERA_SUITE"
)

// LoadSuite reads a built-in suite by name or a JSONL file by path; "era"
// reads the owner's file from NANOLLM_ERA_SUITE.
func LoadSuite(nameOrPath string) ([]Item, string, error) {
	if nameOrPath == EraSuite {
		p := os.Getenv(EraEnv)
		if p == "" {
			return nil, "", fmt.Errorf("the era suite is the owner's call scripts, not part of NanoLLM: set %s to its JSONL file (shape: docs/eval/era.example.jsonl)", EraEnv)
		}
		nameOrPath = p
	}
	if nameOrPath == "long-context" {
		items := LongContext([]int{1024, 4096, 8192, 16384})
		return items, "generated:long-context/v1", nil
	}
	if b, err := builtin.ReadFile("suites/" + nameOrPath + ".jsonl"); err == nil {
		return ReadSuite(bytes.NewReader(b))
	}
	f, err := os.Open(nameOrPath)
	if err != nil {
		return nil, "", fmt.Errorf("suite %q is neither built in (%s) nor a file: %w", nameOrPath, strings.Join(Suites(), ", "), err)
	}
	defer f.Close()
	return ReadSuite(f)
}

// LongContext generates needle-in-a-haystack items: a fact buried at the
// start, the middle and the end of a context of each length (in approximate
// tokens), for quality by position up to 16k (ROADMAP 2.5). The filler is
// deterministic, so the suite is the same on every run.
func LongContext(lengths []int) []Item {
	var items []Item
	for _, n := range lengths {
		for _, at := range []struct {
			name string
			frac float64
		}{{"start", 0.05}, {"middle", 0.5}, {"end", 0.95}} {
			code := strconv.Itoa(1000 + (n*7+int(at.frac*100))%9000)
			needle := fmt.Sprintf("Кодовое слово для сейфа номер %d — «%s».", n, "сокол-"+code)
			hay := haystack(n*3, needle, at.frac)
			items = append(items, Item{
				ID: fmt.Sprintf("needle-%d-%s", n, at.name), Lang: "ru", Capability: "long-context",
				Messages: []core.Message{{Role: core.RoleUser, Content: hay +
					fmt.Sprintf("\n\nКакое кодовое слово для сейфа номер %d? Ответь только кодовым словом.", n)}},
				Checker: Checker{Type: CheckNeedle, Want: "сокол-" + code}, MaxTokens: 32,
			})
		}
	}
	return items
}

// haystack is about chars characters of plain filler with the needle at frac.
func haystack(chars int, needle string, frac float64) string {
	sentences := []string{
		"Утром над рекой поднимался туман, и рыбаки не спешили выходить на лодках.",
		"В старой библиотеке пахло бумагой и пылью, а часы на стене отставали на пять минут.",
		"Поезд из соседнего города опаздывал, поэтому на перроне собралась небольшая толпа.",
		"Садовник поливал розы каждое утро и записывал в тетрадь, сколько бутонов раскрылось.",
		"Над площадью кружили голуби, а продавец газет громко называл свежие заголовки.",
		"Инженер проверял чертежи моста третий раз, потому что не любил ошибок.",
	}
	var sb strings.Builder
	placed := false
	for i := 0; sb.Len() < chars; i++ {
		if !placed && float64(sb.Len()) >= frac*float64(chars) {
			sb.WriteString(needle + " ")
			placed = true
		}
		sb.WriteString(sentences[i%len(sentences)] + " ")
	}
	if !placed {
		sb.WriteString(needle)
	}
	return sb.String()
}
