package eval

import (
	"bytes"
	"context"
	"math/rand/v2"
	"regexp"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/engine"
	"github.com/usunrise88/nanollm/internal/pool"
)

func TestBuiltinSuitesParse(t *testing.T) {
	for _, name := range Suites() {
		items, sha, err := LoadSuite(name)
		if err != nil || len(items) == 0 || sha == "" {
			t.Fatalf("%s: %v (%d items)", name, err, len(items))
		}
	}
	h, _, _ := LoadSuite("human-ru")
	if len(h) != 50 {
		t.Fatalf("the human set has %d prompts, want 50", len(h))
	}
}

func TestCheckers(t *testing.T) {
	yes := func(c Checker, answer string, calls ...core.ToolCall) bool {
		p, _ := c.Check(answer, calls)
		return p != nil && *p
	}
	if !yes(Checker{Type: CheckRegex, Pattern: `^51\.?$`}, " 51 ") || yes(Checker{Type: CheckRegex, Pattern: `^51$`}, "51 apples") {
		t.Fatal("regex")
	}
	if !yes(Checker{Type: CheckContains, Any: []string{"не могу"}}, "Я НЕ МОГУ этого сделать") || yes(Checker{Type: CheckContains, None: []string{"номер"}}, "вот номер") {
		t.Fatal("contains")
	}
	schema := []byte(`{"type":"object","properties":{"age":{"type":"integer"},"s":{"enum":["negative"]}},"required":["age"]}`)
	if !yes(Checker{Type: CheckJSON, Schema: schema}, `{"age": 3, "s": "negative"}`) || yes(Checker{Type: CheckJSON, Schema: schema}, `{"age": 3.5}`) || yes(Checker{Type: CheckJSON}, "nope") {
		t.Fatal("json")
	}
	call := core.ToolCall{Name: "get_weather", Arguments: `{"city":"казань "}`}
	if !yes(Checker{Type: CheckToolCall, Name: "get_weather", Args: map[string]any{"city": "Казань"}}, "", call) ||
		yes(Checker{Type: CheckToolCall, Name: "book"}, "", call) {
		t.Fatal("tool_call")
	}
	if p, _ := (Checker{Type: CheckHuman}).Check("x", nil); p != nil {
		t.Fatal("a human item is not machine-checked")
	}
}

func TestLongContextPlacesNeedle(t *testing.T) {
	items := LongContext([]int{1024})
	if len(items) != 3 {
		t.Fatal(len(items))
	}
	for _, it := range items {
		if !strings.Contains(it.Messages[0].Content, it.Checker.Want) {
			t.Fatalf("%s: needle missing", it.ID)
		}
	}
}

func TestRunOnFake(t *testing.T) {
	p := pool.New(func(ctx context.Context, k pool.Key) (backend.Model, any, error) {
		m, err := fake.New(fake.Options{}).Load(ctx, backend.LoadSpec{NCtx: 8192, NBatch: 8192})
		return m, nil, err
	}, 1)
	defer p.Close()
	svc := engine.New(engine.Config{DefaultModel: "m"}, p, func(context.Context, string, string) (engine.Target, error) {
		return engine.Target{Key: pool.Key{Model: "m", Backend: "fake", NCtx: 8192}}, nil
	})
	suite := `// a note
{"id":"a","lang":"ru","capability":"echo","messages":[{"role":"user","content":"да [[fake:echo]]"}],"checker":{"type":"regex","pattern":"^да$"}}
{"id":"b","lang":"en","capability":"echo","messages":[{"role":"user","content":"no [[fake:echo]]"}],"checker":{"type":"exact","want":"yes"}}
{"id":"c","lang":"ru","capability":"human","messages":[{"role":"user","content":"x"}],"checker":{"type":"human"}}
`
	items, sha, err := ReadSuite(strings.NewReader(suite))
	if err != nil {
		t.Fatal(err)
	}
	rep, err := Run(context.Background(), svc, "t", sha, items, Options{Model: "m", Thinking: "both"})
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Results) != 6 {
		t.Fatalf("%d results", len(rep.Results))
	}
	if s := rep.Summary["echo"]; s.N != 2 || s.Pass != 1 || rep.Summary["human"].Human != 1 {
		t.Fatalf("%+v", rep.Summary)
	}

	// Rating: the human items, in shuffled order, with a resumable q.
	byID := map[string]Item{}
	for _, it := range items {
		byID[it.ID] = it
	}
	var out bytes.Buffer
	n, err := RateReport(strings.NewReader("4\nд\nхорошо\nq\n"), &out, rep, byID, false, rand.New(rand.NewPCG(1, 2)))
	if err != nil || n != 1 || len(rep.Ratings) != 1 {
		t.Fatalf("rated %d, %v, %+v", n, err, rep.Ratings)
	}
	for _, r := range rep.Ratings {
		if r.Score != 4 || !r.Usable || r.Note != "хорошо" {
			t.Fatalf("%+v", r)
		}
	}

	// A/B: answering "1" every time must not always pick the same report.
	prefs, err := CompareReports(strings.NewReader(strings.Repeat("1\n", 6)), &out, "A", rep, "B", rep, byID, rand.New(rand.NewPCG(3, 4)))
	if err != nil || len(prefs) != 6 {
		t.Fatalf("%v %d", err, len(prefs))
	}
	seen := map[string]bool{}
	for _, p := range prefs {
		seen[p.Choice] = true
	}
	if !seen["a"] || !seen["b"] {
		t.Fatalf("the order was not randomised: %+v", prefs)
	}
}

// Every built-in suite parses, and the answer line of the reasoning suite is
// checked by a pattern that accepts the value and nothing near it.
func TestBuiltinSuitesLoad(t *testing.T) {
	for _, name := range Suites() {
		if name == "long-context" {
			continue
		}
		items, _, err := LoadSuite(name)
		if err != nil || len(items) == 0 {
			t.Errorf("%s: %d items, %v", name, len(items), err)
		}
	}
}

func TestReasoningSuiteChecker(t *testing.T) {
	items, _, err := LoadSuite("reasoning")
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 48 {
		t.Fatalf("%d items", len(items))
	}
	it := items[0]                                                                // reason-arith-ru-1: the answer is a number
	m := regexp.MustCompile(`\]\*(\d+)\[`).FindStringSubmatch(it.Checker.Pattern) // …ANSWER:\s*[*`]*355[*`]*…
	if m == nil {
		t.Fatalf("pattern %q", it.Checker.Pattern)
	}
	right := m[1]
	for answer, want := range map[string]bool{
		"Сначала считаем.\nANSWER: " + right:           true,
		"…\n**ANSWER: " + right + "**":                 true,
		"…\nanswer: " + right + ".":                    true,
		"ANSWER: " + right + "0":                       false, // a longer number is not the value
		"ANSWER: 1" + right:                            false,
		"The result is " + right + " (no answer line)": false,
	} {
		if pass, _ := it.Checker.Check(answer, nil); *pass != want {
			t.Errorf("%q: %v", answer, *pass)
		}
	}
}
