package grammar

import (
	"encoding/json"
	"os"
	"regexp"
	"strings"
	"testing"
)

type schemaCase struct {
	Name    string `json:"name"`
	OK      bool   `json:"ok"`
	Schema  string `json:"schema"`
	Grammar string `json:"grammar"`
}

var lineIndent = regexp.MustCompile(`(^|\n)[ \t]+`)

// trim is the test's own normalisation in llama.cpp: outer whitespace and
// each line's indentation go.
func trim(s string) string {
	s = strings.Trim(s, " \n\r\t")
	return lineIndent.ReplaceAllString(s, "$1")
}

// TestUpstreamCases holds the port to llama.cpp's own table
// (tests/test-json-schema-to-grammar.cpp, extracted by tools/grammarcases.py).
func TestUpstreamCases(t *testing.T) {
	b, err := os.ReadFile("testdata/schema_cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []schemaCase
	if err := json.Unmarshal(b, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) != 81 {
		t.Fatalf("%d cases, want the 81 of b11600", len(cases))
	}
	failed := 0
	for _, tc := range cases {
		got, _, err := SchemaToGBNF([]byte(tc.Schema))
		switch {
		case tc.OK && err != nil:
			t.Errorf("%s: unexpected error %v", tc.Name, err)
			failed++
		case !tc.OK && err == nil:
			t.Errorf("%s: expected a failure, got\n%s", tc.Name, got)
			failed++
		case tc.OK && trim(got) != trim(tc.Grammar):
			t.Errorf("%s:\n--- want\n%s\n--- got\n%s", tc.Name, trim(tc.Grammar), trim(got))
			failed++
		}
	}
	if failed > 0 {
		t.Logf("%d of %d cases differ", failed, len(cases))
	}
}

func TestNlohmannFloat(t *testing.T) {
	for in, want := range map[float64]string{
		1: "1.0", 1.5: "1.5", 0.0001: "0.0001", 0.00001: "1e-05", 1e15: "1e+15", 123456789012345: "123456789012345.0",
		-2.25: "-2.25", 1e300: "1e+300",
	} {
		if got := nlohmannFloat(in); got != want {
			t.Errorf("%v: got %s want %s", in, got, want)
		}
	}
}
