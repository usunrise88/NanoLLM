package eval

import (
	"bufio"
	"fmt"
	"io"
	"math/rand/v2"
	"strings"

	"github.com/usunrise88/nanollm/internal/core"
)

// The human side of the gate: one rater, small sets (ROADMAP: 50 prompts for
// the baseline, 20–30 pairs for a regression). Everything is driven from the
// keyboard, and nothing on screen says which configuration produced what.

// Rating is a person's judgement of one answer.
type Rating struct {
	Score int `json:"score"` // 1–5
	// Usable is «годится для звонка»: could this answer go to a caller.
	Usable bool   `json:"usable"`
	Note   string `json:"note,omitempty"`
}

// Preference is one blind A/B comparison; A and B name the reports, and
// Choice is "a", "b" or "tie" after unblinding.
type Preference struct {
	ID     string `json:"id"`
	A      string `json:"a"`
	B      string `json:"b"`
	Choice string `json:"choice"`
}

// RatingKey is how a result is addressed in Report.Ratings.
func RatingKey(o Outcome) string {
	if o.Thinking {
		return o.ID + "+thinking"
	}
	return o.ID
}

func lastUser(msgs []core.Message) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == core.RoleUser {
			return msgs[i].Content
		}
	}
	return ""
}

// RateReport asks for a score and the usable mark on every result left to a
// person (or every result, with all), skipping ones already rated so a
// session can be resumed. Typing q stops and keeps what was rated. The order
// is shuffled with rng, so a rater does not learn the suite's order.
func RateReport(in io.Reader, out io.Writer, rep *Report, items map[string]Item, all bool, rng *rand.Rand) (int, error) {
	if rep.Ratings == nil {
		rep.Ratings = map[string]Rating{}
	}
	var todo []Outcome
	for _, o := range rep.Results {
		if _, done := rep.Ratings[RatingKey(o)]; done {
			continue
		}
		if all || o.Pass == nil {
			todo = append(todo, o)
		}
	}
	rng.Shuffle(len(todo), func(i, j int) { todo[i], todo[j] = todo[j], todo[i] })
	r := bufio.NewReader(in)
	rated := 0
	for n, o := range todo {
		fmt.Fprintf(out, "\n──── %d / %d ────\n\nЗапрос:\n%s\n\nОтвет:\n%s\n\n", n+1, len(todo), lastUser(items[o.ID].Messages), o.Answer)
		score, err := askInt(r, out, "Оценка 1–5 (q — закончить): ", 1, 5)
		if err != nil {
			return rated, err
		}
		if score == 0 {
			return rated, nil
		}
		usable, err := askYesNo(r, out, "Годится для звонка? y/n: ")
		if err != nil {
			return rated, err
		}
		fmt.Fprint(out, "Заметка (Enter — без заметки): ")
		note, _ := r.ReadString('\n')
		rep.Ratings[RatingKey(o)] = Rating{Score: score, Usable: usable, Note: strings.TrimSpace(note)}
		rated++
	}
	return rated, nil
}

// CompareReports runs a blind A/B over the items both reports answered: the
// two answers are shown as 1 and 2 in random order, and the choice is
// recorded against the report names only afterwards.
func CompareReports(in io.Reader, out io.Writer, nameA string, a *Report, nameB string, b *Report, items map[string]Item, rng *rand.Rand) ([]Preference, error) {
	byKey := map[string]Outcome{}
	for _, o := range b.Results {
		byKey[RatingKey(o)] = o
	}
	var pairs [][2]Outcome
	for _, o := range a.Results {
		if ob, ok := byKey[RatingKey(o)]; ok {
			pairs = append(pairs, [2]Outcome{o, ob})
		}
	}
	rng.Shuffle(len(pairs), func(i, j int) { pairs[i], pairs[j] = pairs[j], pairs[i] })
	r := bufio.NewReader(in)
	var prefs []Preference
	for n, p := range pairs {
		swap := rng.IntN(2) == 1
		first, second := p[0], p[1]
		if swap {
			first, second = second, first
		}
		fmt.Fprintf(out, "\n──── %d / %d ────\n\nЗапрос:\n%s\n\n[1]\n%s\n\n[2]\n%s\n\n", n+1, len(pairs), lastUser(items[p[0].ID].Messages), first.Answer, second.Answer)
		choice := ""
		for choice == "" {
			fmt.Fprint(out, "Лучше 1, 2 или = (q — закончить): ")
			line, err := r.ReadString('\n')
			if err != nil && line == "" {
				return prefs, err
			}
			switch strings.TrimSpace(line) {
			case "1":
				choice = map[bool]string{false: "a", true: "b"}[swap]
			case "2":
				choice = map[bool]string{false: "b", true: "a"}[swap]
			case "=":
				choice = "tie"
			case "q":
				return prefs, nil
			}
		}
		prefs = append(prefs, Preference{ID: RatingKey(p[0]), A: nameA, B: nameB, Choice: choice})
	}
	return prefs, nil
}

func askInt(r *bufio.Reader, out io.Writer, prompt string, lo, hi int) (int, error) {
	for {
		fmt.Fprint(out, prompt)
		line, err := r.ReadString('\n')
		s := strings.TrimSpace(line)
		if s == "q" {
			return 0, nil
		}
		var n int
		if _, e := fmt.Sscan(s, &n); e == nil && n >= lo && n <= hi {
			return n, nil
		}
		if err != nil {
			return 0, err
		}
	}
}

func askYesNo(r *bufio.Reader, out io.Writer, prompt string) (bool, error) {
	for {
		fmt.Fprint(out, prompt)
		line, err := r.ReadString('\n')
		switch strings.ToLower(strings.TrimSpace(line)) {
		case "y", "д", "да", "yes":
			return true, nil
		case "n", "н", "нет", "no":
			return false, nil
		}
		if err != nil {
			return false, err
		}
	}
}
