package grammar

import (
	"errors"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// The converter, ported from llama.cpp's common/json-schema-to-grammar.cpp
// (b11600) function for function, so that a request's grammar is the one
// llama-server would build for it. testdata/schema_cases.json is upstream's
// own test table, extracted by tools/grammarcases.py; every case must match.

const maxInt = math.MaxInt32

func buildRepetition(item string, minItems, maxItems int, sep string) string {
	hasMax := maxItems != maxInt
	if maxItems == 0 {
		return ""
	}
	if minItems == 0 && maxItems == 1 {
		return item + "?"
	}
	if sep == "" {
		if minItems == 1 && !hasMax {
			return item + "+"
		}
		if minItems == 0 && !hasMax {
			return item + "*"
		}
		s := item + "{" + strconv.Itoa(minItems) + ","
		if hasMax {
			s += strconv.Itoa(maxItems)
		}
		return s + "}"
	}
	m := 0
	if minItems > 0 {
		m = minItems - 1
	}
	mx := maxItems
	if hasMax {
		mx = maxItems - 1
	}
	result := item + " " + buildRepetition("("+sep+" "+item+")", m, mx, "")
	if minItems == 0 {
		result = "(" + result + ")?"
	}
	return result
}

func buildMinMaxInt(minV, maxV int64, out *strings.Builder, decimalsLeft int, topLevel bool) {
	hasMin := minV != math.MinInt64
	hasMax := maxV != math.MaxInt64

	digitRange := func(from, to byte) {
		out.WriteByte('[')
		if from == to {
			out.WriteByte(from)
		} else {
			out.WriteByte(from)
			out.WriteByte('-')
			out.WriteByte(to)
		}
		out.WriteByte(']')
	}
	moreDigits := func(minD, maxD int) {
		out.WriteString("[0-9]")
		if minD == maxD && minD == 1 {
			return
		}
		out.WriteString("{" + strconv.Itoa(minD))
		if maxD != minD {
			out.WriteString(",")
			if maxD != maxInt {
				out.WriteString(strconv.Itoa(maxD))
			}
		}
		out.WriteString("}")
	}
	var uniformRange func(from, to string)
	uniformRange = func(from, to string) {
		i := 0
		for i < len(from) && i < len(to) && from[i] == to[i] {
			i++
		}
		if i > 0 {
			out.WriteString(`"` + from[:i] + `"`)
		}
		if i < len(from) && i < len(to) {
			if i > 0 {
				out.WriteString(" ")
			}
			subLen := len(from) - i - 1
			if subLen > 0 {
				fromSub, toSub := from[i+1:], to[i+1:]
				zeros, nines := strings.Repeat("0", subLen), strings.Repeat("9", subLen)
				toReached := false
				out.WriteString("(")
				if fromSub == zeros {
					digitRange(from[i], to[i]-1)
					out.WriteString(" ")
					moreDigits(subLen, subLen)
				} else {
					out.WriteString("[" + string(from[i]) + "] ")
					out.WriteString("(")
					uniformRange(fromSub, nines)
					out.WriteString(")")
					if from[i] < to[i]-1 {
						out.WriteString(" | ")
						if toSub == nines {
							digitRange(from[i]+1, to[i])
							toReached = true
						} else {
							digitRange(from[i]+1, to[i]-1)
						}
						out.WriteString(" ")
						moreDigits(subLen, subLen)
					}
				}
				if !toReached {
					out.WriteString(" | ")
					digitRange(to[i], to[i])
					out.WriteString(" ")
					uniformRange(zeros, toSub)
				}
				out.WriteString(")")
			} else {
				out.WriteString("[" + string(from[i]) + "-" + string(to[i]) + "]")
			}
		}
	}

	if hasMin && hasMax {
		if minV < 0 && maxV < 0 {
			out.WriteString(`"-" (`)
			buildMinMaxInt(-maxV, -minV, out, decimalsLeft, true)
			out.WriteString(")")
			return
		}
		if minV < 0 {
			out.WriteString(`"-" (`)
			buildMinMaxInt(0, -minV, out, decimalsLeft, true)
			out.WriteString(") | ")
			minV = 0
		}
		minS, maxS := strconv.FormatInt(minV, 10), strconv.FormatInt(maxV, 10)
		for digits := len(minS); digits < len(maxS); digits++ {
			uniformRange(minS, strings.Repeat("9", digits))
			minS = "1" + strings.Repeat("0", digits)
			out.WriteString(" | ")
		}
		uniformRange(minS, maxS)
		return
	}

	lessDecimals := max(decimalsLeft-1, 1)

	if hasMin {
		switch {
		case minV < 0:
			out.WriteString(`"-" (`)
			buildMinMaxInt(math.MinInt64, -minV, out, decimalsLeft, false)
			out.WriteString(") | [0] | [1-9] ")
			moreDigits(0, decimalsLeft-1)
		case minV == 0:
			if topLevel {
				out.WriteString("[0] | [1-9] ")
				moreDigits(0, lessDecimals)
			} else {
				moreDigits(1, decimalsLeft)
			}
		case minV <= 9:
			c := byte('0' + minV)
			rangeStart := byte('0')
			if topLevel {
				rangeStart = '1'
			}
			if c > rangeStart {
				digitRange(rangeStart, c-1)
				out.WriteString(" ")
				moreDigits(1, lessDecimals)
				out.WriteString(" | ")
			}
			digitRange(c, '9')
			out.WriteString(" ")
			moreDigits(0, lessDecimals)
		default:
			minS := strconv.FormatInt(minV, 10)
			l := len(minS)
			c := minS[0]
			if c > '1' {
				start := byte('0')
				if topLevel {
					start = '1'
				}
				digitRange(start, c-1)
				out.WriteString(" ")
				moreDigits(l, lessDecimals)
				out.WriteString(" | ")
			}
			digitRange(c, c)
			out.WriteString(" (")
			rest, _ := strconv.ParseInt(minS[1:], 10, 64)
			buildMinMaxInt(rest, math.MaxInt64, out, lessDecimals, false)
			out.WriteString(")")
			if c < '9' {
				out.WriteString(" | ")
				digitRange(c+1, '9')
				out.WriteString(" ")
				moreDigits(l-1, lessDecimals)
			}
		}
		return
	}

	if hasMax {
		if maxV >= 0 {
			if topLevel {
				out.WriteString(`"-" [1-9] `)
				moreDigits(0, lessDecimals)
				out.WriteString(" | ")
			}
			buildMinMaxInt(0, maxV, out, decimalsLeft, true)
		} else {
			out.WriteString(`"-" (`)
			buildMinMaxInt(-maxV, math.MaxInt64, out, decimalsLeft, false)
			out.WriteString(")")
		}
		return
	}
	panic("buildMinMaxInt: at least one of min and max must be set")
}

const spaceRule = `| " " | "\n"{1,2} [ \t]{0,20}`

type builtinRule struct {
	content string
	deps    []string
}

var primitiveRules = map[string]builtinRule{
	"boolean":       {`("true" | "false")`, nil},
	"decimal-part":  {`[0-9]{1,16}`, nil},
	"integral-part": {`[0] | [1-9] [0-9]{0,15}`, nil},
	"number":        {`("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?`, []string{"integral-part", "decimal-part"}},
	"integer":       {`("-"? integral-part)`, []string{"integral-part"}},
	"value":         {`object | array | string | number | boolean | null`, []string{"object", "array", "string", "number", "boolean", "null"}},
	"object":        {`"{" space ( string ":" space value ("," space string ":" space value)* )? space "}"`, []string{"string", "value"}},
	"array":         {`"[" space ( value ("," space value)* )? space "]"`, []string{"value"}},
	"uuid":          {`"\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\""`, nil},
	"char":          {`[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})`, nil},
	"string":        {`"\"" char* "\""`, []string{"char"}},
	"null":          {`"null"`, nil},
}

var stringFormatRules = map[string]builtinRule{
	"date":             {`[0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )`, nil},
	"time":             {`([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )`, nil},
	"date-time":        {`date "T" time`, []string{"date", "time"}},
	"date-string":      {`"\"" date "\""`, []string{"date"}},
	"time-string":      {`"\"" time "\""`, []string{"time"}},
	"date-time-string": {`"\"" date-time "\""`, []string{"date-time"}},
}

func isReservedName(name string) bool {
	if name == "root" {
		return true
	}
	_, p := primitiveRules[name]
	_, f := stringFormatRules[name]
	return p || f
}

var invalidRuleChars = regexp.MustCompile(`[^a-zA-Z0-9-]+`)

func formatLiteral(lit string) string {
	var sb strings.Builder
	sb.WriteByte('"')
	for i := 0; i < len(lit); i++ {
		switch c := lit[i]; c {
		case '\r':
			sb.WriteString(`\r`)
		case '\n':
			sb.WriteString(`\n`)
		case '"':
			sb.WriteString(`\"`)
		case '\\':
			sb.WriteString(`\\`)
		default:
			sb.WriteByte(c)
		}
	}
	sb.WriteByte('"')
	return sb.String()
}

// gbnfEscapeLength is the length of a GBNF escape at pos, 0 when there is
// none the grammar parser (llama-grammar.cpp parse_char) would accept.
func gbnfEscapeLength(p string, pos int) int {
	if pos+1 >= len(p) || p[pos] != '\\' {
		return 0
	}
	nHex := 0
	switch p[pos+1] {
	case 'x':
		nHex = 2
	case 'u':
		nHex = 4
	case 'U':
		nHex = 8
	case 't', 'r', 'n', '\\', '"', '[', ']', '-':
		return 2
	default:
		return 0
	}
	if pos+2+nHex > len(p) {
		return 0
	}
	for _, h := range p[pos+2 : pos+2+nHex] {
		if !(h >= '0' && h <= '9' || h >= 'a' && h <= 'f' || h >= 'A' && h <= 'F') {
			return 0
		}
	}
	return 2 + nHex
}

const maxPatternDepth = 100

func isNonLiteral(c byte) bool { return strings.IndexByte("|.()[]{}*+?^$", c) >= 0 }

func escapedInRegexpsNotLiterals(c byte) bool { return strings.IndexByte("^$.[]()|{}*+?", c) >= 0 }

type converter struct {
	dotall    bool
	rules     map[string]string
	resolving map[string]bool
	errs      []string
	warnings  []string
}

func newConverter(dotall bool) *converter {
	return &converter{dotall: dotall, rules: map[string]string{"space": spaceRule}, resolving: map[string]bool{}}
}

func (c *converter) addRule(name, rule string) string {
	esc := invalidRuleChars.ReplaceAllString(name, "-")
	if r, ok := c.rules[esc]; !ok || r == rule {
		c.rules[esc] = rule
		return esc
	}
	i := 0
	for {
		r, ok := c.rules[esc+strconv.Itoa(i)]
		if !ok || r == rule {
			break
		}
		i++
	}
	key := esc + strconv.Itoa(i)
	c.rules[key] = rule
	return key
}

func (c *converter) addPrimitive(name string, rule builtinRule) string {
	n := c.addRule(name, rule.content)
	for _, dep := range rule.deps {
		r, ok := primitiveRules[dep]
		if !ok {
			r, ok = stringFormatRules[dep]
			if !ok {
				c.errs = append(c.errs, "Rule "+dep+" not known")
				continue
			}
		}
		if _, have := c.rules[dep]; !have {
			c.addPrimitive(dep, r)
		}
	}
	return n
}

func (c *converter) unionRule(name string, alts []*snode) string {
	rules := make([]string, len(alts))
	for i, a := range alts {
		sep := "-"
		if name == "" {
			sep = "alternative-"
		}
		rules[i] = c.visit(a, name+sep+strconv.Itoa(i))
	}
	return strings.Join(rules, " | ")
}

type unsupportedPattern struct{ msg string }
type invalidPattern struct{ msg string }

func (c *converter) visitPattern(pattern, name string) (out string) {
	snapshot := make(map[string]string, len(c.rules))
	for k, v := range c.rules {
		snapshot[k] = v
	}
	defer func() {
		switch e := recover().(type) {
		case nil:
		case unsupportedPattern:
			c.rules = snapshot
			c.warnings = append(c.warnings, "pattern "+pattern+" is not supported ("+e.msg+"), accepting any string")
			out = c.addRule(name, c.addPrimitive("string", primitiveRules["string"]))
		case invalidPattern:
			c.rules = snapshot
			c.errs = append(c.errs, "Invalid pattern "+pattern+": "+e.msg)
			out = ""
		default:
			panic(e)
		}
	}()
	return c.patternToRule(pattern, name)
}

type litOrRule struct {
	s       string
	literal bool
}

func toRule(ls litOrRule) string {
	if ls.literal {
		return `"` + ls.s + `"`
	}
	return ls.s
}

func (c *converter) patternToRule(pattern, name string) string {
	if len(pattern) < 2 || pattern[0] != '^' || pattern[len(pattern)-1] != '$' {
		panic(unsupportedPattern{"not anchored with '^' and '$'"})
	}
	sub := pattern[1 : len(pattern)-1]
	subRuleIDs := map[string]string{}
	var subOrder int
	i, length, depth := 0, len(sub), 0

	var transform func() litOrRule
	transform = func() litOrRule {
		var seq []litOrRule
		getDot := func() string {
			rule := `[^\x0A\x0D]`
			if c.dotall {
				rule = `[\U00000000-\U0010FFFF]`
			}
			return c.addRule("dot", rule)
		}
		joinSeq := func() litOrRule {
			var ret []litOrRule
			lit := ""
			flush := func() {
				if lit != "" {
					ret = append(ret, litOrRule{lit, true})
					lit = ""
				}
			}
			for _, it := range seq {
				if it.literal {
					lit += it.s
				} else {
					flush()
					ret = append(ret, it)
				}
			}
			flush()
			parts := make([]string, len(ret))
			for k, it := range ret {
				parts[k] = toRule(it)
			}
			return litOrRule{strings.Join(parts, " "), false}
		}

		for i < length {
			ch := sub[i]
			switch {
			case ch == '.':
				seq = append(seq, litOrRule{getDot(), false})
				i++
			case ch == '(':
				i++
				if i < length && sub[i] == '?' {
					if i+1 < length && sub[i+1] == ':' {
						i += 2
					} else {
						panic(unsupportedPattern{"unsupported group syntax"})
					}
				}
				depth++
				if depth > maxPatternDepth {
					panic(unsupportedPattern{"pattern nesting too deep"})
				}
				seq = append(seq, litOrRule{"(" + toRule(transform()) + ")", false})
			case ch == ')':
				i++
				if depth == 0 {
					panic(invalidPattern{"unbalanced parentheses"})
				}
				depth--
				return joinSeq()
			case ch == '^' || ch == '$':
				panic(unsupportedPattern{"anchor inside the pattern"})
			case ch == '[':
				sb := "["
				i++
				for i < length && sub[i] != ']' {
					if sub[i] == '\\' {
						n := gbnfEscapeLength(sub, i)
						if n == 0 {
							panic(unsupportedPattern{"unsupported escape in character class: " + sub[i:min(i+2, length)]})
						}
						sb += sub[i : i+n]
						i += n
					} else {
						sb += string(sub[i])
						i++
					}
				}
				if i >= length {
					panic(invalidPattern{"unterminated character class"})
				}
				sb += "]"
				i++
				seq = append(seq, litOrRule{sb, false})
			case ch == '|':
				seq = append(seq, litOrRule{"|", false})
				i++
			case ch == '*' || ch == '+' || ch == '?':
				if len(seq) == 0 {
					panic(invalidPattern{"nothing to repeat"})
				}
				seq[len(seq)-1] = litOrRule{toRule(seq[len(seq)-1]) + string(ch), false}
				i++
			case ch == '{':
				cb := "{"
				i++
				for i < length && sub[i] != '}' {
					cb += string(sub[i])
					i++
				}
				if i >= length {
					panic(unsupportedPattern{"unterminated curly brackets"})
				}
				cb += "}"
				i++
				nums := strings.Split(cb[1:len(cb)-1], ",")
				minT, maxT := 0, maxInt
				if len(nums) != 1 && len(nums) != 2 {
					panic(unsupportedPattern{"wrong number of values in curly brackets"})
				}
				atoi := func(s string) int {
					n, err := cppStoi(s)
					if err != nil {
						panic(unsupportedPattern{"invalid number in curly brackets"})
					}
					return n
				}
				if len(nums) == 1 {
					minT = atoi(nums[0])
					maxT = minT
				} else {
					if nums[0] != "" {
						minT = atoi(nums[0])
					}
					if nums[1] != "" {
						maxT = atoi(nums[1])
					}
				}
				if len(seq) == 0 {
					panic(invalidPattern{"nothing to repeat"})
				}
				last := &seq[len(seq)-1]
				subS, isLit := last.s, last.literal
				if !isLit {
					id, ok := subRuleIDs[subS]
					if !ok {
						subOrder++
						id = c.addRule(name+"-"+strconv.Itoa(subOrder), subS)
						subRuleIDs[subS] = id
					}
					subS = id
				}
				item := subS
				if isLit {
					item = `"` + subS + `"`
				}
				*last = litOrRule{buildRepetition(item, minT, maxT, ""), false}
			default:
				lit := ""
				for i < length {
					if sub[i] == '\\' {
						if i == length-1 {
							panic(invalidPattern{"trailing backslash"})
						}
						next := sub[i+1]
						if escapedInRegexpsNotLiterals(next) {
							i++
							lit += string(sub[i])
							i++
						} else {
							n := gbnfEscapeLength(sub, i)
							if n == 0 {
								panic(unsupportedPattern{"unsupported escape: " + sub[i:i+2]})
							}
							lit += sub[i : i+n]
							i += n
						}
					} else if sub[i] == '"' {
						lit += `\"`
						i++
					} else if !isNonLiteral(sub[i]) && (i == length-1 || lit == "" || sub[i+1] == '.' || !isNonLiteral(sub[i+1])) {
						lit += string(sub[i])
						i++
					} else {
						break
					}
				}
				if lit == "" {
					panic(unsupportedPattern{"unsupported character: " + string(ch)})
				}
				seq = append(seq, litOrRule{lit, true})
			}
		}
		return joinSeq()
	}

	rule := toRule(transform())
	if depth != 0 {
		panic(invalidPattern{"unbalanced parentheses"})
	}
	return c.addRule(name, `"\"" (`+rule+`) "\""`)
}

// cppStoi is std::stoi: leading whitespace, an optional sign, digits, and
// whatever follows ignored; no digits or out of range is an error.
func cppStoi(s string) (int, error) {
	s = strings.TrimLeft(s, " \t\n\v\f\r")
	j := 0
	if j < len(s) && (s[j] == '+' || s[j] == '-') {
		j++
	}
	k := j
	for k < len(s) && s[k] >= '0' && s[k] <= '9' {
		k++
	}
	if k == j {
		return 0, errors.New("no digits")
	}
	n, err := strconv.ParseInt(s[:k], 10, 32)
	return int(n), err
}

// notStrings is a rule for a JSON string that is none of strs, built from a
// trie of their code points.
func (c *converter) notStrings(strs []string) string {
	type tnode struct {
		children map[rune]int
		pattern  int
	}
	nodes := []tnode{{children: map[rune]int{}, pattern: -1}}
	nPatterns := 0
	for _, w := range strs {
		cur := 0
		for _, r := range w {
			if r == utf8.RuneError {
				break
			}
			child, ok := nodes[cur].children[r]
			if !ok {
				nodes = append(nodes, tnode{children: map[rune]int{}, pattern: -1})
				child = len(nodes) - 1
				nodes[cur].children[r] = child
			}
			cur = child
		}
		if nodes[cur].pattern < 0 {
			nodes[cur].pattern = nPatterns
			nPatterns++
		}
	}
	charRule := c.addPrimitive("char", primitiveRules["char"])
	var out strings.Builder
	out.WriteString(`["] ( `)
	var visit func(idx int)
	visit = func(idx int) {
		node := nodes[idx]
		keys := make([]rune, 0, len(node.children))
		for r := range node.children {
			keys = append(keys, r)
		}
		sort.Slice(keys, func(a, b int) bool { return keys[a] < keys[b] })
		rejects := ""
		for k, r := range keys {
			ch := string(r)
			rejects += ch
			if k > 0 {
				out.WriteString(" | ")
			}
			out.WriteString("[" + ch + "]")
			child := node.children[r]
			if len(nodes[child].children) > 0 {
				out.WriteString(" (")
				visit(child)
				out.WriteString(")")
			} else {
				out.WriteString(" " + charRule + "+")
			}
		}
		if len(keys) > 0 {
			out.WriteString(` | [^"` + rejects + `] ` + charRule + "*")
		}
	}
	visit(0)
	out.WriteString(" )")
	if nodes[0].pattern < 0 {
		out.WriteString("?")
	}
	out.WriteString(` ["]`)
	return out.String()
}

var nonAlnum = regexp.MustCompile(`[^a-zA-Z0-9-]+`)

func (c *converter) resolveRef(n *snode) string {
	frag := n.ref
	if i := strings.IndexByte(frag, '#'); i >= 0 {
		frag = frag[i+1:]
	}
	refName := "ref" + nonAlnum.ReplaceAllString(frag, "-")
	if _, ok := c.rules[refName]; !ok && !c.resolving[n.ref] {
		if n.target == nil {
			c.errs = append(c.errs, "Unresolved $ref "+n.ref)
			return ""
		}
		c.resolving[n.ref] = true
		refName = c.visit(n.target, refName)
		delete(c.resolving, n.ref)
	}
	return refName
}

type propRef struct {
	name   string
	schema *snode
}

func (c *converter) buildObjectRule(props []propRef, required map[string]bool, name string, additional *snode) string {
	var requiredProps, optionalProps, propNames []string
	kv := map[string]string{}
	dash := "-"
	if name == "" {
		dash = ""
	}
	for _, p := range props {
		ruleName := c.visit(p.schema, name+dash+p.name)
		kv[p.name] = c.addRule(name+dash+p.name+"-kv", formatLiteral(jv{kind: jString, s: p.name}.dump())+` space ":" space `+ruleName)
		if required[p.name] {
			requiredProps = append(requiredProps, p.name)
		} else {
			optionalProps = append(optionalProps, p.name)
		}
		propNames = append(propNames, p.name)
	}
	if additional != nil {
		subName := name + dash + "additional"
		var valueRule string
		if additional.kind != kAny {
			valueRule = c.visit(additional, subName+"-value")
		} else {
			valueRule = c.addPrimitive("value", primitiveRules["value"])
		}
		var keyRule string
		if len(propNames) == 0 {
			keyRule = c.addPrimitive("string", primitiveRules["string"])
		} else {
			keyRule = c.addRule(subName+"-k", c.notStrings(propNames))
		}
		kv["*"] = c.addRule(subName+"-kv", keyRule+` ":" space `+valueRule)
		optionalProps = append(optionalProps, "*")
	}
	if len(requiredProps) == 0 && len(optionalProps) == 0 {
		return `"{" space "}"`
	}
	rule := `"{" space `
	for i, p := range requiredProps {
		if i > 0 {
			rule += ` "," space `
		}
		rule += kv[p]
	}
	if len(optionalProps) > 0 {
		rule += " ("
		if len(requiredProps) > 0 {
			rule += ` "," space ( `
		}
		var recursive func(ks []string, firstOptional bool) string
		recursive = func(ks []string, firstOptional bool) string {
			if len(ks) == 0 {
				return ""
			}
			k := ks[0]
			kvName := kv[k]
			commaRef := `( "," space ` + kvName + ` )`
			var res string
			if firstOptional {
				if k == "*" {
					res = commaRef + "*"
				} else {
					res = commaRef + "?"
				}
			} else {
				res = kvName
				if k == "*" {
					res += " " + commaRef + "*"
				}
			}
			if len(ks) > 1 {
				res += " " + c.addRule(name+dash+k+"-rest", recursive(ks[1:], true))
			}
			return res
		}
		for i := range optionalProps {
			if i > 0 {
				rule += " | "
			}
			rule += recursive(optionalProps[i:], false)
		}
		if len(requiredProps) > 0 {
			rule += " )"
		}
		rule += " )?"
	}
	return rule + ` space "}"`
}

func (c *converter) visitPrimitive(ruleName, typ string) string {
	name := typ
	if ruleName == "root" {
		name = "root"
	}
	return c.addPrimitive(name, primitiveRules[typ])
}

func constantRule(v jv) string { return formatLiteral(v.dump()) }

func (c *converter) visitAllOf(n *snode, name, ruleName string) string {
	required := map[string]bool{}
	var props []propRef
	enumCounts := map[string]int{}
	var add func(comp *snode, isRequired bool)
	add = func(comp *snode, isRequired bool) {
		switch comp.kind {
		case kRef:
			if comp.target != nil {
				add(comp.target, isRequired)
			}
		case kObject:
			for _, p := range comp.props {
				props = append(props, propRef{p.name, p.schema})
				if isRequired {
					required[p.name] = true
				}
			}
		case kEnum:
			for _, v := range comp.values {
				enumCounts[constantRule(v)]++
			}
		}
	}
	for _, child := range n.children {
		if child.kind == kAnyOf {
			for _, alt := range child.children {
				add(alt, false)
			}
		} else {
			add(child, true)
		}
	}
	if len(enumCounts) > 0 {
		var inter []string
		for lit, n2 := range enumCounts {
			if n2 == len(n.children) {
				inter = append(inter, lit)
			}
		}
		sort.Strings(inter)
		if len(inter) > 0 {
			return c.addRule(ruleName, "("+strings.Join(inter, " | ")+")")
		}
	}
	return c.addRule(ruleName, c.buildObjectRule(props, required, name, nil))
}

func (c *converter) visit(n *snode, name string) string {
	ruleName := name
	switch {
	case isReservedName(name):
		ruleName = name + "-"
	case name == "":
		ruleName = "root"
	}
	subName := name
	if name != "" {
		subName += "-"
	}
	switch n.kind {
	case kRef:
		return c.addRule(ruleName, c.resolveRef(n))
	case kAnyOf:
		return c.addRule(ruleName, c.unionRule(name, n.children))
	case kAllOf:
		return c.visitAllOf(n, name, ruleName)
	case kConst:
		return c.addRule(ruleName, constantRule(n.value))
	case kEnum:
		vals := make([]string, len(n.values))
		for i, v := range n.values {
			vals[i] = constantRule(v)
		}
		return c.addRule(ruleName, "("+strings.Join(vals, " | ")+")")
	case kObject:
		if len(n.props) == 0 && n.additional != nil && n.additional.kind == kAny {
			return c.addRule(ruleName, c.addPrimitive("object", primitiveRules["object"]))
		}
		var props []propRef
		required := map[string]bool{}
		for _, p := range n.props {
			props = append(props, propRef{p.name, p.schema})
			if p.required {
				required[p.name] = true
			}
		}
		return c.addRule(ruleName, c.buildObjectRule(props, required, name, n.additional))
	case kTuple:
		rule := `"[" space `
		for i, it := range n.items {
			if i > 0 {
				rule += ` "," space `
			}
			rule += c.visit(it, subName+"tuple-"+strconv.Itoa(i))
		}
		return c.addRule(ruleName, rule+` space "]"`)
	case kArray:
		if n.item.kind == kAny && n.minItems == 0 && n.maxItems < 0 {
			return c.visitPrimitive(ruleName, "array")
		}
		itemRule := c.visit(n.item, subName+"item")
		mx := n.maxItems
		if mx < 0 {
			mx = maxInt
		}
		return c.addRule(ruleName, `"[" space `+buildRepetition(itemRule, n.minItems, mx, `"," space`)+` space "]"`)
	case kString:
		if n.pattern != "" {
			return c.visitPattern(n.pattern, ruleName)
		}
		if n.format == fmtUUID {
			return c.visitPrimitive(ruleName, "uuid")
		}
		if n.format != fmtNone {
			prim := map[strFormat]string{fmtDate: "date", fmtTime: "time", fmtDateTime: "date-time"}[n.format] + "-string"
			return c.addRule(ruleName, c.addPrimitive(prim, stringFormatRules[prim]))
		}
		if n.minLength > 0 || n.maxLength >= 0 {
			charRule := c.addPrimitive("char", primitiveRules["char"])
			mx := n.maxLength
			if mx < 0 {
				mx = maxInt
			}
			return c.addRule(ruleName, `"\"" `+buildRepetition(charRule, n.minLength, mx, "")+` "\""`)
		}
		return c.visitPrimitive(ruleName, "string")
	case kInteger:
		if n.minimum == math.MinInt64 && n.maximum == math.MaxInt64 {
			return c.visitPrimitive(ruleName, "integer")
		}
		var out strings.Builder
		out.WriteString("(")
		buildMinMaxInt(n.minimum, n.maximum, &out, 16, true)
		out.WriteString(")")
		return c.addRule(ruleName, out.String())
	case kNumber:
		return c.visitPrimitive(ruleName, "number")
	case kBoolean:
		return c.visitPrimitive(ruleName, "boolean")
	case kNull:
		return c.visitPrimitive(ruleName, "null")
	case kAny:
		return c.addRule(ruleName, c.addPrimitive("value", primitiveRules["value"]))
	}
	return ""
}

func (c *converter) check() error {
	if len(c.errs) > 0 {
		return errors.New("JSON schema conversion failed:\n" + strings.Join(c.errs, "\n"))
	}
	return nil
}

// format renders the rules sorted by name, as std::map iterates them.
func (c *converter) format() string {
	names := make([]string, 0, len(c.rules))
	for k := range c.rules {
		names = append(names, k)
	}
	sort.Strings(names)
	var sb strings.Builder
	for _, k := range names {
		sb.WriteString(k + " ::= " + c.rules[k] + "\n")
	}
	return sb.String()
}

// SchemaToGBNF compiles a JSON Schema to a GBNF grammar with root "root", as
// json_schema_to_grammar does. Warnings name the parts that were accepted
// more loosely than the schema says (an unsupported regex becomes any
// string).
func SchemaToGBNF(schema []byte) (gbnf string, warnings []string, err error) {
	v, err := parseJSON(schema)
	if err != nil {
		return "", nil, errors.New("JSON schema conversion failed:\n" + err.Error())
	}
	root, err := parseSchema(v)
	if err != nil {
		return "", nil, errors.New("JSON schema conversion failed:\n" + err.Error())
	}
	c := newConverter(false)
	c.visit(root, "")
	if err := c.check(); err != nil {
		return "", c.warnings, err
	}
	return c.format(), c.warnings, nil
}
