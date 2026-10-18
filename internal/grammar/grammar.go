// Package grammar builds the GBNF grammars that constrain sampling (SPEC §5):
// JSON for response_format, JSON Schema compiled the way llama.cpp's
// json_schema_to_grammar compiles it, and the Gemma 4 tool-call syntax for
// tool_choice.
//
// A grammar is enforced by llama.cpp's grammar sampler, so output is valid by
// construction; nothing here validates output after the fact.
package grammar

// Spec is a grammar as the sampler takes it. With Triggers set the grammar is
// lazy: generation is free until one of the trigger strings appears, and the
// grammar applies from there (tool_choice auto).
type Spec struct {
	GBNF     string
	Root     string
	Triggers []string
	Warnings []string
}

// jsonGBNF is llama.cpp's grammars/json.gbnf with object as the root.
const jsonGBNF = `root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

ws ::= | " " | "\n" [ \t]{0,20}
`

// JSONObject is the grammar for response_format json_object: any JSON object.
func JSONObject() Spec { return Spec{GBNF: jsonGBNF, Root: "root"} }

// FromSchema compiles a JSON Schema the way llama.cpp's json_schema_to_grammar
// does, for response_format json_schema. Warnings name what the grammar
// accepts more loosely than the schema says.
func FromSchema(schema []byte) (Spec, error) {
	g, warnings, err := SchemaToGBNF(schema)
	if err != nil {
		return Spec{}, err
	}
	return Spec{GBNF: g, Root: "root", Warnings: warnings}, nil
}
