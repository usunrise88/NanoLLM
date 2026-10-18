#!/usr/bin/env python3
# Extract the json_schema_to_grammar test cases from llama.cpp.
#
#     tools/grammarcases.py llama.cpp/tests/test-json-schema-to-grammar.cpp OUT.json
#
# Each test({SUCCESS|FAILURE, name, raw schema, raw grammar}) in test_all
# becomes {"name", "ok", "schema", "grammar"}; internal/grammar must produce
# the same grammar (after llama.cpp's own whitespace trim) for every case.
import json
import re
import sys

src = open(sys.argv[1]).read()
# Only the cases of test_all, the shared table every converter is held to.
body = src[src.index("static void test_all("):]
body = body[: body.index("\nint main(")]
pat = re.compile(
    r'test\(\{\s*(SUCCESS|FAILURE),\s*"((?:[^"\\]|\\.)*)",\s*(?:R"""\((.*?)\)"""|"((?:[^"\\]|\\.)*)"),\s*(?:R"""\((.*?)\)"""|"")\s*,?\s*\}\);',
    re.S,
)
cases = [
    {"name": json.loads('"%s"' % m[1]), "ok": m[0] == "SUCCESS",
     "schema": m[2] if m[2] else json.loads('"%s"' % m[3]), "grammar": m[4]}
    for m in pat.findall(body)
]
json.dump(cases, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
print(len(cases), "cases")
