#!/usr/bin/env python3
"""Render chat-template fixtures with the model's own Jinja template.

    tools/chatfixtures/gen.py MODEL.gguf CASES.json OUT.json

The template is read from the GGUF's tokenizer.chat_template and rendered with
jinja2 configured the way transformers' apply_chat_template configures it
(trim_blocks, lstrip_blocks, loop controls, a sandbox). internal/chat renders
the same cases in Go and must match byte for byte.

Cases are OpenAI-shaped: tool-call arguments are JSON strings. They are handed
to the template as mappings when they parse as a JSON object, which is the
form the template renders exactly; internal/chat does the same conversion.

Each case yields two renders: "template", the messages as they are, and
"prompt", after the tool-result folding and generation-prompt fix that
llama.cpp applies for Gemma 4 (common/parsers/gemma4.cpp), which is what a
request is served with.
"""
import json
import struct
import sys

from jinja2.sandbox import ImmutableSandboxedEnvironment


def chat_template(path):
    with open(path, "rb") as f:
        magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
        assert magic == b"GGUF", "not a GGUF file"

        def string():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode()

        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def skip(t):
            if t == 8:
                string()
            elif t == 9:
                et, n = struct.unpack("<IQ", f.read(12))
                for _ in range(n):
                    skip(et)
            else:
                f.read(sizes[t])

        for _ in range(n_kv):
            key = string()
            (t,) = struct.unpack("<I", f.read(4))
            if key == "tokenizer.chat_template":
                return string()
            skip(t)
    raise SystemExit("no tokenizer.chat_template in " + path)


def convert(messages):
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                c = json.loads(json.dumps(c))
                args = c["function"].get("arguments")
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                        if isinstance(parsed, dict):
                            c["function"]["arguments"] = parsed
                    except ValueError:
                        pass
                calls.append(c)
            m["tool_calls"] = calls
        out.append(m)
    return out


def fold_tool_responses(messages):
    """Port of llama.cpp's workaround::convert_tool_responses_gemma4, except
    that results are matched to calls by id first."""
    out, i = [], 0
    while i < len(messages):
        m = messages[i]
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            out.append(m)
            i += 1
            continue
        calls = m["tool_calls"]
        turn = {"role": "assistant", "tool_calls": calls}
        if isinstance(m.get("reasoning_content"), str):
            turn["reasoning_content"] = m["reasoning_content"]
        i += 1
        responses = []
        while i < len(messages) and messages[i].get("role") == "tool":
            t = messages[i]
            content = t.get("content")
            response = None
            if isinstance(content, str):
                try:
                    response = json.loads(content)
                except ValueError:
                    response = content
            elif "content" in t:
                response = content
            # By call id first: llama.cpp matches by position only, which
            # names results wrongly when they arrive in another order.
            name = ""
            for call in calls:
                if call.get("id") and call.get("id") == t.get("tool_call_id"):
                    name = call.get("function", {}).get("name", "")
            if not name and len(responses) < len(calls):
                name = calls[len(responses)].get("function", {}).get("name", "")
            if not name:
                name = t.get("tool_call_id", "")
            responses.append({"name": name, "response": response})
            i += 1
        if responses:
            turn["tool_responses"] = responses
        if i < len(messages) and messages[i].get("role") == "assistant":
            nxt = messages[i]
            if not nxt.get("tool_calls") and nxt.get("content"):
                turn["content"] = nxt["content"]
                i += 1
        out.append(turn)
    return out


def main():
    model, cases_path, out_path = sys.argv[1:4]
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])
    tpl = env.from_string(chat_template(model))
    cases = json.load(open(cases_path))
    out = {}
    for c in cases:
        agp = c.get("add_generation_prompt", True)

        def render(messages):
            return tpl.render(
                messages=messages,
                tools=c.get("tools") or None,
                enable_thinking=c.get("enable_thinking", False),
                add_generation_prompt=agp,
                bos_token="<bos>",
                eos_token="<eos>",
            )

        msgs = convert(c["messages"])
        prompt = render(fold_tool_responses(msgs))
        if agp and prompt.endswith("<turn|>\n"):
            prompt += "<|turn>model\n"
        out[c["name"]] = {"template": render(msgs), "prompt": prompt}
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
