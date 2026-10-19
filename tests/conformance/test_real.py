"""Conformance against a real model (NANOLLM_BASE_URL): what the scripted fake
cannot show. Prompt cache hits, sessions, a cancelled stream, several clients
at once, tool calls and JSON output with actual weights. Skipped on the fake."""
import json
import threading

import pytest


@pytest.fixture(autouse=True)
def real_only(server):
    if server["fake"]:
        pytest.skip("needs a real model")


SYSTEM = "Ты — помощник. Отвечай коротко и по делу. " * 12


def chat(client, model, messages, **kw):
    kw.setdefault("max_tokens", 24)
    kw.setdefault("temperature", 0)
    return client.chat.completions.create(model=model, messages=messages, **kw)


def cached(r):
    d = r.usage.prompt_tokens_details
    return d.cached_tokens if d else 0


def test_prompt_cache_hit(client, model):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Назови столицу Франции."}]
    first = chat(client, model, msgs)
    second = chat(client, model, msgs + [
        {"role": "assistant", "content": first.choices[0].message.content},
        {"role": "user", "content": "А Германии?"}])
    assert cached(second) > 50, second.usage


def test_session_keeps_the_cache(client, model):
    msgs = [{"role": "system", "content": SYSTEM + " (session)"}, {"role": "user", "content": "Сколько будет 2+2?"}]
    first = chat(client, model, msgs, extra_body={"session_id": "conformance-1"})
    second = chat(client, model, msgs + [
        {"role": "assistant", "content": first.choices[0].message.content},
        {"role": "user", "content": "А 3+3?"}], extra_body={"session_id": "conformance-1"})
    assert cached(second) > 50, second.usage


def test_cancelled_stream_keeps_the_prefix(client, model):
    msgs = [{"role": "system", "content": SYSTEM + " (cancel)"},
            {"role": "user", "content": "Расскажи длинную историю про дракона."}]
    stream = client.chat.completions.create(model=model, messages=msgs, max_tokens=400, temperature=0, stream=True)
    got = 0
    for chunk in stream:
        got += 1
        if got >= 5:
            break
    stream.close()  # the client goes away mid-answer
    again = chat(client, model, msgs, max_tokens=8)
    assert cached(again) > 50, again.usage


def test_stream_matches_buffered_greedy(client, model):
    msgs = [{"role": "user", "content": "Перечисли три цвета радуги через запятую."}]
    extra = {"cache_prompt": False}
    whole = chat(client, model, msgs, max_tokens=24, extra_body=extra).choices[0].message.content
    parts = client.chat.completions.create(model=model, messages=msgs, max_tokens=24, temperature=0, stream=True, extra_body=extra)
    streamed = "".join(c.choices[0].delta.content or "" for c in parts if c.choices)
    assert streamed == whole


def test_parallel_clients(client, model):
    results, errors = [], []

    def one(i):
        try:
            r = chat(client, model, [{"role": "user", "content": "Назови число %d словами." % (i + 3)}], max_tokens=16)
            results.append(r.choices[0].finish_reason)
        except Exception as e:  # collected for the assertion
            errors.append(e)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=600)
    assert not errors, errors
    assert len(results) == 4 and all(r in ("stop", "length") for r in results)


def test_tool_call_with_a_real_model(client, model):
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Погода в городе",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    r = chat(client, model, [{"role": "user", "content": "Какая сейчас погода в Париже? Используй инструмент."}],
             tools=tools, max_tokens=80)
    calls = r.choices[0].message.tool_calls
    assert calls, r.choices[0]
    assert calls[0].function.name == "get_weather"
    assert "city" in json.loads(calls[0].function.arguments)


def test_json_object_parses(client, model):
    r = chat(client, model, [{"role": "user", "content": "Верни JSON с полями name и age для вымышленного человека."}],
             response_format={"type": "json_object"}, max_tokens=80)
    assert isinstance(json.loads(r.choices[0].message.content), dict)
