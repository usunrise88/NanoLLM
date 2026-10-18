import json

import openai
import pytest


def test_models(client):
    models = client.models.list()
    assert models.data, "the catalog lists at least one model"
    assert client.models.retrieve(models.data[0].id).id == models.data[0].id


def test_chat_basic(client, model):
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Say hello."}], max_tokens=32)
    c = r.choices[0]
    assert c.message.role == "assistant"
    assert isinstance(c.message.content, str) and c.message.content
    assert c.finish_reason in ("stop", "length")
    assert r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens
    assert r.id.startswith("chatcmpl-")


def test_chat_echo(client, model, fake_only):
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Привет, мир [[fake:echo]]"}])
    assert r.choices[0].message.content == "Привет, мир"
    assert r.choices[0].finish_reason == "stop"


def test_stream_matches_buffered(client, model, fake_only):
    msgs = [{"role": "user", "content": "[[fake:long 120]]"}]
    full = client.chat.completions.create(model=model, messages=msgs).choices[0].message.content
    chunks = list(client.chat.completions.create(model=model, messages=msgs, stream=True,
                                                 stream_options={"include_usage": True}))
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == full
    assert chunks[0].choices[0].delta.role == "assistant"
    finishes = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]
    assert finishes == ["stop"]
    assert chunks[-1].usage is not None and chunks[-1].usage.completion_tokens == 120


def test_max_tokens_length(client, model):
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Count to one thousand. [[fake:loop]]"}],
                                       max_tokens=5)
    assert r.choices[0].finish_reason == "length"
    assert r.usage.completion_tokens == 5


def test_stop_sequence(client, model, fake_only):
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "alpha beta gamma [[fake:echo]]"}],
                                       stop=[" gamma"])
    assert r.choices[0].message.content == "alpha beta"


def test_tool_call(client, model, fake_only):
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather",
                                                "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                               "required": ["city"]}}}]
    r = client.chat.completions.create(model=model, tools=tools,
                                       messages=[{"role": "user", "content": '[[fake:tool get_weather {city:<|"|>Paris<|"|>}]]'}])
    c = r.choices[0]
    assert c.finish_reason == "tool_calls"
    call = c.message.tool_calls[0]
    assert call.type == "function" and call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"city": "Paris"}
    # The result goes back in, and the conversation continues.
    r2 = client.chat.completions.create(model=model, tools=tools, messages=[
        {"role": "user", "content": "Weather in Paris? [[fake:echo]]"},
        {"role": "assistant", "tool_calls": [call.model_dump()]},
        {"role": "tool", "tool_call_id": call.id, "content": '{"temp": 21}'},
    ])
    assert r2.choices[0].finish_reason in ("stop", "length")


def test_tool_call_stream(client, model, fake_only):
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    chunks = list(client.chat.completions.create(model=model, tools=tools, stream=True,
                                                 messages=[{"role": "user", "content": "[[fake:tool f {a:1}]]"}]))
    calls = [tc for c in chunks if c.choices for tc in (c.choices[0].delta.tool_calls or [])]
    assert len(calls) == 1 and calls[0].index == 0 and calls[0].id
    assert json.loads(calls[0].function.arguments) == {"a": 1}
    assert [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason] == ["tool_calls"]


def test_reasoning_content(client, model, fake_only):
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "[[fake:think]]"}])
    msg = r.choices[0].message
    assert msg.content == "Here is a short answer."
    assert getattr(msg, "reasoning_content", None) or msg.model_extra.get("reasoning_content")
    assert r.usage.completion_tokens_details.reasoning_tokens > 0


def test_logprobs(client, model):
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Hi [[fake:echo]]"}],
                                       logprobs=True, top_logprobs=2, max_tokens=8)
    lp = r.choices[0].logprobs.content
    assert lp and all(e.logprob <= 0 for e in lp)
    assert all(len(e.top_logprobs) == 2 for e in lp)
    assert all(bytes(e.bytes).decode("utf-8", "replace") for e in lp)


def test_json_object(client, model, fake_only):
    r = client.chat.completions.create(model=model, response_format={"type": "json_object"},
                                       messages=[{"role": "user", "content": "[[fake:json]]"}])
    assert json.loads(r.choices[0].message.content)["ok"] is True


def test_errors(client, model):
    with pytest.raises(openai.NotFoundError) as e:
        client.chat.completions.create(model="no-such-model", messages=[{"role": "user", "content": "x"}])
    assert e.value.body["type"] == "invalid_request_error"
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(model=model, n=2, messages=[{"role": "user", "content": "x"}])
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(model=model, messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}])
    with pytest.raises(openai.NotFoundError):
        list(client.chat.completions.create(model="no-such-model", stream=True, messages=[{"role": "user", "content": "x"}]))


def test_completions(client, model):
    # Greedy, so the streamed and the buffered answer must be the same text.
    r = client.completions.create(model=model, prompt="Once upon a time [[fake:loop]]", max_tokens=6, temperature=0)
    assert r.object == "text_completion"
    assert r.choices[0].finish_reason == "length"
    assert r.usage.completion_tokens == 6
    s = list(client.completions.create(model=model, prompt="Once upon a time [[fake:loop]]", max_tokens=6, temperature=0, stream=True))
    assert "".join(c.choices[0].text for c in s if c.choices) == r.choices[0].text


def test_prompt_cache(client, model, fake_only):
    msgs = [{"role": "system", "content": "A long system prompt. " * 20}, {"role": "user", "content": "one [[fake:echo]]"}]
    client.chat.completions.create(model=model, messages=msgs)
    r = client.chat.completions.create(model=model, messages=msgs + [
        {"role": "assistant", "content": "one"}, {"role": "user", "content": "two [[fake:echo]]"}])
    assert r.usage.prompt_tokens_details.cached_tokens > 100
