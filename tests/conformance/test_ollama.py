"""The official `ollama` Python client against the ollama dialect.

Skipped as a whole when the package is not installed (pip install ollama), and
when a running server named by NANOLLM_BASE_URL does not mount the dialect.
Tests that depend on the fake backend's scripted answers skip against a real
model, as in the rest of the suite.
"""
import asyncio
import json
import os
import urllib.error
import urllib.request

import pytest

ollama = pytest.importorskip("ollama")

ASK = [{"role": "user", "content": "Say hello."}]


def echo(text):
    """A conversation that the fake backend answers with `text`."""
    return [{"role": "user", "content": text + " [[fake:echo]]"}]


def _headers():
    key = os.environ.get("NANOLLM_API_KEY")
    return {"Authorization": "Bearer " + key} if key else {}


@pytest.fixture(scope="session")
def ollama_host(server):
    req = urllib.request.Request(server["base"] + "/api/version", headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            json.load(r)
    except (urllib.error.URLError, ValueError):
        pytest.skip("the server does not mount the ollama dialect")
    return server["base"]


@pytest.fixture(scope="session")
def oc(ollama_host):
    return ollama.Client(host=ollama_host, headers=_headers())


@pytest.fixture(scope="session")
def tag(oc):
    """The model's name as /api/tags gives it, which is what clients send back."""
    models = oc.list().models
    assert models, "the server lists a local model"
    return models[0].model


def test_root_and_version(ollama_host):
    req = urllib.request.Request(ollama_host + "/", headers=_headers())
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.read() == b"Ollama is running"
    req = urllib.request.Request(ollama_host + "/api/version", headers=_headers())
    with urllib.request.urlopen(req, timeout=10) as r:
        assert json.load(r)["version"][0] == "0"


def test_list(oc):
    models = oc.list().models
    assert models
    m = models[0]
    assert m.model.endswith(":latest") and m.size > 0 and m.modified_at is not None
    assert len(m.digest) == 64
    assert m.details.format == "gguf" and m.details.quantization_level == "Q4_0"
    assert m.details.family and m.details.parameter_size


def test_show(oc, tag):
    r = oc.show(tag)
    assert "completion" in r.capabilities and "tools" in r.capabilities
    assert r.details.family == r.modelinfo["general.architecture"]
    assert r.modelfile.startswith("#") and "FROM " in r.modelfile


def test_show_unknown(oc):
    with pytest.raises(ollama.ResponseError) as e:
        oc.show("llama3.2")
    assert e.value.status_code == 404 and "not found" in e.value.error


def test_chat(oc, tag, fake_only):
    r = oc.chat(model=tag, messages=echo("Привет, мир"))
    assert r.message.role == "assistant" and r.message.content == "Привет, мир"
    assert r.done and r.done_reason == "stop" and r.model == tag
    assert r.total_duration > 0 and r.prompt_eval_count > 0 and r.eval_count > 0
    assert r.load_duration is not None and r.eval_duration is not None and r.prompt_eval_duration is not None


def test_chat_basic_on_any_model(oc, tag):
    r = oc.chat(model=tag, messages=ASK, options={"num_predict": 32})
    assert isinstance(r.message.content, str) and r.message.content
    assert r.done_reason in ("stop", "length")


def test_chat_stream_matches_buffered(oc, tag, fake_only):
    msgs = [{"role": "user", "content": "[[fake:long 120]]"}]
    full = oc.chat(model=tag, messages=msgs).message.content
    parts = list(oc.chat(model=tag, messages=msgs, stream=True))
    assert len(parts) > 2
    assert "".join(p.message.content for p in parts) == full
    assert [p.done for p in parts] == [False] * (len(parts) - 1) + [True]
    last = parts[-1]
    assert last.done_reason == "stop" and last.eval_count and last.total_duration > 0


def test_generate(oc, tag, fake_only):
    r = oc.generate(model=tag, prompt="hello [[fake:echo]]", system="be brief")
    assert r.response == "hello" and r.done and r.done_reason == "stop"
    parts = list(oc.generate(model=tag, prompt="[[fake:long 60]]", stream=True))
    assert "".join(p.response for p in parts) == oc.generate(model=tag, prompt="[[fake:long 60]]").response
    assert parts[-1].done and parts[-1].eval_count


def test_model_names(oc, tag, fake_only):
    for name in (tag, tag.removesuffix(":latest"), "gemma4", "gemma4:e4b"):
        r = oc.chat(model=name, messages=echo("hi"))
        assert r.message.content == "hi" and r.model == name


def test_num_predict_and_stop(oc, tag, fake_only):
    r = oc.chat(model=tag, messages=[{"role": "user", "content": "[[fake:loop]]"}], options={"num_predict": 5})
    assert r.done_reason == "length" and r.eval_count == 5
    r = oc.chat(model=tag, messages=echo("alpha beta gamma"), options={"stop": [" gamma"]})
    assert r.message.content == "alpha beta"


def test_options_are_accepted(oc, tag):
    r = oc.chat(model=tag, messages=ASK, options={
        "temperature": 0, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.1, "seed": 1,
        "num_predict": 8, "num_ctx": 4096, "num_thread": 4, "mirostat": 0,
    }, keep_alive="5m")
    assert r.done


def test_format_json(oc, tag, fake_only):
    r = oc.chat(model=tag, messages=[{"role": "user", "content": "[[fake:json]]"}], format="json")
    assert isinstance(json.loads(r.message.content), dict)


def test_format_schema(oc, tag, fake_only):
    pydantic = pytest.importorskip("pydantic")

    class Answer(pydantic.BaseModel):
        ok: bool

    r = oc.chat(model=tag, messages=[{"role": "user", "content": "[[fake:json]]"}], format=Answer.model_json_schema())
    assert Answer.model_validate_json(r.message.content).ok


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Weather in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
ASK_WEATHER = [{"role": "user", "content": '[[fake:tool get_weather {city:<|"|>Paris<|"|>}]]'}]


def test_tool_call_and_result(oc, tag, fake_only):
    r = oc.chat(model=tag, messages=ASK_WEATHER, tools=[WEATHER])
    calls = r.message.tool_calls
    assert len(calls) == 1 and calls[0].function.name == "get_weather"
    assert calls[0].function.arguments == {"city": "Paris"}
    # The result goes back in, as the Ollama documentation does it, and the conversation goes on.
    msgs = [{"role": "user", "content": "Weather in Paris? [[fake:echo]]"}, r.message,
            {"role": "tool", "tool_name": "get_weather", "content": json.dumps({"temp": 21})}]
    r2 = oc.chat(model=tag, messages=msgs, tools=[WEATHER])
    assert r2.done_reason in ("stop", "length") and not r2.message.tool_calls


def test_tool_call_stream(oc, tag, fake_only):
    parts = list(oc.chat(model=tag, messages=ASK_WEATHER, tools=[WEATHER], stream=True))
    calls = [c for p in parts for c in (p.message.tool_calls or [])]
    assert len(calls) == 1 and calls[0].function.arguments == {"city": "Paris"}
    assert parts[-1].done


def test_python_function_as_a_tool(oc, tag, fake_only):
    def get_weather(city: str) -> str:
        """Weather in a city.

        Args:
            city: The city.
        """
        return "sunny"

    r = oc.chat(model=tag, messages=ASK_WEATHER, tools=[get_weather])
    assert r.message.tool_calls[0].function.name == "get_weather"


def test_think(oc, tag, fake_only):
    msgs = [{"role": "user", "content": "[[fake:think]]"}]
    r = oc.chat(model=tag, messages=msgs, think=True)
    assert r.message.thinking and r.message.content == "Here is a short answer."
    parts = list(oc.chat(model=tag, messages=msgs, think=True, stream=True))
    assert "".join(p.message.thinking or "" for p in parts) == r.message.thinking
    assert "".join(p.message.content for p in parts) == r.message.content
    g = oc.generate(model=tag, prompt="[[fake:think]]", think=True)
    assert g.thinking and g.response == "Here is a short answer."


def test_logprobs(oc, tag, fake_only):
    r = oc.chat(model=tag, messages=echo("abc"), logprobs=True, top_logprobs=2)
    assert len(r.logprobs) >= 3
    assert all(lp.logprob <= 0 and lp.top_logprobs and len(lp.top_logprobs) == 2 for lp in r.logprobs)


def test_ps_after_a_chat(oc, tag):
    oc.chat(model=tag, messages=ASK, options={"num_predict": 4})
    running = oc.ps().models
    assert [m.model for m in running] == [tag]
    assert running[0].size > 0 and running[0].expires_at is not None


def test_preload(oc, tag):
    r = oc.chat(model=tag, messages=[])
    assert r.done and r.done_reason == "load"
    r = oc.generate(model=tag, prompt="")
    assert r.done and r.done_reason == "load"


def test_unknown_model(oc):
    with pytest.raises(ollama.ResponseError) as e:
        oc.chat(model="llama3.2", messages=ASK)
    assert e.value.status_code == 404 and "model 'llama3.2' not found" in e.value.error
    with pytest.raises(ollama.ResponseError) as e:
        list(oc.chat(model="llama3.2", messages=ASK, stream=True))
    assert e.value.status_code == 404
    with pytest.raises(ollama.ResponseError) as e:
        oc.pull("llama3.2")
    assert e.value.status_code == 404


def test_images_are_refused(oc, tag):
    with pytest.raises(ollama.ResponseError) as e:
        oc.chat(model=tag, messages=[{"role": "user", "content": "what is this?", "images": [b"\x89PNG"]}])
    assert e.value.status_code == 400 and "images" in e.value.error


def test_not_implemented(oc, tag):
    for call in (lambda: oc.embed(model=tag, input="x"), lambda: oc.embeddings(model=tag, prompt="x"),
                 lambda: oc.delete(tag), lambda: oc.copy(tag, "copy")):
        with pytest.raises(ollama.ResponseError) as e:
            call()
        assert e.value.status_code == 501 and e.value.error


def test_closing_a_stream_frees_the_server(oc, tag, fake_only):
    stream = oc.chat(model=tag, messages=[{"role": "user", "content": "[[fake:loop]]"}], stream=True)
    next(stream)
    stream.close()
    assert oc.chat(model=tag, messages=echo("still here")).message.content == "still here"


def test_async_client(ollama_host, tag, fake_only):
    async def go():
        c = ollama.AsyncClient(host=ollama_host, headers=_headers())
        parts = [p async for p in await c.chat(model=tag, messages=echo("async"), stream=True)]
        assert "".join(p.message.content for p in parts) == "async" and parts[-1].done
        assert (await c.chat(model=tag, messages=echo("async"))).message.content == "async"
        assert (await c.list()).models

    asyncio.run(go())


def test_api_key(secured_server):
    base, key = secured_server["base"], secured_server["key"]
    good = ollama.Client(host=base, headers={"Authorization": "Bearer " + key})
    assert good.list().models
    assert good.chat(model=good.list().models[0].model, messages=echo("hi")).message.content == "hi"
    for headers in ({}, {"Authorization": "Bearer sk-wrong"}):
        with pytest.raises(ollama.ResponseError) as e:
            ollama.Client(host=base, headers=headers).list()
        assert e.value.status_code == 401
