#!/usr/bin/env python3
"""Tiny Gemma 4 fixtures for the native tests (STAGE1 1.0, 1.5.M).

    python3 tools/fixtures/make_tiny.py --llama-cpp /path/to/llama.cpp --out /tmp/fixtures

The target is tiny-random/gemma-4-e at a pinned revision: Gemma 4 E-shaped
(per-layer inputs, shared KV layers, sliding and full attention) with a
hidden size of 8 and random weights. No tiny drafter exists, and a drafter
must match its target (n_embd_out, the head dims and KV heads of the
target layers whose cache it reads), so one is made here: a
Gemma4AssistantForCausalLM initialized with a fixed seed from a config
derived from the target's. Both are converted with convert_hf_to_gguf.py
of the pinned llama.cpp tag.

The weights are random, so the fixtures check mechanics and equivalence
(MTP on and off give the same greedy tokens; the binding matches
llama-server), never quality; acceptance > 0 is checked only on E4B.
Needs torch (CPU), transformers >= 5 and safetensors.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request

TARGET_REPO = "tiny-random/gemma-4-e"
TARGET_REV = "9a7c1e443a09488bf1a8d5fb031b213791875368"
TARGET_FILES = [
    "config.json", "generation_config.json", "model.safetensors",
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "processor_config.json",
]
SEED = 20261018


def fetch_target(dst):
    os.makedirs(dst, exist_ok=True)
    for f in TARGET_FILES:
        path = os.path.join(dst, f)
        if os.path.exists(path):
            continue
        url = f"https://huggingface.co/{TARGET_REPO}/resolve/{TARGET_REV}/{f}"
        print("fetch", url)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)


def make_assistant(target_dir, dst):
    import torch
    from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM, Gemma4TextConfig

    with open(os.path.join(target_dir, "config.json")) as f:
        tc = json.load(f)["text_config"]
    # The assistant's layers read the target's last sliding and last full
    # layer caches, so their head geometry is the target's.
    text = Gemma4TextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        num_attention_heads=tc["num_attention_heads"],
        num_key_value_heads=tc["num_key_value_heads"],
        head_dim=tc["head_dim"],
        global_head_dim=tc["global_head_dim"],
        sliding_window=tc["sliding_window"],
        rope_parameters=tc["rope_parameters"],
        final_logit_softcapping=tc["final_logit_softcapping"],
        vocab_size=tc["vocab_size"],
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
        num_kv_shared_layers=2,
        rms_norm_eps=tc["rms_norm_eps"],
        tie_word_embeddings=True,
    )
    cfg = Gemma4AssistantConfig(text_config=text, backbone_hidden_size=tc["hidden_size"])
    cfg.architectures = ["Gemma4AssistantForCausalLM"]
    torch.manual_seed(SEED)
    model = Gemma4AssistantForCausalLM(cfg).to(torch.bfloat16)
    os.makedirs(dst, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True)
    for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
        shutil.copy(os.path.join(target_dir, f), os.path.join(dst, f))


def convert(llama_cpp, src, out):
    # Run from the checkout so the script imports its own gguf-py.
    subprocess.run([sys.executable, os.path.join(llama_cpp, "convert_hf_to_gguf.py"), src,
                    "--outfile", out, "--outtype", "f32"], check=True, cwd=llama_cpp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-cpp", required=True, help="llama.cpp checkout at the pinned tag")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    hf = os.path.join(out, "hf")
    target, assistant = os.path.join(hf, "tiny-gemma4"), os.path.join(hf, "tiny-gemma4-assistant")
    fetch_target(target)
    make_assistant(target, assistant)
    convert(os.path.abspath(a.llama_cpp), target, os.path.join(out, "tiny-gemma4.gguf"))
    convert(os.path.abspath(a.llama_cpp), assistant, os.path.join(out, "tiny-gemma4-mtp.gguf"))
    print("wrote", os.path.join(out, "tiny-gemma4.gguf"), os.path.join(out, "tiny-gemma4-mtp.gguf"))


if __name__ == "__main__":
    main()
