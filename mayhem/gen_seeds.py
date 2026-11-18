#!/usr/bin/env python3
"""Regenerates the seed corpora under mayhem/<target>/testsuite. Deterministic; standard library only."""
import json
import os
import random
import struct

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- GGUF
F32, F16, Q4_0, Q4_1, Q8_0, Q4_K, Q6_K = 0, 1, 2, 3, 8, 12, 14
BLOCK = {F32: (1, 4), F16: (1, 2), Q4_0: (32, 18), Q4_1: (32, 20), Q8_0: (32, 34), Q4_K: (256, 144), Q6_K: (256, 210)}
T_I32, T_F32, T_STR = 5, 6, 8


def gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def kv(key, typ, val):
    out = gstr(key) + struct.pack("<I", typ)
    if typ == T_STR:
        return out + gstr(val)
    return out + struct.pack("<i" if typ == T_I32 else "<f", val)


def tensor_bytes(rng, typ, dims):
    n = 1
    for d in dims:
        n *= d
    per, size = BLOCK[typ]
    assert n % per == 0, (dims, typ)
    if typ == F32:
        return struct.pack("<%df" % n, *[rng.uniform(-0.5, 0.5) for _ in range(n)])
    if typ == F16:
        return b"".join(struct.pack("<e", rng.uniform(-0.5, 0.5)) for _ in range(n))
    # Quantized blocks: random payload with a small positive fp16 scale at the start of each block.
    out = bytearray()
    for _ in range(n // per):
        blk = bytearray(rng.randrange(256) for _ in range(size))
        blk[0:2] = struct.pack("<e", 0.05)
        if typ == Q4_1:
            blk[2:4] = struct.pack("<e", 0.0)
        if typ in (Q4_K, Q6_K):
            blk[size - 2:size] = struct.pack("<e", 0.05)
        out += blk
    return bytes(out)


def gguf(kvs, tensors, seed=1):
    """tensors: list of (name, ggml_type, dims in ggml order)."""
    rng = random.Random(seed)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs)
    infos, blobs, off = b"", [], 0
    for name, typ, dims in tensors:
        blob = tensor_bytes(rng, typ, dims)
        infos += gstr(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        infos += struct.pack("<IQ", typ, off)
        pad = (-len(blob)) % 32
        blobs.append(blob + b"\0" * pad)
        off += len(blob) + pad
    prefix = head + infos
    prefix += b"\0" * ((-len(prefix)) % 32)
    return prefix + b"".join(blobs)


def llama(types, hidden, heads, kv_heads, hs, ff, vocab, file_type, arch="llama"):
    kvs = [
        kv("general.architecture", T_STR, arch),
        kv("general.file_type", T_I32, file_type),
        kv(arch + ".block_count", T_I32, 1),
        kv(arch + ".attention.head_count", T_I32, heads),
        kv(arch + ".attention.head_count_kv", T_I32, kv_heads),
        kv(arch + ".embedding_length", T_I32, hidden),
        kv(arch + ".context_length", T_I32, 64),
        kv(arch + ".attention.layer_norm_rms_epsilon", T_F32, 1e-5),
        kv(arch + ".rope.freq_base", T_F32, 10000.0),
    ]
    t = types
    tensors = [
        ("token_embd.weight", t["embed"], [hidden, vocab]),
        ("output_norm.weight", F32, [hidden]),
        ("output.weight", t["out"], [hidden, vocab]),
        ("blk.0.attn_norm.weight", F32, [hidden]),
        ("blk.0.ffn_norm.weight", F32, [hidden]),
        ("blk.0.attn_q.weight", t["attn"], [hidden, heads * hs]),
        ("blk.0.attn_k.weight", t["attn"], [hidden, kv_heads * hs]),
        ("blk.0.attn_v.weight", t["attn"], [hidden, kv_heads * hs]),
        ("blk.0.attn_output.weight", t["attn_o"], [heads * hs, hidden]),
        ("blk.0.ffn_gate.weight", t["ffn"], [hidden, ff]),
        ("blk.0.ffn_up.weight", t["ffn"], [hidden, ff]),
        ("blk.0.ffn_down.weight", t["ffn_down"], [ff, hidden]),
    ]
    return gguf(kvs, tensors)


def lora_gguf():
    kvs = [kv("general.architecture", T_STR, "llama"), kv("general.type", T_STR, "adapter")]
    tensors = [
        ("blk.0.attn_q.weight.lora_a", F32, [32, 4]),
        ("blk.0.attn_q.weight.lora_b", F32, [4, 32]),
        ("blk.0.ffn_up.weight.lora_a", F32, [32, 4]),
        ("blk.0.ffn_up.weight.lora_b", F32, [4, 64]),
    ]
    return gguf(kvs, tensors, seed=7)


def write(target, name, data):
    d = os.path.join(ROOT, target, "testsuite")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "wb" if isinstance(data, bytes) else "w") as f:
        f.write(data)


def gguf_seeds():
    same = lambda t: {k: t for k in ("embed", "out", "attn", "attn_o", "ffn", "ffn_down")}
    seeds = {
        "llama_f32.gguf": llama(same(F32), 32, 4, 4, 8, 64, 16, 0),
        "llama_f16.gguf": llama(same(F16), 32, 4, 4, 8, 64, 16, 1),
        "llama_q4_0.gguf": llama(same(Q4_0), 32, 4, 2, 8, 64, 16, 2),
        "llama_q4_1.gguf": llama(same(Q4_1), 32, 4, 4, 8, 64, 16, 3),
        "llama_q8_0.gguf": llama(same(Q8_0), 32, 4, 4, 8, 64, 16, 7),
        "qwen2_q4_0.gguf": llama(same(Q4_0), 32, 4, 4, 8, 64, 16, 2, arch="qwen2"),
        # K-quants need a row length that is a whole number of 256-weight super-blocks.
        "llama_q4_k.gguf": llama({"embed": Q6_K, "out": Q6_K, "attn": Q4_K, "attn_o": F16, "ffn": Q4_K, "ffn_down": F16},
                                 256, 1, 1, 32, 32, 4, 15),
    }
    for name, data in seeds.items():
        write("fuzz_gguf_model", name, data)
    write("fuzz_gguf_lora", "lora_f32.gguf", lora_gguf())
    write("fuzz_gguf_lora", "llama_f16.gguf", seeds["llama_f16.gguf"])


# ---------------------------------------------------------------- safetensors
def safetensors(entries, metadata=None, order=None):
    rng = random.Random(3)
    header, blob = {}, b""
    if metadata:
        header["__metadata__"] = metadata
    for name, dtype, shape in entries:
        n = 1
        for d in shape:
            n *= d
        width = {"F32": 4, "F16": 2, "BF16": 2}[dtype]
        data = bytes(rng.randrange(256) for _ in range(n * width)) if dtype != "F32" else \
            struct.pack("<%df" % n, *[rng.uniform(-1, 1) for _ in range(n)])
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(blob), len(blob) + len(data)]}
        blob += data
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((-len(h)) % 8)
    return struct.pack("<Q", len(h)) + h + blob


def safetensors_seeds():
    p = "base_model.model.model.layers.0.self_attn.q_proj"
    write("fuzz_safetensors_lora", "lora_f32.safetensors", safetensors(
        [(p + ".lora_A.weight", "F32", [4, 32]), (p + ".lora_B.weight", "F32", [32, 4])]))
    q = "base_model.model.model.layers.0.mlp.up_proj"
    write("fuzz_safetensors_lora", "lora_f16_alpha.safetensors", safetensors(
        [(p + ".lora_A.weight", "F16", [4, 32]), (p + ".lora_B.weight", "F16", [32, 4]),
         (q + ".lora_A.weight", "BF16", [4, 32]), (q + ".lora_B.weight", "BF16", [64, 4]),
         (p + ".alpha", "F32", [1])], metadata={"format": "pt"}))
    write("fuzz_safetensors_lora", "diffusers_lora.safetensors", safetensors(
        [("unet.down_blocks.0.attentions.0.to_q.lora.down.weight", "F32", [4, 16]),
         ("unet.down_blocks.0.attentions.0.to_q.lora.up.weight", "F32", [16, 4])]))


# ---------------------------------------------------------------- text seeds
def text_seeds():
    jc = {
        "chat.json": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"},
                      {"role": "assistant", "content": "Hello!", "reasoning_content": "greet"}],
        "tool_call.json": {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "units": ["c", "f"]}}}]},
        "nested.json": {"a": [1, 2.5, -3e10, True, None, "sé中\\n"], "b": {"c": {"d": []}}},
    }
    for name, value in jc.items():
        write("fuzz_json_container", name, json.dumps(value))
    write("fuzz_json_container", "escapes.json", '"\\u00e9\\ud83d\\ude00 \\n\\t\\"x\\""')

    parsers = {
        "reasoning.txt": b"\x00<think>First, add 2 and 3.</think>The answer is 5.",
        "reasoning_noopen.txt": b"\x01Let me think.</a>Done.",
        "deepseek.txt": b"\x02I should check the units.</think>42 km",
        "phi4.txt": b"\x03<think>step one\nstep two</think>result",
        "llama3_json.txt": b'\x04{"type": "function", "function": {"name": "get_weather", "parameters": {"city": "Paris"}}}',
        "llama3_pythonic.txt": b"\x05[get_weather(city='New York, NY', unit='celsius'), get_time(tz=\"UTC\", n=3, f=1.5)]",
        "stream_think.txt": b"\x80\x07<think>\x07step one\x07</think>\x07answer",
        "stream_r1.txt": b"\x84\x03abc\x07 </think\x07> done",
    }
    for name, data in parsers.items():
        write("fuzz_parsers", name, data)

    gc = {
        "greedy.json": {"max_new_tokens": 128, "eos_token_id": 2, "bos_token_id": 1, "pad_token_id": 0},
        "sampling.json": {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 40, "repetition_penalty": 1.1,
                          "max_length": 512, "eos_token_id": [2, 32000], "stop_strings": ["</s>"]},
        "beam.json": {"num_beams": 4, "num_return_sequences": 2, "num_beam_groups": 2, "diversity_penalty": 1.0,
                      "length_penalty": 0.8, "no_repeat_ngram_size": 3, "early_stopping": True, "max_new_tokens": 32},
    }
    for name, value in gc.items():
        write("fuzz_generation_config", name, json.dumps(value))

    misaki = {
        "hello.txt": b"\x00Hello, world! This is a test.",
        "numbers.txt": b"\x01It costs $12.50 on 3rd May, at 10:30 pm, 1,234 items (2.5%).",
        "words.txt": b"\x00The quick brown fox jumps over the lazy dog; don't stop -- read [this](/\xcb\x88\xc3\xb0\xc9\xaas/).",
    }
    for name, data in misaki.items():
        write("fuzz_misaki_g2p", name, data)

    rng = random.Random(5)
    header = struct.pack("<5H", 8, 8, 8, 12, 10)
    write("fuzz_clip_preprocess", "small.bin", header + bytes(rng.randrange(256) for _ in range(12 * 10 * 3)))
    header = struct.pack("<5H", 224, 224, 224, 40, 24)
    write("fuzz_clip_preprocess", "upscale.bin", header + bytes(rng.randrange(256) for _ in range(40 * 24 * 3)))
    header = struct.pack("<5H", 4, 16, 2, 30, 30)
    write("fuzz_clip_preprocess", "crop_larger.bin", header + bytes(rng.randrange(256) for _ in range(30 * 30 * 3)))


def chat_template_seeds():
    # [flags][template]\0[JSON]; templates in the shapes models ship in tokenizer_config.json.
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "Weather?"}]
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
    chatml = ("{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + "
              "'<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
    llama = ("{{ bos_token }}{% for m in messages %}{% if m.role == 'user' %}{{ '[INST] ' + m.content | trim + ' [/INST]' }}"
             "{% elif m.role == 'assistant' %}{{ ' ' + m.content + eos_token }}{% else %}{{ m.content }}{% endif %}{% endfor %}")
    toolish = ("{%- if tools %}{%- for t in tools %}{{ t | tojson }}{%- endfor %}{%- endif %}{%- set ns = namespace(n=0) -%}"
               "{%- for m in messages -%}{%- set ns.n = ns.n + 1 -%}{{ loop.index }}:{{ m['role'] | upper }}={{ m.content }};"
               "{%- endfor -%}{{ ns.n }}{% if x is undefined %}U{% endif %}{{ messages[-1].content[:3] }}")
    macro = ("{% macro r(x) %}{{ x | replace('a', 'b') }}{% endmacro %}{% for m in messages|selectattr('role','equalto','user') %}"
             "{{ r(m.content) }}{{ m.content.split(' ')|join(',') }}{% endfor %}{{ range(3)|list }}{{ {'k': 1}|items|list }}")
    write("fuzz_chat_template", "chatml.bin", b"\x01" + chatml.encode() + b"\0" + json.dumps(msgs).encode())
    write("fuzz_chat_template", "llama.bin", b"\x00" + llama.encode() + b"\0" + json.dumps(msgs[1:]).encode())
    write("fuzz_chat_template", "tools.bin", b"\x07" + toolish.encode() + b"\0" +
          json.dumps({"messages": msgs, "tools": tools, "extra_context": {"enable_thinking": True}}).encode())
    write("fuzz_chat_template", "macro.bin", b"\x0a" + macro.encode())


if __name__ == "__main__":
    gguf_seeds()
    safetensors_seeds()
    text_seeds()
    chat_template_seeds()
