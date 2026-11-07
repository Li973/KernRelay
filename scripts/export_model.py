"""Export a HuggingFace LLM (Qwen2.5 dense / Qwen3-VL dense text / Qwen3-Omni
inventory) into KernRelay's manifest + weights.bin layout.

Usage:
  python scripts/export_model.py --model /path/to/model --out /path/to/cache/kr \
      [--seq 2048] [--max-pos 4096] [--arch auto|qwen2.5|qwen3_vl|inventory]

KernRelay never vendors model weights: the cache directory is external and is
passed to the runtime with --model/--model-dir.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def tensor_bytes(t):
    t = t.detach().cpu().contiguous()
    dtype = str(t.dtype).replace("torch.", "")
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return t.numpy().tobytes(), dtype


def detect(cfg):
    mt = cfg.get("model_type", "")
    if mt in ("qwen3_omni_moe",) or "thinker_config" in cfg:
        return "inventory"
    if "text_config" in cfg:
        return "qwen3_vl"
    return "qwen2.5"


def text_spec(cfg, arch):
    if arch == "qwen3_vl":
        t = cfg["text_config"]
        return {
            "text": t, "H": t["hidden_size"], "I": t["intermediate_size"],
            "L": t["num_hidden_layers"], "HQ": t["num_attention_heads"],
            "HKV": t["num_key_value_heads"],
            "D": t.get("head_dim", t["hidden_size"] // t["num_attention_heads"]),
            "V": t["vocab_size"], "eps": t.get("rms_norm_eps", 1e-6),
            "qk_norm": True, "bias": t.get("attention_bias", False),
            "prefix": "model.language_model.",
            "layers": "model.language_model.layers.%d.",
            "embed": "model.language_model.embed_tokens.weight",
            "norm": "model.language_model.norm.weight",
            "theta": t.get("rope_theta", 1e6),
        }
    t = cfg
    return {
        "text": t, "H": t["hidden_size"], "I": t["intermediate_size"],
        "L": t["num_hidden_layers"], "HQ": t["num_attention_heads"],
        "HKV": t["num_key_value_heads"],
        "D": t.get("head_dim", t["hidden_size"] // t["num_attention_heads"]),
        "V": t["vocab_size"], "eps": t.get("rms_norm_eps", 1e-6),
        "qk_norm": False, "bias": t.get("attention_bias", True),
        "prefix": "model.", "layers": "model.layers.%d.",
        "embed": "model.embed_tokens.weight", "norm": "model.norm.weight",
        "theta": t.get("rope_theta", 1e6),
    }


def rope_table(max_pos, D, theta):
    inv = 1.0 / (torch.tensor(theta, dtype=torch.float64) **
                 (torch.arange(0, D, 2, dtype=torch.float64) / D))
    pos = torch.arange(max_pos, dtype=torch.float64)
    freqs = torch.outer(pos, inv)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(torch.bfloat16)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(torch.bfloat16)
    return cos.unsqueeze(0).unsqueeze(1), sin.unsqueeze(0).unsqueeze(1)


def load_tensors(model_dir):
    from safetensors import safe_open
    out = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for k in sf.keys():
                out[k] = sf.get_tensor(k)
    return out


def inventory(cfg, arch):
    if arch != "inventory":
        return None
    think = cfg.get("thinker_config", {})
    text = think.get("text_config", {})
    if not text:
        text = cfg.get("text_config", {})
    n_exp = text.get("num_experts", 0)
    per = text.get("num_experts_per_tok", 0)
    return {
        "arch": "qwen3_omni_moe",
        "hidden": text.get("hidden_size"),
        "intermediate": text.get("intermediate_size"),
        "layers": text.get("num_hidden_layers"),
        "heads": text.get("num_attention_heads"),
        "kv_heads": text.get("num_key_value_heads"),
        "head_dim": text.get("head_dim"),
        "experts": n_exp, "experts_per_tok": per,
        "moe_intermediate": text.get("moe_intermediate_size"),
        "shared_intermediate": text.get("shared_expert_intermediate_size"),
        "vocab": text.get("vocab_size"),
        "executable": False,
        "reason": "MoE grouped matmul/routing not implemented in KernRelay 1.0",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-pos", type=int, default=4096)
    ap.add_argument("--arch", default="auto")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    model_dir = Path(args.model)
    cfg = json.loads((model_dir / "config.json").read_text())
    arch = detect(cfg) if args.arch == "auto" else args.arch
    cache = Path(os.environ.get("KERNRELAY_CACHE", Path.home() / ".cache/kernrelay"))
    out_dir = Path(args.out) if args.out else (cache / model_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    if arch == "inventory":
        inv = inventory(cfg, arch)
        (out_dir / "inventory.json").write_text(
            json.dumps(inv, ensure_ascii=False, indent=1), encoding="utf-8")
        print("inventory only:", json.dumps(inv, ensure_ascii=False))
        return

    sp = text_spec(cfg, arch)
    tensors = load_tensors(model_dir)
    manifest = {
        "config": {
            "arch": arch, "hidden_size": sp["H"],
            "intermediate_size": sp["I"], "num_hidden_layers": sp["L"],
            "num_attention_heads": sp["HQ"], "num_key_value_heads": sp["HKV"],
            "head_dim": sp["D"], "vocab_size": sp["V"],
            "rms_norm_eps": sp["eps"], "qk_norm": sp["qk_norm"],
            "attention_bias": sp["bias"],
            "tie_word_embeddings": bool(cfg.get("tie_word_embeddings", True)),
            "rope_theta": sp["theta"], "hidden_act": "silu",
        },
        "tensors": {}, "seq": args.seq, "seed": args.seed,
        "max_pos": args.max_pos,
    }
    out = open(out_dir / "weights.bin", "wb")
    cursor = [0]

    def put(name, t, transpose=False, source_shape=None):
        if transpose:
            t = t.t().contiguous()
        data, dtype = tensor_bytes(t)
        manifest["tensors"][name] = {
            "offset": cursor[0], "bytes": len(data), "dtype": dtype,
            "shape": list(t.shape)}
        if source_shape is not None:
            manifest["tensors"][name]["source_shape"] = list(source_shape)
        out.write(data)
        cursor[0] += len(data)

    for l in range(sp["L"]):
        p = sp["layers"] % l
        for role in ("q_proj", "k_proj", "v_proj", "o_proj"):
            src = p + "self_attn." + role + ".weight"
            t = tensors.pop(src)
            put("L%02d.%s.weight_t" % (l, role), t, True, t.shape)
        if sp["bias"]:
            for role in ("q_proj", "k_proj", "v_proj"):
                src = p + "self_attn." + role + ".bias"
                if src in tensors:
                    put("L%02d.%s.bias" % (l, role), tensors.pop(src))
        for role in ("gate_proj", "up_proj", "down_proj"):
            src = p + "mlp." + role + ".weight"
            t = tensors.pop(src)
            put("L%02d.%s.weight_t" % (l, role), t, True, t.shape)
        for role in ("input_layernorm", "post_attention_layernorm"):
            put("L%02d.%s.weight" % (l, role), tensors.pop(p + role + ".weight"))
        if sp["qk_norm"]:
            for role in ("q_norm", "k_norm"):
                src = p + "self_attn." + role + ".weight"
                put("L%02d.%s.weight" % (l, role), tensors.pop(src))

    emb = tensors.pop(sp["embed"])
    if emb.dtype != torch.bfloat16:
        emb = emb.to(torch.bfloat16)
    put("embed.weight", emb)
    lm = emb if cfg.get("tie_word_embeddings", True) else tensors.pop("lm_head.weight")
    put("lm_head.weight_t", lm, True, lm.shape)
    put("final_norm.weight", tensors.pop(sp["norm"]))
    del tensors

    cos, sin = rope_table(args.max_pos, sp["D"], sp["theta"])
    put("rope.cos", cos)
    put("rope.sin", sin)

    # Deterministic ids + reference logits from the HF model (text-only prefill).
    gen = torch.Generator().manual_seed(args.seed)
    ids = torch.randint(0, sp["V"], (1, args.seq), generator=gen,
                        dtype=torch.long)
    ids.numpy().tofile(out_dir / "ids.bin")
    from transformers import AutoModelForCausalLM
    if arch == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration as Cls
    else:
        Cls = AutoModelForCausalLM
    model = Cls.from_pretrained(str(model_dir), dtype=torch.bfloat16).eval().npu(args.device)
    with torch.no_grad():
        logits = model(ids.npu(args.device)).logits
    last = logits[0, -1].float().cpu()
    last.numpy().tofile(out_dir / "ref_last_logits.bin")
    ref = {"argmax_last": int(last.argmax()), "sum": float(last.sum()),
           "abs_mean": float(last.abs().mean()), "arch": arch}
    (out_dir / "ref.json").write_text(json.dumps(ref, indent=1))
    del model
    out.close()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("wrote", out_dir, "bytes", cursor[0], "ref", ref)


if __name__ == "__main__":
    main()
