"""Model adapters / config inventory for the 0.3 optimize pipeline."""

import json


def inventory_config(cfg):
    think = cfg.get("thinker_config", {})
    text = think.get("text_config", {}) or cfg.get("text_config", {})
    n_exp = text.get("num_experts", 0)
    if n_exp:
        return {
            "arch": cfg.get("model_type", "qwen3_omni_moe"),
            "hidden": text.get("hidden_size"),
            "intermediate": text.get("intermediate_size"),
            "layers": text.get("num_hidden_layers"),
            "heads": text.get("num_attention_heads"),
            "kv_heads": text.get("num_key_value_heads"),
            "head_dim": text.get("head_dim"),
            "experts": n_exp,
            "experts_per_tok": text.get("num_experts_per_tok"),
            "moe_intermediate": text.get("moe_intermediate_size"),
            "shared_intermediate": text.get("shared_expert_intermediate_size"),
            "vocab": text.get("vocab_size"),
            "executable": False,
            "reason": "MoE grouped matmul/routing not implemented in KernRelay 0.3",
        }
    return {"arch": cfg.get("model_type"), "executable": True}


def load_manifest_config(kr_dir):
    p = json_load(kr_dir + "/manifest.json")
    return p["config"]


def json_load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
