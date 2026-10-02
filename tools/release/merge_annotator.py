"""Merge the supplied standard LoRA adapter into its Qwen base, shard by shard.

Uses W + (alpha / rank) * B @ A, computed in float32 and saved in base dtype.
Validates every adapter key against the base shard index before writing.
Run with --base BASE --adapter ADAPTER --out NEW_DIRECTORY.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import torch
from safetensors.torch import load_file, save_file


def adapter_key_to_base(name: str) -> str:
    """base_model.model.model.layers.N.X.lora_A.weight -> model.language_model.layers.N.X.weight"""
    assert name.startswith("base_model.model."), name
    core = name[len("base_model.model."):]
    for suf in (".lora_A.weight", ".lora_B.weight"):
        if core.endswith(suf):
            core = core[: -len(suf)] + ".weight"
            break
    if core.startswith("model.layers."):
        core = "model.language_model.layers." + core[len("model.layers."):]
    elif core.startswith("model.") and not core.startswith("model.language_model."):
        core = "model.language_model." + core[len("model."):]
    return core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    cfg = json.load(open(os.path.join(args.adapter, "adapter_config.json"),
                        encoding="utf-8"))
    assert cfg.get("peft_type") == "LORA", cfg.get("peft_type")
    assert not cfg.get("modules_to_save"), "modules_to_save не поддержан"
    assert cfg.get("bias", "none") == "none", "смещения не поддержаны"
    assert not cfg.get("use_dora", False), "DoRA не поддержана"
    r = int(cfg["r"])
    scale = float(cfg["lora_alpha"]) / r
    if cfg.get("use_rslora"):
        scale = float(cfg["lora_alpha"]) / (r ** 0.5)
    print(f"адаптер: r {r}, alpha {cfg['lora_alpha']}, множитель {scale}", flush=True)

    ad = load_file(os.path.join(args.adapter, "adapter_model.safetensors"))
    pairs = {}
    for k, v in ad.items():
        base_key = adapter_key_to_base(k)
        slot = pairs.setdefault(base_key, {})
        slot["A" if ".lora_A." in k else "B"] = v
    bad = [k for k, v in pairs.items() if set(v) != {"A", "B"}]
    assert not bad, f"неполные пары весов: {bad[:5]}"
    print(f"модулей к слиянию {len(pairs)}", flush=True)

    index_path = os.path.join(args.base, "model.safetensors.index.json")
    index = json.load(open(index_path, encoding="utf-8"))
    wmap = index["weight_map"]
    missing = [k for k in pairs if k not in wmap]
    assert not missing, f"ключей нет в базе: {missing[:5]}"

    shards = sorted(set(wmap.values()))
    os.makedirs(args.out, exist_ok=False)
    n_merged = 0
    for si, shard in enumerate(shards, 1):
        src = os.path.join(args.base, shard)
        sd = load_file(src)
        touched = 0
        for key in list(sd.keys()):
            if key not in pairs:
                continue
            A = pairs[key]["A"].to(torch.float32)
            B = pairs[key]["B"].to(torch.float32)
            W = sd[key]
            delta = (B @ A) * scale
            assert delta.shape == W.shape, (key, tuple(delta.shape), tuple(W.shape))
            sd[key] = (W.to(torch.float32) + delta).to(W.dtype)
            touched += 1
            n_merged += 1
        save_file(sd, os.path.join(args.out, shard), metadata={"format": "pt"})
        del sd
        print(f"срез {si} из {len(shards)}: {shard}, слито весов {touched}, "
              f"прошло {time.time() - t0:.0f} c", flush=True)

    assert n_merged == len(pairs), f"слито {n_merged}, ожидалось {len(pairs)}"

    for name in sorted(os.listdir(args.base)):
        if name.endswith(".safetensors"):
            continue
        src = os.path.join(args.base, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.out, name))

    meta = {"base": args.base, "adapter": args.adapter, "scale": scale,
            "r": r, "lora_alpha": cfg["lora_alpha"],
            "modules_merged": len(pairs), "method": "прямое слияние по срезам",
            "adapter_meta": (json.load(open(os.path.join(args.adapter,
                                                         "train_meta.json"),
                                           encoding="utf-8"))
                             if os.path.exists(os.path.join(args.adapter,
                                                            "train_meta.json"))
                             else None),
            "merged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(args.out, "merge_meta.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
