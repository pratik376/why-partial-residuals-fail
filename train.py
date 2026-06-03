# train.py
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from config import (
    BATCH_SIZE,
    BLOCK_SIZE,
    BG,
    CONFIGS,
    DEFAULT_BASE_DIR,
    DEFAULT_DATA_DIR,
    EVAL_INTERVAL,
    EVAL_ITERS,
    GRAD_ACCUM,
    GRID,
    LR_DECAY_ITERS,
    MAX_ITERS,
    MAX_LR,
    MIN_LR,
    N_EMBD,
    N_HEAD,
    N_LAYER,
    PANEL,
    RESULTS_FILENAME,
    SEED,
    TXT,
    WARMUP_ITERS,
    config_by_name,
    config_names,
)
from data import ensure_openwebtext_cache, load_data_tensors
from model import GPT

def detect_runtime():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = "bfloat16"
    elif device == "cuda":
        dtype = "float16"
    else:
        dtype = "float32"

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]

    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype) if device == "cuda" else None
    return device, dtype, ctx

DEVICE, DTYPE, AMP_CTX = detect_runtime()

def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_lr(it: int) -> float:
    if it < WARMUP_ITERS:
        return MAX_LR * it / WARMUP_ITERS
    if it > LR_DECAY_ITERS:
        return MIN_LR
    decay = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * decay)) * (MAX_LR - MIN_LR)

def get_batch(split: str, train_data: torch.Tensor, val_data: torch.Tensor):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE].long() for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1].long() for i in ix])

    if DEVICE == "cuda":
        x = x.pin_memory().to(DEVICE, non_blocking=True)
        y = y.pin_memory().to(DEVICE, non_blocking=True)
    else:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(EVAL_ITERS):
            xb, yb = get_batch(split, train_data, val_data)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if DEVICE == "cuda" and DTYPE == "bfloat16" else torch.autocast(device_type="cuda", dtype=torch.float16) if DEVICE == "cuda" else torch.no_grad():
                _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out

@torch.no_grad()
def collect_activation_stats(model):
    model.eval()
    stats = {}
    for i, ffn in enumerate(model.get_ffn_layers()):
        if ffn._act_out is None:
            continue
        flat = ffn._act_out.float().cpu().reshape(-1).numpy()
        stats[str(i)] = {
            "dead": float((flat <= 0).mean()),
            "near_zero": float((np.abs(flat) < 0.01).mean()),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
        }
    model.train()
    return stats

def collect_gradient_norms(model):
    norms = {}
    for i, ffn in enumerate(model.get_ffn_layers()):
        g1 = ffn.fc1.weight.grad
        g2 = ffn.fc2.weight.grad
        if g1 is not None and g2 is not None:
            norms[str(i)] = {"fc1": g1.norm(2).item(), "fc2": g2.norm(2).item()}
    return norms

@torch.no_grad()
def collect_hidden_norms(model):
    model.eval()
    norms = {str(i): b._h_norm for i, b in enumerate(model.get_blocks())}
    model.train()
    return norms

def collect_weight_norms(model):
    norms = {}
    for i, ffn in enumerate(model.get_ffn_layers()):
        w1 = ffn.fc1.weight.data.norm(2).item()
        w2 = ffn.fc2.weight.data.norm(2).item()
        norms[str(i)] = {"fc1": w1, "fc2": w2, "mean": (w1 + w2) / 2}
    return norms

def ckpt_path(base_dir: Path, cfg_name: str) -> Path:
    return base_dir / "checkpoints" / f"ckpt_{cfg_name}.pt"

def save_checkpoint(base_dir: Path, step: int, model, optimizer, scaler, stats, cfg_name: str):
    path = ckpt_path(base_dir, cfg_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "stats": stats,
            "cfg_name": cfg_name,
        },
        path,
    )
    return path

def load_checkpoint(base_dir: Path, cfg_name: str, model, optimizer, scaler):
    path = ckpt_path(base_dir, cfg_name)
    if not path.exists():
        return 0, {}
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt["step"]), ckpt.get("stats", {})

def save_results(base_dir: Path, results: dict):
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / RESULTS_FILENAME
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    return results_file

def run_config(cfg: dict, base_dir: Path, train_data, val_data, vocab_size: int, resume: bool = True):
    set_seed(SEED)

    model = GPT(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        dropout=cfg.get("dropout", 0.1),
        attn_res=cfg["attn_res"],
        ffn_res=cfg["ffn_res"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MAX_LR,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    use_scaler = DEVICE == "cuda" and DTYPE == "float16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    start_step = 0
    saved_stats = {}
    if resume:
        start_step, saved_stats = load_checkpoint(base_dir, cfg["name"], model, optimizer, scaler)

    iters_log = saved_stats.get("iters_log", [])
    train_losses = saved_stats.get("train_losses", [])
    val_losses = saved_stats.get("val_losses", [])
    lr_log = saved_stats.get("lr_log", [])
    gpu_mems = saved_stats.get("gpu_mems", [])
    tokens_list = saved_stats.get("tokens_list", [])
    tok_rates = saved_stats.get("tok_rates", [])
    grad_norm_history = saved_stats.get("grad_norm_history", [])
    hidden_norm_history = saved_stats.get("hidden_norm_history", [])
    act_stats_history = saved_stats.get("act_stats_history", [])
    weight_norm_history = saved_stats.get("weight_norm_history", [])

    tokens_per_step = BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM
    tokens_processed = start_step * tokens_per_step
    t0 = time.time()

    for it in range(start_step, MAX_ITERS):
        lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        for _ in range(GRAD_ACCUM):
            xb, yb = get_batch("train", train_data, val_data)
            if AMP_CTX is not None:
                with AMP_CTX:
                    _, loss = model(xb, yb)
            else:
                _, loss = model(xb, yb)
            scaler.scale(loss / GRAD_ACCUM).backward()

        scaler.unscale_(optimizer)
        gnorms = collect_gradient_norms(model)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        tokens_processed += tokens_per_step

        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            astats = collect_activation_stats(model)
            hnorms = collect_hidden_norms(model)
            wnorms = collect_weight_norms(model)

            elapsed = max(time.time() - t0, 1e-6)
            gpu_mb = torch.cuda.memory_allocated() / 1e6 if DEVICE == "cuda" else 0.0
            tok_s = tokens_processed / elapsed
            ppl = math.exp(min(losses["val"], 20))
            gap = losses["val"] - losses["train"]

            iters_log.append(it)
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            lr_log.append(lr)
            gpu_mems.append(gpu_mb)
            tokens_list.append(tokens_processed)
            tok_rates.append(tok_s)
            grad_norm_history.append(gnorms)
            hidden_norm_history.append(hnorms)
            act_stats_history.append(astats)
            weight_norm_history.append(wnorms)

            stats_snap = {
                "iters_log": iters_log,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "lr_log": lr_log,
                "gpu_mems": gpu_mems,
                "tokens_list": tokens_list,
                "tok_rates": tok_rates,
                "grad_norm_history": grad_norm_history,
                "hidden_norm_history": hidden_norm_history,
                "act_stats_history": act_stats_history,
                "weight_norm_history": weight_norm_history,
            }

            save_checkpoint(base_dir, it, model, optimizer, scaler, stats_snap, cfg["name"])

            print(
                f'{cfg["name"]} | step {it:>5,} | '
                f'train {losses["train"]:.4f} | '
                f'val {losses["val"]:.4f} | '
                f'ppl {ppl:.1f} | '
                f'gap {gap:+.4f} | '
                f'lr {lr:.2e} | '
                f'{tok_s:,.0f} tok/s'
            )

    final_train = train_losses[-1]
    final_val = val_losses[-1]
    final_ppl = math.exp(min(final_val, 20))

    result = {
        **stats_snap,
        "final_train": final_train,
        "final_val": final_val,
        "final_ppl": final_ppl,
        "n_params": model.n_params(),
        "complete": True,
    }

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--target_docs", type=int, default=100_000)
    parser.add_argument("--config", choices=config_names(), default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--refresh_data", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    data_dir = Path(args.data_dir)
    (base_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (base_dir / "results").mkdir(parents=True, exist_ok=True)
    (base_dir / "figures").mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_data or not (data_dir / "train.bin").exists():
        ensure_openwebtext_cache(data_dir, target_docs=args.target_docs)
    train_data, val_data = load_data_tensors(data_dir)

    vocab_size = 50_257

    results_file = base_dir / "results" / RESULTS_FILENAME
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    configs = [config_by_name(args.config)] if args.config else CONFIGS

    for cfg in configs:
        if all_results.get(cfg["name"], {}).get("complete"):
            print(f"Skipping completed config: {cfg['name']}")
            continue

        result = run_config(
            cfg=cfg,
            base_dir=base_dir,
            train_data=train_data,
            val_data=val_data,
            vocab_size=vocab_size,
            resume=not args.no_resume,
        )
        all_results[cfg["name"]] = result
        save_results(base_dir, all_results)

    print("\nFinal results:")
    print(f'{"Config":<22} | {"Val Loss":>10} | {"PPL":>8} | {"Train":>10}')
    print("-" * 58)
    for name, res in all_results.items():
        if res.get("complete"):
            print(f'{name:<22} | {res["final_val"]:>10.4f} | {res["final_ppl"]:>8.1f} | {res["final_train"]:>10.4f}')

if __name__ == "__main__":
    main()