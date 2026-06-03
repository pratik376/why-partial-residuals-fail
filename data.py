# data.py
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import tiktoken
from datasets import load_dataset

def get_tokenizer(name: str = "gpt2"):
    enc = tiktoken.get_encoding(name)
    return enc, enc.n_vocab

def ensure_openwebtext_cache(
    cache_dir: str | Path,
    target_docs: int = 100_000,
    tokenizer_name: str = "gpt2",
) -> tuple[Path, Path, Path, int]:
    """
    Creates:
      train.bin, val.bin, meta.json
    inside cache_dir if they do not already exist.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_bin = cache_dir / "train.bin"
    val_bin = cache_dir / "val.bin"
    meta_file = cache_dir / "meta.json"

    enc, vocab_size = get_tokenizer(tokenizer_name)

    if not train_bin.exists() or train_bin.stat().st_size < 1_000_000:
        print("Downloading OpenWebText...")
        dataset = load_dataset("openwebtext", split="train", streaming=True)

        train_tokens: list[int] = []
        val_tokens: list[int] = []
        n_docs = 0

        for doc in dataset:
            text = doc["text"]
            tokens = enc.encode_ordinary(text)
            tokens.append(enc.eot_token)

            if n_docs % 1000 == 0:
                val_tokens.extend(tokens)
            else:
                train_tokens.extend(tokens)

            n_docs += 1
            if n_docs % 5000 == 0:
                print(f"  {n_docs:,} docs | {len(train_tokens)/1e6:.0f}M tokens", end="\r")

            if n_docs >= target_docs:
                break

        print(f"\n{len(train_tokens)/1e9:.3f}B train | {len(val_tokens)/1e6:.1f}M val")

        np.array(train_tokens, dtype=np.uint16).tofile(train_bin)
        np.array(val_tokens, dtype=np.uint16).tofile(val_bin)

        with open(meta_file, "w") as f:
            json.dump(
                {"train_tokens": len(train_tokens), "val_tokens": len(val_tokens)},
                f,
                indent=2,
            )
        print("Saved cache files")

    if not meta_file.exists():
        train_tokens = train_bin.stat().st_size // 2
        val_tokens = val_bin.stat().st_size // 2
        with open(meta_file, "w") as f:
            json.dump(
                {"train_tokens": int(train_tokens), "val_tokens": int(val_tokens)},
                f,
                indent=2,
            )

    return train_bin, val_bin, meta_file, vocab_size

def load_data_tensors(cache_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dir = Path(cache_dir)
    train_bin = cache_dir / "train.bin"
    val_bin = cache_dir / "val.bin"

    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")

    return (
        torch.from_numpy(train_data.astype(np.int64)),
        torch.from_numpy(val_data.astype(np.int64)),
    )