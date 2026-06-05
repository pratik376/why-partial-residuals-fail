# config.py
from __future__ import annotations

N_LAYER = 12
N_EMBD = 768
N_HEAD = 12
DROPOUT = 0.1

BATCH_SIZE = 16
BLOCK_SIZE = 1024
GRAD_ACCUM = 8
MAX_ITERS = 20_000
EVAL_INTERVAL = 500
EVAL_ITERS = 100

MAX_LR = 6e-4
MIN_LR = MAX_LR / 10
WARMUP_ITERS = 2_000
LR_DECAY_ITERS = MAX_ITERS

SEED = 1337
USE_GRAD_CKPT = True

BG = "#0f0f1a"
PANEL = "#1a1a2e"
GRID = "#2a2a4a"
TXT = "#e0e0f0"

CONFIGS = [
    {"name": "Full_Residual", "attn_res": True, "ffn_res": True, "color": "#00d4ff"},
    {"name": "No_Residual", "attn_res": False, "ffn_res": False, "color": "#ff6b6b"},
    {"name": "Attn_Only_Res", "attn_res": True, "ffn_res": False, "color": "#a9e34b"},
    {"name": "FFN_Only_Res", "attn_res": False, "ffn_res": True, "color": "#ffd43b"},
]

RESULTS_FILENAME = "residual_results_124M.json"
DEFAULT_BASE_DIR = "./ResidualExperiment"
DEFAULT_DATA_DIR = "./owt_cache"

def config_names() -> list[str]:
    return [c["name"] for c in CONFIGS]

def config_by_name(name: str) -> dict:
    for cfg in CONFIGS:
        if cfg["name"] == name:
            return cfg
    raise KeyError(f"Unknown config: {name}")