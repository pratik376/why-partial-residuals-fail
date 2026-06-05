#!/bin/bash
set -e

python train.py --config Full_Residual
python train.py --config No_Residual
python train.py --config Attn_Only_Res
python train.py --config FFN_Only_Res

python plot_results.py
echo "All done — check figures/124M/"