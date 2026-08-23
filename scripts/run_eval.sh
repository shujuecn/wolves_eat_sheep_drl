#!/usr/bin/env bash
# 对照硬解表库评估（需要 data/ws_tb_dtc_v2_c/ 就绪）
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

CKPT="${1:-data/models/latest.pt}"
python -m az.evaluate --ckpt "$CKPT" --n-pos "${NPOS:-2000}" --sims "${SIMS:-600}" \
    --n-match "${NMATCH:-300}" --out-dir results
