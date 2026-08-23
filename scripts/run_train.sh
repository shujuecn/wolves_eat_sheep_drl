#!/usr/bin/env bash
# 启动/续训 AlphaZero（后台 nohup，日志 logs/train_full.log）
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch
mkdir -p logs data/models

ARGS=("--actors" "${ACTORS:-20}" "--games-per-actor" "${GPA:-24}"
      "--sims" "${SIMS:-160}" "--iters" "${ITERS:-60}")
if [[ "$1" == "resume" ]]; then ARGS+=("--resume"); fi

nohup python -m az.train "${ARGS[@]}" > logs/train_full.log 2>&1 &
echo "训练已启动 pid=$!  日志: logs/train_full.log"
