"""训练超参数配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    # 路径
    root: str = field(default_factory=_repo_root)
    model_dir: str = "data/models"
    log_dir: str = "logs"
    tb_dir: str = "data/ws_tb_dtc_v2_c"   # 硬解表库目录（评估用）

    # 自对弈
    actors: int = 20            # 自对弈进程数
    games_per_actor: int = 24   # 每进程并发的对局数（lockstep 批推理粒度）
    sims: int = 160             # 每步 MCTS 模拟次数
    c_puct: float = 1.7
    dir_alpha: float = 0.55     # 根节点 Dirichlet 噪声 α
    dir_eps: float = 0.25
    temp_moves: int = 14        # 前 N 步 τ=1 采样，之后 argmax

    # 训练
    batch_size: int = 512
    steps_per_iter: int = 320   # 每积累 iter_games 局训练的 minibatch 步数
    lr: float = 1.5e-3
    lr_min: float = 1.0e-4
    cosine_steps: int = 60 * 320
    weight_decay: float = 1.0e-4
    value_loss_weight: float = 1.0
    buffer_size: int = 400_000  # 样本回放池容量（按局面计）
    min_buffer: int = 25_000    # 开始训练前的最少样本量
    max_grad_norm: float = 2.0
    augment_flip: bool = True   # 水平镜像数据增广（规则不变变换）

    # 网络
    blocks: int = 6
    channels: int = 64

    # 流程
    iter_games: int = 384       # 每迭代目标对局数
    total_iters: int = 60
    save_every_iters: int = 4
    seed: int = 42

    # ---- 推导路径 ----
    def path_model_dir(self) -> str:
        p = self.model_dir
        return p if os.path.isabs(p) else os.path.join(self.root, p)

    def path_log_dir(self) -> str:
        p = self.log_dir
        return p if os.path.isabs(p) else os.path.join(self.root, p)

    def path_tb(self) -> str:
        p = self.tb_dir
        return p if os.path.isabs(p) else os.path.join(self.root, p)
