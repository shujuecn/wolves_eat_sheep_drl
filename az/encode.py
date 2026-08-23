"""状态编码与动作空间变换（含水平镜像增广）。"""

from __future__ import annotations

import numpy as np

from .game import CELLS, NUM_ACTIONS, PREY

PLANES = 3


def state_planes(sheep: int, wolf: int, turn: int) -> np.ndarray:
    """(3,5,5) uint8：0=羊位板，1=狼位板，2=行棋方（狼=全 1）。"""
    x = np.zeros((PLANES, 5, 5), dtype=np.uint8)
    bb = sheep
    while bb:
        lsb = bb & (-bb)
        i = lsb.bit_length() - 1
        x[0, i // 5, i % 5] = 1
        bb ^= lsb
    bb = wolf
    while bb:
        lsb = bb & (-bb)
        i = lsb.bit_length() - 1
        x[1, i // 5, i % 5] = 1
        bb ^= lsb
    if turn == 1:
        x[2, :, :] = 1
    return x


def flip_action(a: int) -> int:
    """水平镜像（列 c -> 4-c）下的动作像。方向左右互换（2<->3）。"""
    frm = a >> 3
    d = (a >> 1) & 3
    cap = a & 1
    r, c = frm // 5, frm % 5
    nfrm = r * 5 + (4 - c)
    nd = d if d < 2 else 5 - d
    return (nfrm << 3) | (nd << 1) | cap


# F[a] = flip_action(a)；镜像策略向量 out = pi[:, F]（F 为对合）
FLIP_PERM = np.array([flip_action(a) for a in range(NUM_ACTIONS)], dtype=np.int64)

ACTION_FROM = np.array([a >> 3 for a in range(NUM_ACTIONS)], dtype=np.int64)


def legal_mask(acts) -> np.ndarray:
    m = np.zeros(NUM_ACTIONS, dtype=bool)
    if acts:
        m[list(acts)] = True
    return m


def visits_to_policy(acts, visits) -> np.ndarray:
    """访问计数 -> 稀疏归一化策略向量（200 维 float32，仅合法动作非零）。"""
    pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
    total = float(sum(visits))
    for a, n in zip(acts, visits):
        pi[a] = n / total
    return pi


def flip_planes(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x[:, :, ::-1])


def sample_flip_policy(pi: np.ndarray) -> np.ndarray:
    return pi[FLIP_PERM]
