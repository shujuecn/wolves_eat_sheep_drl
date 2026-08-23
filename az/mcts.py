"""MCTS 与自对弈单局搜索（纯 Python，无 torch 依赖，便于多进程 fork）。

实现要点：
- 标准 AlphaZero PUCT 搜索；叶节点由外部批量评估（worker 收集后送 GPU）；
  终局节点用精确价值直接备份（不消耗网络推理）。
- 根节点展开时注入 Dirichlet 噪声；前 temp_moves 步按访问计数采样，之后 argmax。
- 走完一步后重建新树（不做子树复用），训练样本在 finalize 时落盘。
"""

from __future__ import annotations

import math
import random

import numpy as np

from . import game
from .encode import legal_mask, state_planes, visits_to_policy


class Node:
    __slots__ = ("acts", "P", "N", "W", "children")

    def __init__(self) -> None:
        self.acts: tuple[int, ...] | None = None   # 合法动作（展开后）
        self.P: list[float] | None = None          # 先验（与 acts 对齐）
        self.N: list[int] | None = None            # 边访问数
        self.W: list[float] | None = None          # 边累计价值（父回合方视角）
        self.children: dict[int, Node] = {}


class SelfPlayGame:
    """一局自对弈：协作式推进，advance() 返回待评估局面或 None。"""

    def __init__(self, sims: int, c_puct: float, dir_alpha: float,
                 dir_eps: float, temp_moves: int, rng: random.Random):
        self.sims = sims
        self.c_puct = c_puct
        self.dir_alpha = dir_alpha
        self.dir_eps = dir_eps
        self.temp_moves = temp_moves
        self.rng = rng
        self.env = game.Env()
        self.samples: list[tuple[np.ndarray, np.ndarray, int]] = []
        self.done = False
        self.outcome: int | None = None
        self._start_search()

    # ---- 生命周期 ----
    def reset(self) -> None:
        self.env.reset()
        self.samples.clear()
        self.done = False
        self.outcome = None
        self._start_search()

    def _start_search(self) -> None:
        self.search_root_state = self.env.clone_state_key()
        self.root = Node()
        self.sims_done = 0
        self.noise_added = False
        self.pending: tuple[Node, game.State, list] | None = None

    @property
    def plies(self) -> int:
        return len(self.samples)

    # ---- 协作式推进 ----
    def advance(self):
        """推进到需要一次网络评估为止；返回 (planes, mask) 或 None。

        返回 None 的情形：本步搜索已完成并落子（可能对局已结束）。
        """
        assert not self.done and self.pending is None
        while True:
            if self.sims_done >= self.sims:
                self._finalize_move()
                if self.done:
                    return None
                continue
            req = self._simulate()
            if req is None:
                self.sims_done += 1
                continue
            return req

    def _simulate(self):
        node = self.root
        s = self.search_root_state
        path: list[tuple[Node, int]] = []
        while True:
            res = game.immediate_result(s)
            if res != game.ONGOING:
                self._backup(path, game.result_for_mover(res, s))
                return None
            if node.acts is None:
                acts = game.legal_actions(s)
                planes = state_planes(s.sheep, s.wolf, s.turn)
                self.pending = (node, s, path)
                return planes, legal_mask(acts)
            # ---- PUCT 选择 ----
            n_list = node.N
            w_list = node.W
            sqrt_total = math.sqrt(sum(n_list))
            best_i = -1
            best_v = -1e18
            p_list = node.P
            cpuct = self.c_puct
            for i in range(len(n_list)):
                q = w_list[i] / n_list[i] if n_list[i] else 0.0
                val = q + cpuct * p_list[i] * sqrt_total / (1 + n_list[i])
                if val > best_v:
                    best_v = val
                    best_i = i
            a = node.acts[best_i]
            path.append((node, best_i))
            s = game.apply_action(s, a)
            child = node.children.get(a)
            if child is None:
                child = Node()
                node.children[a] = child
            node = child

    def _backup(self, path, v: float) -> None:
        for node, i in reversed(path):
            v = -v
            node.N[i] += 1
            node.W[i] += v

    def receive(self, policy: list[float], value: float) -> None:
        """网络返回后：展开叶节点（含根噪声）、备份价值。"""
        assert self.pending is not None
        node, s, path = self.pending
        self.pending = None
        acts = game.legal_actions(s)
        k = len(acts)
        pri = [max(policy[a], 0.0) for a in acts]
        total = sum(pri)
        if total <= 0:
            pri = [1.0 / k] * k
        else:
            pri = [p / total for p in pri]
        if node is self.root and not self.noise_added:
            self.noise_added = True
            noise = [self.rng.gammavariate(self.dir_alpha, 1.0) for _ in range(k)]
            ns = sum(noise)
            if ns > 0:
                noise = [x / ns for x in noise]
                eps = self.dir_eps
                pri = [(1 - eps) * p + eps * x for p, x in zip(pri, noise)]
        node.acts = acts
        node.P = pri
        node.N = [0] * k
        node.W = [0.0] * k
        self._backup(path, value)
        self.sims_done += 1

    # ---- 落子与样本 ----
    def _finalize_move(self) -> None:
        root = self.root
        acts = root.acts
        visits = root.N
        s = self.search_root_state
        planes = state_planes(s.sheep, s.wolf, s.turn)
        pi = visits_to_policy(acts, visits)
        ply = s.ply
        if ply < self.temp_moves:
            choice = self.rng.choices(list(acts), weights=visits, k=1)[0]
        else:
            mx = max(visits)
            best = [a for a, n in zip(acts, visits) if n == mx]
            choice = self.rng.choice(best)
        self.samples.append((planes, pi, s.turn))
        self.env.step(choice)
        if self.env.result != game.ONGOING:
            self.done = True
            self.outcome = self.env.result
        else:
            self._start_search()

    def flush_record(self) -> dict:
        """对局结束后导出训练样本。"""
        assert self.done and self.outcome is not None
        outcome = self.outcome
        n = len(self.samples)
        planes = np.stack([sp[0] for sp in self.samples])
        pi = np.stack([sp[1] for sp in self.samples]).astype(np.float16)
        z = np.zeros(n, dtype=np.float32)
        if outcome != game.DRAW:
            win_side = 1 if outcome == game.WOLF_WIN else 0
            for i, (_pl, _pi, side) in enumerate(self.samples):
                z[i] = 1.0 if side == win_side else -1.0
        return {
            "planes": planes,
            "pi": pi,
            "z": z,
            "result": int(outcome),
            "plies": int(n),
        }
