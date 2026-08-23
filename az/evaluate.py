"""对拍硬解最优解的评估器。

采样真实可达局面（模型自对弈轨迹去重），逐局面对照硬解表库：

指标：
1) m_opt       —— 模型首选走法落在「价值保持最优集」内的比例；
2) m_strict    —— 落在「DTM 严格最优集」内的比例（胜方最快将杀/败方最长抵抗）；
3) m_value     —— 根价值估计与表库结论（胜/和/负）一致的比例；
4) m_achieve   —— 从该局面出发让模型自弈到终局，实际结果达到表库结论的比例。

用法：
  python -m az.evaluate --ckpt data/models/latest.pt --n-pos 2000 --sims 600
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np

from . import game
from .encode import legal_mask, state_planes
from .game import DRAW, ONGOING, SHEEP_WIN, State, WOLF_WIN
from .mcts import Node
from .tablebase import Tablebase, load_tablebase


# ----------------------------------------------------------------
# 协作式评估搜索（静态局面，无环境、无噪声，最终取 argmax）
# ----------------------------------------------------------------
class EvalSearch:
    def __init__(self, start: State, sims: int, c_puct: float):
        self.start = start
        self.sims = sims
        self.c_puct = c_puct
        self.root = Node()
        self.sims_done = 0
        self.pending: tuple[Node, State, list] | None = None

    @property
    def done(self) -> bool:
        return self.sims_done >= self.sims and self.pending is None

    def advance(self):
        assert not self.done
        while True:
            if self.sims_done >= self.sims:
                return None
            req = self._simulate()
            if req is None:
                self.sims_done += 1
                continue
            return req

    def _simulate(self):
        node = self.root
        s = self.start
        path: list[tuple[Node, int]] = []
        while True:
            res = game.immediate_result(s)
            if res != ONGOING:
                v = game.result_for_mover(res, s)
                for nd, i in reversed(path):
                    v = -v
                    nd.N[i] += 1
                    nd.W[i] += v
                return None
            if node.acts is None:
                acts = game.legal_actions(s)
                self.pending = (node, s, path)
                return state_planes(s.sheep, s.wolf, s.turn), legal_mask(acts)
            n_list, w_list, p_list = node.N, node.W, node.P
            sqrt_total = math.sqrt(sum(n_list))
            best_i, best_v = -1, -1e18
            for i in range(len(n_list)):
                q = w_list[i] / n_list[i] if n_list[i] else 0.0
                val = q + self.c_puct * p_list[i] * sqrt_total / (1 + n_list[i])
                if val > best_v:
                    best_v, best_i = val, i
            a = node.acts[best_i]
            path.append((node, best_i))
            s = game.apply_action(s, a)
            child = node.children.get(a)
            if child is None:
                child = Node()
                node.children[a] = child
            node = child

    def receive(self, policy: list[float], value: float) -> None:
        node, s, path = self.pending
        self.pending = None
        acts = game.legal_actions(s)
        k = len(acts)
        pri = [max(policy[a], 0.0) for a in acts]
        tot = sum(pri)
        pri = [p / tot for p in pri] if tot > 0 else [1.0 / k] * k
        node.acts = acts
        node.P = pri
        node.N = [0] * k
        node.W = [0.0] * k
        v = value
        for nd, i in reversed(path):
            v = -v
            nd.N[i] += 1
            nd.W[i] += v
        self.sims_done += 1

    def best_action(self) -> int:
        acts, visits = self.root.acts, self.root.N
        mx = max(visits)
        best = [a for a, n in zip(acts, visits) if n == mx]
        return best[0]

    def root_value(self) -> float:
        return sum(self.root.W) / max(sum(self.root.N), 1)


def batched_eval_searches(searches: list[EvalSearch], forward, log_every=0):
    """把多个 EvalSearch 的叶请求合并成批送入 forward(planes,masks)->(pol,val)。"""
    active = [s for s in searches if not s.done]
    while active:
        idxs, planes_l, masks_l = [], [], []
        for i, se in enumerate(active):
            r = se.advance()
            if r is not None:
                idxs.append(i)
                planes_l.append(r[0])
                masks_l.append(r[1])
        if not planes_l:
            break
        pol, val = forward(np.stack(planes_l), np.stack(masks_l))
        for j, i in enumerate(idxs):
            active[i].receive(pol[j].tolist(), float(val[j]))
        active = [s for s in active if not s.done]
        if log_every and random.random() < log_every:
            pass


def forward_factory(net, device):
    import torch

    @torch.inference_mode()
    def forward(planes: np.ndarray, masks: np.ndarray):
        x = torch.from_numpy(planes).to(device).float()
        m = torch.from_numpy(masks).to(device)
        logits, v = net(x, m)
        return torch.softmax(logits, 1).cpu().numpy(), v.cpu().numpy()

    return forward


# ----------------------------------------------------------------
# 表库查询辅助
# ----------------------------------------------------------------
def tb_child_info(tb: Tablebase, s: State, a: int):
    """走一步后的 (abs_result, dist)。终局直接判定；否则查表。"""
    ns = game.apply_action(s, a)
    res = game.immediate_result(ns)
    if res != ONGOING:
        return int(res), 0
    known, r, d = tb.lookup_state(ns)
    if not known:
        return None, -1
    return int(r), int(d)


def mover_class(abs_result: int, s: State) -> int:
    """回合方视角：+1 胜 / 0 和 / -1 负。"""
    if abs_result == DRAW:
        return 0
    mover_wins = (abs_result == WOLF_WIN) == (s.turn == 1)
    return 1 if mover_wins else -1


def optimal_sets(tb: Tablebase, s: State):
    """返回 (value_set, strict_set, cls, dist)：两个最优动作集合与表库结论。"""
    known, abs_r, dist = tb.lookup_state(s)
    assert known, f"position not in tablebase: {s}"
    cls = mover_class(abs_r, s)
    my_abs_win = WOLF_WIN if s.turn == 1 else SHEEP_WIN
    value_set, strict_set = [], []
    best_loss_dist = -1
    child_infos = []
    for a in game.legal_actions(s):
        r, d = tb_child_info(tb, s, a)
        if r is None:
            continue
        child_infos.append((a, r, d))
    # 败方的最长抵抗距离
    if cls == -1:
        best_loss_dist = max(d for _a, _r, d in child_infos)
    for a, r, d in child_infos:
        if cls == 1:
            if r != my_abs_win:
                continue
            value_set.append(a)
            if d == dist - 1:
                strict_set.append(a)
        elif cls == 0:
            if r == DRAW:
                value_set.append(a)
                strict_set.append(a)
        else:
            value_set.append(a)
            if d == best_loss_dist:
                strict_set.append(a)
    return value_set, strict_set, cls, dist


def tb_perfect_action(tb: Tablebase, s: State) -> int:
    """表库引擎应手（胜>和>负；胜取最短 DTN，负取最长抵抗）。"""
    my_abs_win = WOLF_WIN if s.turn == 1 else SHEEP_WIN
    best_a, best_key = None, None
    res_now = game.immediate_result(s)
    assert res_now == ONGOING
    for a in game.legal_actions(s):
        r, d = tb_child_info(tb, s, a)
        if r is None:
            continue
        rank = 2 if r == my_abs_win else (1 if r == DRAW else 0)
        key = (rank, -d if rank == 2 else (d if rank == 0 else 0))
        if best_key is None or key > best_key:
            best_key, best_a = key, a
    assert best_a is not None
    return best_a


# ----------------------------------------------------------------
# 局面采样：模型自对弈轨迹去重
# ----------------------------------------------------------------
def sample_positions(net, device, n_target: int, seed: int,
                     n_games: int = 40, sims: int = 96,
                     max_per_k: int = 400) -> list[State]:
    rng = random.Random(seed)
    searches = [_PlaySearch(rng) for _ in range(n_games)]
    seen: set[tuple] = set()
    bucket: dict[int, list] = defaultdict(list)

    import torch

    @torch.inference_mode()
    def fwd(planes: np.ndarray, masks: np.ndarray):
        x = torch.from_numpy(planes).to(device).float()
        m = torch.from_numpy(masks).to(device)
        logits, v = net(x, m)
        return torch.softmax(logits, 1).cpu().numpy(), v.cpu().numpy()

    while any(not ps.done for ps in searches):
        idxs, planes_l, masks_l = [], [], []
        for i, ps in enumerate(searches):
            if ps.done:
                continue
            r = ps.advance()
            if r is not None:
                idxs.append(i)
                planes_l.append(r[0])
                masks_l.append(r[1])
        if not planes_l:
            break
        pol, val = fwd(np.stack(planes_l), np.stack(masks_l))
        for j, i in enumerate(idxs):
            searches[i].receive(pol[j].tolist(), float(val[j]))
        # 收集刚到达的局面（去重 + 分桶）
        for i in idxs:
            st = searches[i].env.state
            key = (st.sheep, st.wolf, st.turn)
            if key in seen or game.immediate_result(st) != ONGOING:
                continue
            k = st.sheep.bit_count()
            if len(bucket[k]) >= max_per_k:
                continue
            seen.add(key)
            bucket[k].append(st)
        # 完局重开
        for i, ps in enumerate(searches):
            if ps.done:
                searches[i] = _PlaySearch(rng)

    pool = [s for ks in bucket.values() for s in ks]
    rng.shuffle(pool)
    return pool[:n_target]


class _PlaySearch:
    """轻量自对弈（用于采样局面）：复用 SelfPlayGame。"""

    def __init__(self, rng):
        from .mcts import SelfPlayGame
        self.g = SelfPlayGame(sims=96, c_puct=1.7, dir_alpha=0.55, dir_eps=0.25,
                              temp_moves=12, rng=rng)
        self.env = self.g.env

    @property
    def done(self):
        return self.g.done

    def advance(self):
        return self.g.advance()

    def receive(self, policy, value):
        self.g.receive(policy, value)


# ----------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/models/latest.pt")
    ap.add_argument("--tb-dir", default=None)
    ap.add_argument("--n-pos", type=int, default=2000)
    ap.add_argument("--sims", type=int, default=600)
    ap.add_argument("--c-puct", type=float, default=1.7)
    ap.add_argument("--n-match", type=int, default=300, help="自弈达成率抽样数")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import torch
    from .network import AlphaZeroNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    net = AlphaZeroNet(ck.get("cfg_blocks", 6), ck.get("cfg_channels", 64)).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"[eval] 已加载 {args.ckpt}（iter={ck.get('iter')} steps={ck.get('steps')} "
          f"games={ck.get('games_total')}）device={device}", flush=True)

    tb = load_tablebase(args.tb_dir) if args.tb_dir else load_tablebase(None)

    print(f"[eval] 采样 {args.n_pos} 个可达局面 ...", flush=True)
    t0 = time.time()
    positions = sample_positions(net, device, args.n_pos, args.seed)
    print(f"[eval] 采样完成：{len(positions)} 局面，{time.time()-t0:.0f}s", flush=True)

    forward = forward_factory(net, device)
    searches = [EvalSearch(s, args.sims, args.c_puct) for s in positions]
    t0 = time.time()
    batched_eval_searches(searches, forward)
    print(f"[eval] MCTS({args.sims} sims) 完成：{time.time()-t0:.0f}s", flush=True)

    # ---- 逐局面指标 ----
    n_opt = n_strict = n_value = 0
    per_k = defaultdict(lambda: [0, 0, 0, 0])  # k -> [n, opt, strict, value]
    cls_dist = defaultdict(int)
    for s, se in zip(positions, searches):
        value_set, strict_set, cls, dist = optimal_sets(tb, s)
        a_star = se.best_action()
        v_est = se.root_value()
        ok_opt = a_star in value_set
        ok_strict = a_star in strict_set
        v_cls = 1 if v_est > 0.33 else (-1 if v_est < -0.33 else 0)
        ok_val = v_cls == cls
        n_opt += ok_opt
        n_strict += ok_strict
        n_value += ok_val
        row = per_k[s.sheep.bit_count()]
        row[0] += 1
        row[1] += ok_opt
        row[2] += ok_strict
        row[3] += ok_val
        cls_dist[cls] += 1

    n = len(positions)
    m_opt = n_opt / n
    m_strict = n_strict / n
    m_value = n_value / n

    # ---- 自弈达成率 ----
    rng = random.Random(args.seed + 1)
    match_idx = rng.sample(range(n), min(args.n_match, n))
    n_ach = 0
    for i in match_idx:
        s = positions[i]
        _, _, cls, _dist = optimal_sets(tb, s)
        # 纯规则推进（表库不建模重复规则，与求解器口径一致）
        st = s._replace(ply=0)
        final = DRAW
        while True:
            res = game.immediate_result(st)
            if res != ONGOING:
                final = res
                break
            if st.ply >= game.MAX_PLY:
                final = DRAW
                break
            se = EvalSearch(st, 200, args.c_puct)
            batched_eval_searches([se], forward)
            st = game.apply_action(st, se.best_action())
        got_cls = mover_class(final, s)
        n_ach += got_cls == cls
    m_achieve = n_ach / len(match_idx)

    # ---- 报告 ----
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or os.path.basename(args.ckpt).replace(".pt", "")
    report = {
        "ckpt": args.ckpt,
        "iter": ck.get("iter"),
        "steps": ck.get("steps"),
        "games": ck.get("games_total"),
        "n_positions": n,
        "sims": args.sims,
        "metrics": {
            "move_in_optimal_set": m_opt,
            "move_in_strict_dtm_set": m_strict,
            "value_class_accuracy": m_value,
            "outcome_achievement": m_achieve,
        },
        "headline_similarity_pct": round(100 * (0.5 * (m_opt + m_achieve)), 2),
        "class_distribution": dict(cls_dist),
        "per_k": {
            str(k): {
                "n": row[0],
                "opt": row[1] / row[0],
                "strict": row[2] / row[0],
                "value": row[3] / row[0],
            }
            for k, row in sorted(per_k.items())
        },
    }
    jpath = os.path.join(args.out_dir, f"eval_{tag}.json")
    with open(jpath, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# 硬解最优解相似度评估",
        "",
        f"- checkpoint: `{args.ckpt}` (iter={report['iter']}, steps={report['steps']}, "
        f"games={report['games']})",
        f"- 局面数: {n}（自对弈轨迹去重采样），MCTS sims={args.sims}",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 最优走法命中率（价值保持集） | {100*m_opt:.2f}% |",
        f"| 最优走法命中率（严格 DTM 集） | {100*m_strict:.2f}% |",
        f"| 价值结论一致率（胜/和/负） | {100*m_value:.2f}% |",
        f"| 自弈结果达成率 | {100*m_achieve:.2f}% |",
        f"| **综合相似度** | **{report['headline_similarity_pct']}%** |",
        "",
        "## 分羊数统计",
        "",
        "| 羊数 k | 局面 | 最优命中 | 严格命中 | 价值一致 |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, row in sorted(per_k.items()):
        lines.append(
            f"| {k} | {row[0]} | {100*row[1]/row[0]:.1f}% | "
            f"{100*row[2]/row[0]:.1f}% | {100*row[3]/row[0]:.1f}% |")
    mpath = os.path.join(args.out_dir, f"eval_{tag}.md")
    with open(mpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[eval] 报告已写入 {jpath} 与 {mpath}")


if __name__ == "__main__":
    main()
