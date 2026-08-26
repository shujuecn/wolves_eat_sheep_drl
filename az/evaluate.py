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
        """推进一次模拟。

        返回 (planes, mask) 表示需要网络评估；
        返回 None 表示本模拟无需评估（终局精确价值备份）或搜索已完成。
        调用方用 .done 区分两种情形。
        """
        if self.done:
            return None
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


def batched_eval_searches(searches: list[EvalSearch], forward,
                          progress_every: int = 0, tag: str = ""):
    """把多个 EvalSearch 的叶请求合并成批送入 forward(planes,masks)->(pol,val)。"""
    active = [s for s in searches if not s.done]
    rounds = 0
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
        rounds += 1
        if progress_every and rounds % progress_every == 0:
            print(f"[eval] {tag} 搜索轮次 {rounds}，剩余 {len(active)}",
                  flush=True)


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
def _capture_biased_walk_positions(rng: random.Random, quota_per_k: int,
                                   max_plies: int = 4_000_000) -> list[State]:
    """狼偏好吃子的随机游走，采集低羊数（k<=11）真实可达局面。"""
    out: list[State] = []
    cnt: dict[int, int] = defaultdict(int)
    plies = 0
    while plies < max_plies:
        env = game.Env()
        while plies < max_plies:
            st = env.state
            res = game.immediate_result(st)
            if res != ONGOING or st.ply >= game.MAX_PLY:
                break
            acts = game.legal_actions(st)
            if st.turn == 1:  # 狼：一半概率只考虑跳吃
                caps = [a for a in acts if a & 1]
                if caps and rng.random() < 0.5:
                    acts = caps
            key = (st.sheep, st.wolf, st.turn)
            k = st.sheep.bit_count()
            if key not in seen_keys and k <= 11 and cnt[k] < quota_per_k:
                seen_keys.add(key)
                cnt[k] += 1
                out.append(st)
            env.step(rng.choice(acts))
            plies += 1
        if all(cnt[k] >= quota_per_k for k in range(4, 12)):
            break
    return out


seen_keys: set = set()


def sample_positions(net, device, n_target: int, seed: int,
                     n_games: int = 40, sims: int = 96,
                     max_per_k: int = 400, min_per_k: int = 80) -> list[State]:
    """分层采样：先随机+吃子偏好游走补齐各 k 配额，再用模型自对弈轨迹填充。"""
    rng = random.Random(seed)

    # 1) 无模型阶段：随机游走（含吃子偏好）保证低 k 覆盖
    pool: list[State] = []
    cnt: dict[int, int] = defaultdict(int)
    walk_pos = _capture_biased_walk_positions(rng, min_per_k * 3)
    for st in walk_pos:
        k = st.sheep.bit_count()
        if cnt[k] < max_per_k:
            cnt[k] += 1
            pool.append(st)

    # 2) 模型自对弈轨迹补充（更贴近实际对局分布）
    searches = [_PlaySearch(random.Random(seed + 7)) for _ in range(n_games)]
    import torch

    @torch.inference_mode()
    def fwd(planes: np.ndarray, masks: np.ndarray):
        x = torch.from_numpy(planes).to(device).float()
        m = torch.from_numpy(masks).to(device)
        logits, v = net(x, m)
        return torch.softmax(logits, 1).cpu().numpy(), v.cpu().numpy()

    rounds_without_progress = 0
    while True:
        idxs, planes_l, masks_l = [], [], []
        any_active = False
        for i, ps in enumerate(searches):
            if ps.done:
                continue
            any_active = True
            r = ps.advance()
            if r is not None:
                idxs.append(i)
                planes_l.append(r[0])
                masks_l.append(r[1])
        if not any_active:
            break
        if planes_l:
            pol, val = fwd(np.stack(planes_l), np.stack(masks_l))
            for j, i in enumerate(idxs):
                searches[i].receive(pol[j].tolist(), float(val[j]))
        # 收集新到达的局面
        progressed = False
        for i in idxs:
            st = searches[i].env.state
            key = (st.sheep, st.wolf, st.turn)
            if key in seen_keys or game.immediate_result(st) != ONGOING:
                continue
            k = st.sheep.bit_count()
            if cnt[k] < max_per_k:
                seen_keys.add(key)
                cnt[k] += 1
                pool.append(st)
                progressed = True
        for i, ps in enumerate(searches):
            if ps.done:
                searches[i] = _PlaySearch(random.Random(seed + 7 + i))
        # 终止条件：总量足够且没有 k 缺口
        total = len(pool)
        gaps = [k for k in range(4, 16) if cnt[k] < min_per_k]
        if total >= n_target * 2 and not gaps:
            break
        if not progressed:
            rounds_without_progress += 1
            if rounds_without_progress > 50:
                break
        else:
            rounds_without_progress = 0

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
class Continuation:
    """从给定局面开始让模型自弈到终局（协作式，可批量并发）。"""

    def __init__(self, start: State, sims: int, c_puct: float):
        self.st = start._replace(ply=0)
        self.sims = sims
        self.c_puct = c_puct
        self.search: EvalSearch | None = None
        self.done = False
        self.final: int | None = None
        self._check_terminal()

    def _check_terminal(self) -> None:
        res = game.immediate_result(self.st)
        if res != ONGOING:
            self.final = res
            self.done = True
        elif self.st.ply >= game.MAX_PLY:
            self.final = DRAW
            self.done = True

    def advance(self):
        if self.done:
            return None
        # 搜索已完成：落子并推进局面
        while self.search is not None and self.search.done:
            a = self.search.best_action()
            self.st = game.apply_action(self.st, a)
            self.search = None
            self._check_terminal()
            if self.done:
                return None
        if self.search is None:
            self.search = EvalSearch(self.st, self.sims, self.c_puct)
        r = self.search.advance()
        if r is not None:
            return r
        # r=None：本次消耗了终局模拟或搜索刚完成；交由下一轮用 done 判断
        return None


def batched_continuations(conts: list[Continuation], forward) -> None:
    active = [c for c in conts if not c.done]
    while active:
        idxs, planes_l, masks_l = [], [], []
        for i, c in enumerate(active):
            r = c.advance()
            if r is not None:
                idxs.append(i)
                planes_l.append(r[0])
                masks_l.append(r[1])
        if not planes_l:
            break
        pol, val = forward(np.stack(planes_l), np.stack(masks_l))
        for j, i in enumerate(idxs):
            active[i].search.receive(pol[j].tolist(), float(val[j]))
        active = [c for c in active if not c.done]


class VsPerfectGame:
    """模型 vs 表库完美引擎：model_side 指定模型执哪一方（1=狼 0=羊）。

    协作式推进（可批量）；模型每步用 EvalSearch(sims)，对手用表库最优应手。
    """

    def __init__(self, start: State, model_side: int, sims: int, c_puct: float,
                 tb: Tablebase):
        self.st = start._replace(ply=0)
        self.model_side = model_side
        self.sims = sims
        self.c_puct = c_puct
        self.tb = tb
        self.search: EvalSearch | None = None
        self.done = False
        self.final: int | None = None
        self._check_terminal()

    def _check_terminal(self) -> None:
        res = game.immediate_result(self.st)
        if res != ONGOING:
            self.final = res
            self.done = True
        elif self.st.ply >= game.MAX_PLY:
            self.final = DRAW
            self.done = True

    def advance(self):
        if self.done:
            return None
        # 完成上一手模型的搜索
        while self.search is not None and self.search.done:
            a = self.search.best_action()
            self.search = None
            self.st = game.apply_action(self.st, a)
            self._check_terminal()
            if self.done:
                return None
        if self.search is None:
            # 对手回合：表库完美应手
            if self.st.turn != self.model_side:
                opp = tb_perfect_action(self.tb, self.st)
                self.st = game.apply_action(self.st, opp)
                self._check_terminal()
                if self.done:
                    return None
            self.search = EvalSearch(self.st, self.sims, self.c_puct)
        r = self.search.advance()
        if r is not None:
            return r
        return None


def batched_vs_perfect(games: list[VsPerfectGame], forward) -> None:
    active = [g for g in games if not g.done]
    while active:
        idxs, planes_l, masks_l = [], [], []
        for i, g in enumerate(active):
            r = g.advance()
            if r is not None:
                idxs.append(i)
                planes_l.append(r[0])
                masks_l.append(r[1])
        if not planes_l:
            break
        pol, val = forward(np.stack(planes_l), np.stack(masks_l))
        for j, i in enumerate(idxs):
            active[i].search.receive(pol[j].tolist(), float(val[j]))
        active = [g for g in active if not g.done]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/models/latest.pt")
    ap.add_argument("--tb-dir", default=None)
    ap.add_argument("--n-pos", type=int, default=2000)
    ap.add_argument("--sims", type=int, default=600)
    ap.add_argument("--c-puct", type=float, default=1.7)
    ap.add_argument("--n-match", type=int, default=300, help="自弈达成率抽样数")
    ap.add_argument("--cont-sims", type=int, default=200, help="自弈达成率每步模拟数")
    ap.add_argument("--vp-sims", type=int, default=400, help="对抗完美引擎时模型每步模拟数")
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
    # 分块搜索 + 逐块统计（抗长跑中断，可观测进度）
    import faulthandler
    faulthandler.enable()
    CHUNK = 240
    n_opt = n_strict = n_value = 0
    per_k = defaultdict(lambda: [0, 0, 0, 0])
    cls_dist = defaultdict(int)
    done_rows: list[dict] = []
    t_search = time.time()
    for c0 in range(0, len(positions), CHUNK):
        chunk = positions[c0:c0 + CHUNK]
        searches = [EvalSearch(s, args.sims, args.c_puct) for s in chunk]
        batched_eval_searches(searches, forward,
                              progress_every=100,
                              tag=f"chunk{c0//CHUNK}")
        for s, se in zip(chunk, searches):
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
            done_rows.append({"s": s, "cls": cls})
        el = time.time() - t_search
        n_done = len(done_rows)
        print(f"[eval] 进度 {n_done}/{len(positions)} "
              f"最优={100*n_opt/n_done:.2f}% 严格={100*n_strict/n_done:.2f}% "
              f"价值={100*n_value/n_done:.2f}% （{el:.0f}s）", flush=True)
        # 增量落盘，中断不丢
        with open(os.path.join(args.out_dir, f"partial_{args.tag or 'eval'}.json"), "w") as f:
            json.dump({"done": n_done, "opt": n_opt, "strict": n_strict,
                       "value": n_value}, f)

    n = len(done_rows)
    m_opt = n_opt / n
    m_strict = n_strict / n
    m_value = n_value / n

    # ---- 自弈达成率 ----
    rng = random.Random(args.seed + 1)
    match_idx = rng.sample(range(n), min(args.n_match, n))
    conts = [Continuation(done_rows[i]["s"], args.cont_sims, args.c_puct) for i in match_idx]
    t0 = time.time()
    batched_continuations(conts, forward)
    n_ach = 0
    ach_by_cls = defaultdict(lambda: [0, 0])  # cls -> [n, ach]
    for c, i in zip(conts, match_idx):
        cls = done_rows[i]["cls"]
        got = mover_class(c.final, done_rows[i]["s"])
        ok = got == cls
        n_ach += ok
        ach_by_cls[cls][0] += 1
        ach_by_cls[cls][1] += ok
    m_achieve = n_ach / len(match_idx)
    print(f"[eval] 自弈达成率阶段完成：{time.time()-t0:.0f}s", flush=True)
    for c in (1, 0, -1):
        nn, aa = ach_by_cls[c]
        if nn:
            name = {1: "胜面", 0: "和面", -1: "负面"}[c]
            print(f"[eval]   真值{name}: {aa}/{nn} = {100*aa/nn:.1f}%", flush=True)

    # ---- 对抗硬解完美引擎（能力相似的最直接度量）----
    vp_n = min(args.n_match, n)
    vp_idx = rng.sample(range(n), vp_n)
    t0 = time.time()
    res_by_mode = {"model_mover": [0, 0], "model_defender": [0, 0]}
    for mode, model_side_pick in (("model_mover", None), ("model_defender", None)):
        games = []
        metas = []
        for i in vp_idx:
            s = done_rows[i]["s"]
            known, abs_r, _d = tb.lookup_state(s)
            mover_is_wolf = s.turn == 1
            # 模型执"表库结论的胜方"（胜面时）或回合方（和/负时按表库最优应对）
            if mode == "model_mover":
                ms = s.turn
            else:
                ms = 0 if s.turn == 1 else 1
            games.append(VsPerfectGame(s, ms, args.vp_sims, args.c_puct, tb))
            metas.append((mover_class(abs_r, s), s))
        batched_vs_perfect(games, forward)
        for g, (cls, s) in zip(games, metas):
            got = mover_class(g.final, s)
            ok = got == cls
            res_by_mode[mode][0] += 1
            res_by_mode[mode][1] += ok
        print(f"[eval] {mode}: {res_by_mode[mode][1]}/{res_by_mode[mode][0]} "
              f"= {100*res_by_mode[mode][1]/max(1,res_by_mode[mode][0]):.1f}% "
              f"（{time.time()-t0:.0f}s）", flush=True)

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
