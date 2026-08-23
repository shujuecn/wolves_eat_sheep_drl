"""自对弈 actor 进程：纯 CPU 跑 MCTS，网络评估走主进程推理服务。"""

from __future__ import annotations

import random
import time

import multiprocessing as mp

import numpy as np

from .mcts import SelfPlayGame


def actor_main(actor_id: int,
               sims: int, c_puct: float, dir_alpha: float, dir_eps: float,
               temp_moves: int, games_per_actor: int, seed: int,
               req_q, res_q, games_q) -> None:
    rng = random.Random(seed * 100003 + 7)
    games = [
        SelfPlayGame(sims, c_puct, dir_alpha, dir_eps, temp_moves, rng)
        for _ in range(games_per_actor)
    ]
    games_done = 0
    while True:
        idxs: list[int] = []
        planes_list: list[np.ndarray] = []
        masks_list: list[np.ndarray] = []
        for gi, g in enumerate(games):
            if g.done:
                rec = g.flush_record()
                rec["actor"] = actor_id
                games_q.put(rec)
                games_done += 1
                g.reset()
            r = g.advance()
            if r is not None:
                idxs.append(gi)
                planes_list.append(r[0])
                masks_list.append(r[1])
        if not planes_list:
            time.sleep(0.0005)
            continue
        batch = np.stack(planes_list)
        masks = np.stack(masks_list)
        rid = rng.getrandbits(62)
        req_q.put((actor_id, rid, (batch, masks)))
        # 阻塞等待本批结果（服务端必答）
        while True:
            aid, r2, pol, val = res_q.get()
            if aid == actor_id and r2 == rid:
                break
        for j, gi in enumerate(idxs):
            games[gi].receive(pol[j].tolist(), float(val[j]))


def spawn_actors(cfg_dict: dict, req_q, res_qs: dict, games_q):
    procs = []
    for aid in range(cfg_dict["actors"]):
        p = mp.Process(
            target=actor_main,
            args=(
                aid,
                cfg_dict["sims"],
                cfg_dict["c_puct"],
                cfg_dict["dir_alpha"],
                cfg_dict["dir_eps"],
                cfg_dict["temp_moves"],
                cfg_dict["games_per_actor"],
                cfg_dict["seed"] + aid,
                req_q,
                res_qs[aid],
                games_q,
            ),
            daemon=True,
        )
        p.start()
        procs.append(p)
    return procs
