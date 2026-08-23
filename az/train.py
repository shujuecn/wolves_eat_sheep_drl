"""AlphaZero 训练主控。

架构：
- 本进程：GPU 推理服务（为所有 actor 的 MCTS 叶节点批量评估）+ 回放池 +
  训练 + checkpoint；
- actor 进程（az.selfplay）：纯 CPU 自对弈，评估请求经 mp.Queue 往返，
  完整对局记录经 games_q 送回。

用法：
  python -m az.train --iters 60 --actors 20 --sims 160
  python -m az.train --resume        # 从 data/models/latest.pt 续训
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import queue as pyqueue
import random
import signal
import time

import multiprocessing as mp

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actors", type=int, default=None)
    ap.add_argument("--games-per-actor", type=int, default=None)
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--iter-games", type=int, default=None)
    ap.add_argument("--steps-per-iter", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--min-buffer", type=int, default=None)
    ap.add_argument("--buffer-size", type=int, default=None)
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--channels", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="忽略已有 checkpoint 从零开始")
    args = ap.parse_args()

    from .config import Config
    cfg = Config()
    if args.actors is not None:
        cfg.actors = args.actors
    if args.games_per_actor is not None:
        cfg.games_per_actor = args.games_per_actor
    if args.sims is not None:
        cfg.sims = args.sims
    if args.iters is not None:
        cfg.total_iters = args.iters
    if args.iter_games is not None:
        cfg.iter_games = args.iter_games
    if args.steps_per_iter is not None:
        cfg.steps_per_iter = args.steps_per_iter
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.min_buffer is not None:
        cfg.min_buffer = args.min_buffer
    if args.buffer_size is not None:
        cfg.buffer_size = args.buffer_size
    if args.blocks is not None:
        cfg.blocks = args.blocks
    if args.channels is not None:
        cfg.channels = args.channels
    cfg.cosine_steps = cfg.total_iters * cfg.steps_per_iter

    os.makedirs(cfg.path_model_dir(), exist_ok=True)
    os.makedirs(cfg.path_log_dir(), exist_ok=True)
    latest_pt = os.path.join(cfg.path_model_dir(), "latest.pt")

    # ---- 先起 actor（fork，避免 CUDA 初始化后的 fork 风险）----
    ctx = mp.get_context("fork")
    req_q = ctx.Queue(maxsize=4096)
    games_q = ctx.Queue(maxsize=512)
    res_qs = {aid: ctx.Queue(maxsize=256) for aid in range(cfg.actors)}
    cfg_dict = {
        "actors": cfg.actors,
        "sims": cfg.sims,
        "c_puct": cfg.c_puct,
        "dir_alpha": cfg.dir_alpha,
        "dir_eps": cfg.dir_eps,
        "temp_moves": cfg.temp_moves,
        "games_per_actor": cfg.games_per_actor,
        "seed": cfg.seed,
    }
    from .selfplay import spawn_actors
    procs = spawn_actors(cfg_dict, req_q, res_qs, games_q)
    print(f"[train] 已启动 {len(procs)} 个自对弈 actor"
          f"（每进程 {cfg.games_per_actor} 局并发，{cfg.sims} sims/步）", flush=True)

    # ---- torch 相关在 fork 之后导入 ----
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW

    from .encode import FLIP_PERM
    from .network import AlphaZeroNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AlphaZeroNet(cfg.blocks, cfg.channels).to(device)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start_iter = 0
    steps_done = 0
    games_total = 0

    if args.resume and os.path.exists(latest_pt):
        ck = torch.load(latest_pt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_iter = ck["iter"]
        steps_done = ck["steps"]
        games_total = ck.get("games_total", 0)
        print(f"[train] 续训：iter={start_iter} steps={steps_done} games={games_total}",
              flush=True)

    # ---- 回放池（环形数组）----
    cap = cfg.buffer_size
    buf_planes = np.zeros((cap, 3, 5, 5), dtype=np.uint8)
    buf_pi = np.zeros((cap, 200), dtype=np.float16)
    buf_mask = np.zeros((cap, 200), dtype=bool)
    buf_z = np.zeros(cap, dtype=np.float32)
    buf_len = 0
    buf_pos = 0

    rng = np.random.default_rng(cfg.seed)

    def ingest(rec: dict) -> None:
        nonlocal buf_len, buf_pos, games_total
        from . import game as _g
        n = int(rec["plies"])
        planes = rec["planes"]
        pi = rec["pi"].astype(np.float32)
        z = rec["z"]
        masks = np.zeros((n, 200), dtype=bool)
        for i in range(n):
            x = planes[i]
            sheep = 0
            wolf = 0
            for r in range(5):
                row_s = x[0, r]
                row_w = x[1, r]
                for c in range(5):
                    b = 1 << (r * 5 + c)
                    if row_s[c]:
                        sheep |= b
                    elif row_w[c]:
                        wolf |= b
            turn = 1 if x[2, 0, 0] else 0
            st = _g.State(sheep, wolf, turn, 0)
            for a in _g.legal_actions(st):
                masks[i, a] = True
        if buf_pos + n <= cap:
            sl = slice(buf_pos, buf_pos + n)
            buf_planes[sl] = planes
            buf_pi[sl] = pi
            buf_mask[sl] = masks
            buf_z[sl] = z
        else:  # 环形回绕
            first = cap - buf_pos
            rest = n - first
            buf_planes[buf_pos:] = planes[:first]
            buf_pi[buf_pos:] = pi[:first]
            buf_mask[buf_pos:] = masks[:first]
            buf_z[buf_pos:] = z[:first]
            buf_planes[:rest] = planes[first:]
            buf_pi[:rest] = pi[first:]
            buf_mask[:rest] = masks[first:]
            buf_z[:rest] = z[first:]
        buf_pos = (buf_pos + n) % cap
        buf_len = min(cap, buf_len + n)
        games_total += 1

    log_path = os.path.join(cfg.path_log_dir(), "train_log.csv")
    new_log = not os.path.exists(log_path)
    log_f = open(log_path, "a", newline="")
    logger = csv.writer(log_f)
    if new_log:
        logger.writerow(["t", "iter", "steps", "games_total", "buffer",
                         "pol_loss", "val_loss", "lr", "games_rate"])

    stop_flag = {"stop": False}

    def _sigint(_sig, _frm):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    target_steps = cfg.total_iters * cfg.steps_per_iter
    bs = cfg.batch_size
    model.train()
    t0 = time.time()
    last_games_total = games_total
    last_t = t0
    next_ckpt_iter = start_iter
    last_pl = float("nan")
    last_vl = float("nan")
    lr = cfg.lr

    print(f"[train] 开始训练：目标 {cfg.total_iters} iters / {target_steps} steps；"
          f"device={device}", flush=True)

    while not stop_flag["stop"]:
        # ---- 1) 推理服务：收集一批评估请求 ----
        batch_items = []
        try:
            while len(batch_items) < 512:
                batch_items.append(req_q.get_nowait())
        except pyqueue.Empty:
            pass
        if batch_items:
            planes_np = np.concatenate([it[2][0] for it in batch_items], axis=0)
            masks_np = np.concatenate([it[2][1] for it in batch_items], axis=0)
            planes_t = torch.from_numpy(planes_np)
            masks_t = torch.from_numpy(masks_np)
            logits, values = model.infer_batch(planes_t, masks_t)
            pol = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            val = values.cpu().numpy().astype(np.float32)
            by_actor: dict[int, list[int]] = {}
            row = 0
            for aid, rid, payload in batch_items:
                n = payload[0].shape[0]
                by_actor.setdefault(aid, []).append((rid, row, row + n))
                row += n
            for aid, spans in by_actor.items():
                rid0 = spans[0][0]
                rows = []
                for _r, a, b in spans:
                    rows.extend(range(a, b))
                res_qs[aid].put((aid, rid0, pol[rows], val[rows]))

        # ---- 2) 收对局 ----
        try:
            while True:
                rec = games_q.get_nowait()
                ingest(rec)
        except pyqueue.Empty:
            pass

        # ---- 3) 训练步（按对局到达速度配速）----
        desired = games_total * cfg.steps_per_iter // cfg.iter_games
        done_here = 0
        while (steps_done < target_steps and done_here < 8
               and desired > steps_done and buf_len >= cfg.min_buffer):
            idx = rng.integers(0, buf_len, size=bs)
            x = buf_planes[idx]
            pi_t = buf_pi[idx].astype(np.float32)
            m_t = buf_mask[idx]
            z_t = buf_z[idx]
            if cfg.augment_flip:
                flip = rng.random(bs) < 0.5
                if flip.any():
                    fi = np.nonzero(flip)[0]
                    x = x.copy()
                    x[fi] = x[fi][:, :, :, ::-1]
                    pi_t[fi] = pi_t[fi][:, FLIP_PERM]
                    m_t = m_t.copy()
                    m_t[fi] = m_t[fi][:, FLIP_PERM]
            xt = torch.from_numpy(np.ascontiguousarray(x)).to(device)
            mt = torch.from_numpy(m_t).to(device)
            pit = torch.from_numpy(pi_t).to(device)
            zt = torch.from_numpy(z_t).to(device)

            lr = (cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min)
                  * (1 + math.cos(math.pi * min(1.0, steps_done / cfg.cosine_steps))))
            for gp in opt.param_groups:
                gp["lr"] = lr
            logits, v = model(xt.float(), mt)
            logp = F.log_softmax(logits, dim=1)
            pol_loss = -(pit * logp).sum(dim=1).mean()
            val_loss = F.mse_loss(v, zt)
            loss = pol_loss + cfg.value_loss_weight * val_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            steps_done += 1
            done_here += 1
            last_pl = float(pol_loss.item())
            last_vl = float(val_loss.item())

        # ---- 4) 迭代推进与存档 ----
        cur_iter = games_total // cfg.iter_games
        if cur_iter >= next_ckpt_iter + 1 or (
                next_ckpt_iter == start_iter and cur_iter > start_iter):
            next_ckpt_iter = cur_iter
            snap = {
                "iter": cur_iter,
                "steps": steps_done,
                "games_total": games_total,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "cfg_blocks": cfg.blocks,
                "cfg_channels": cfg.channels,
            }
            tmp = latest_pt + ".tmp"
            torch.save(snap, tmp)
            os.replace(tmp, latest_pt)
            if cur_iter % cfg.save_every_iters == 0:
                torch.save(snap, os.path.join(
                    cfg.path_model_dir(), f"ckpt_iter{cur_iter:04d}.pt"))
            now = time.time()
            gps = (games_total - last_games_total) / max(now - last_t, 1e-9)
            logger.writerow([f"{now - t0:.0f}", cur_iter, steps_done, games_total,
                             buf_len, f"{last_pl:.4f}" if done_here else "",
                             f"{last_vl:.4f}" if done_here else "", f"{lr:.2e}",
                             f"{gps:.1f}"])
            log_f.flush()
            last_games_total = games_total
            last_t = now
            print(f"[train] iter={cur_iter} steps={steps_done}/{target_steps} "
                  f"games={games_total} buffer={buf_len} "
                  f"pol={last_pl:.3f} val={last_vl:.3f} lr={lr:.2e} "
                  f"{gps:.1f} games/s", flush=True)
            model.train()

        # ---- 5) actor 存活检查 ----
        from .selfplay import actor_main
        for i, p in enumerate(procs):
            if not p.is_alive():
                print(f"[train] actor {i} 挂了，重启", flush=True)
                p.join(timeout=1)
                procs[i] = ctx.Process(
                    target=actor_main,
                    args=(i, cfg.sims, cfg.c_puct, cfg.dir_alpha, cfg.dir_eps,
                          cfg.temp_moves, cfg.games_per_actor,
                          cfg.seed + 100000 + i, req_q, res_qs[i], games_q),
                    daemon=True,
                )
                procs[i].start()

        # ---- 6) 停机条件 ----
        if steps_done >= target_steps:
            break
        if not any(p.is_alive() for p in procs) and batch_items == []:
            time.sleep(0.05)
        if not batch_items and done_here == 0:
            time.sleep(0.002)

    # ---- 收尾 ----
    print("[train] 停止中，保存最终 checkpoint ...", flush=True)
    snap = {
        "iter": games_total // cfg.iter_games,
        "steps": steps_done,
        "games_total": games_total,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "cfg_blocks": cfg.blocks,
        "cfg_channels": cfg.channels,
    }
    tmp = latest_pt + ".tmp"
    torch.save(snap, tmp)
    os.replace(tmp, latest_pt)
    for p in procs:
        p.terminate()
    log_f.close()
    print(f"[train] 完成：iter={snap['iter']} steps={steps_done} games={games_total}",
          flush=True)


if __name__ == "__main__":
    main()
