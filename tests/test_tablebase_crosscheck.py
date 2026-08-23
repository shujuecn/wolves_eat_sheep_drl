"""表库读取器对拍测试（只读子仓库参考实现）。

1) 用子仓库 hard_solve_fast 的原语（state_index/CanonIndex/文件读取）构建参考查询，
   与 az.tablebase.Tablebase 在随机可达局面上逐条比对结果与 DTM；
2) 用 web/opening_book.json 锚点验证初始局面附近的最优走法集合。

用法：python tests/test_tablebase_crosscheck.py [n_states]
"""

from __future__ import annotations

import json
import random
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "wolves_eat_sheep_hard_solve"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SUB))

import hard_solve_fast as hsf  # noqa: E402 子仓库参考实现（只读）

from az import game  # noqa: E402
from az.tablebase import Tablebase, encode_sheep_cells, encode_wolf_bb, bits_cells  # noqa: E402

TB_DIR = ROOT / "data" / "ws_tb_dtc_v2_c"


class RefReader:
    """基于 hsf 原语的朴素参考实现（慢但直观）。"""

    def __init__(self, data_dir: Path):
        self.dir = str(data_dir)
        self._canon = {}

    def lookup(self, sheep_bb: int, wolf_bb: int, k: int, wolf_to_move: bool):
        if k < 4:
            return True, hsf.WOLF_WIN, 0
        path = f"{self.dir}/dtc_k{k:02d}.bin"
        with open(path, "rb") as f:
            f.seek(64)
            hdr_extra = f.read(64)
        with open(path, "rb") as f:
            head = f.read(64)
        version, flags = head[4], head[7]
        gr = hsf.encode_wolf(hsf.decode_wolf(0)) if False else None
        wr_cells = sorted(i for i in range(25) if (wolf_bb >> i) & 1)
        wr = hsf.encode_wolf(wr_cells)
        sr = hsf.encode_sheep(
            [i for i in range(25) if (sheep_bb >> i) & 1], wr, k
        )
        with open(path, "rb") as f:
            mm_all = f.read()
        aux_len = struct.unpack_from("<I", head, 28)[0]
        canon = self._canon.get(k)
        if version == 2 and (flags & 1):
            if canon is None:
                cc = struct.unpack_from("<I", head, 32)[0]
                cdlen = struct.unpack_from("<I", head, 36)[0]
                compressed = bool(flags & 2)
                if compressed:
                    aux_off = 64 + cdlen + 12 * cc
                else:
                    total = struct.unpack_from("<Q", head, 16)[0]
                    aux_off = 64 + total
                canon = hsf.CanonIndex()
                ok = canon.parse_aux(mm_all[aux_off:aux_off + aux_len], k)
                assert ok
                self._canon[k] = canon
            slot = canon.state_slot(wr, sr, not wolf_to_move)
        else:
            slot = hsf.state_index(wr, sr, k, not wolf_to_move)
        entry = self._entry(mm_all, head, slot, k)
        return True, entry & 3, (entry >> 2) & 0x3F

    def _entry(self, blob: bytes, head: bytes, slot: int, k: int) -> int:
        version, flags = head[4], head[7]
        if version == 2 and (flags & 1) and (flags & 2):
            cc = struct.unpack_from("<I", head, 32)[0]
            cdlen = struct.unpack_from("<I", head, 36)[0]
            idx_off = 64 + cdlen
            c = slot // hsf.CHUNK_ENTRIES
            off = struct.unpack_from("<Q", blob, idx_off + 12 * c)[0]
            sz = struct.unpack_from("<I", blob, idx_off + 12 * c + 8)[0]
            import zlib
            chunk = zlib.decompress(blob[off:off + sz])
            return chunk[slot - c * hsf.CHUNK_ENTRIES]
        return blob[64 + slot]


def reachable_random_state(rng: random.Random):
    """从初始局面随机游走得到真实可达局面。"""
    env = game.Env()
    while True:
        acts = game.legal_actions(env.state)
        res = game.immediate_result(env.state)
        if res != game.ONGOING or env.state.ply >= game.MAX_PLY:
            env.reset()
            continue
        a = rng.choice(acts)
        done = env.step(a)
        if done:
            env.reset()
            continue
        if rng.random() < 0.35:
            return env.state


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    rng = random.Random(20260823)
    mine = Tablebase(str(TB_DIR))
    ref = RefReader(TB_DIR)

    mism = 0
    from collections import Counter
    kc = Counter()
    for i in range(n):
        s = reachable_random_state(rng)
        k = s.sheep.bit_count()
        kc[k] += 1
        known_r, res_r, dist_r = ref.lookup(s.sheep, s.wolf, k, s.turn == 1)
        known_m, res_m, dist_m = mine.lookup_state(s)
        if (known_r, res_r, dist_r) != (known_m, res_m, dist_m):
            mism += 1
            print(f"MISMATCH i={i} k={k} sheep={s.sheep:#x} wolf={s.wolf:#x} turn={s.turn}\n"
                  f"  ref : {known_r} {res_r} {dist_r}\n  mine: {known_m} {res_m} {dist_m}")
            if mism > 5:
                break
    assert mism == 0, f"{mism} mismatches"

    # 开局库锚点：初始局面狼的最优解应为中路跳吃 (4,2)x(2,2)，且开局为和棋
    s0 = game.initial_state()
    known, result, dist = mine.lookup_state(s0)
    assert known and result == game.DRAW, f"initial should be draw, got {result}"
    my_win = game.WOLF_WIN
    best_acts = []
    for a in game.legal_actions(s0):
        ns = game.apply_action(s0, a)
        kn, r2, _d2 = mine.lookup_state(ns)
        if kn and r2 == DRAW:
            best_acts.append(a)
    mv = [game.action_to_move(a) for a in best_acts]
    center_jump = [(22, 12, 12)]
    assert center_jump in mv, f"center jump missing in optimal set: {mv}"

    # 羊方在狼中路跳吃后的局面应可达成羊胜或和棋（按表库）
    cj = (22, 12, 12)
    s1 = game.apply_action(s0, best_acts[mv.index(cj)])
    kn1, res1, d1 = mine.lookup_state(s1)
    assert kn1 and res1 in (game.SHEEP_WIN, game.DRAW), f"after c-jump: {res1}"
    print(f"OK: 表库对拍 {n} 局面全一致；k 分布 {dict(sorted(kc.items()))}")
    print(f"OK: 初始局面=和棋(DTM={dist})，最优首着含中路跳吃 22x12；跳吃后羊方结论={res1}(d={d1})")


if __name__ == "__main__":
    main()
