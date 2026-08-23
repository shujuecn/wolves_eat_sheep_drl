"""硬解表库读取器（只读）。

支持子仓库表库目录中的三种格式（自动识别，语义与 web/server.py 一致）：
- v1 全量：slot = (wr * C(22,k) + sr) * 2 + turn_bit；
- v2 canonical：镜像规范压缩，slot = pair_id * 2 + turn_bit（aux 段提供映射）；
- v2 compressed：在 v2 基础上按 262144 条目/块做 zlib 分块压缩 + 索引。

约定（与子仓库一致）：
- entry 字节：bit0-1 结果（0=狼胜 1=羊胜 2=和棋），bit2-7 最短步数 DTM；
- slot 低位 turn_bit：0=狼回合，1=羊回合。
"""

from __future__ import annotations

import mmap
import os
import struct
import zlib
from functools import lru_cache

from .game import DRAW, SHEEP_WIN, State, WOLF_WIN, ONGOING, immediate_result

WOLF_COMBOS = 2300
CHUNK_ENTRIES = 262144
MAX_K = 15

# ----------------------------------------------------------------
# 二项式系数与组合编码（combinadic，与子仓库 encode.cpp 同构）
# ----------------------------------------------------------------
BINOM: list[list[int]] = [[0] * 26 for _ in range(26)]
for _n in range(26):
    BINOM[_n][0] = BINOM[_n][_n] = 1
    for _k in range(1, _n):
        BINOM[_n][_k] = BINOM[_n - 1][_k - 1] + BINOM[_n - 1][_k]

SHEEP_COMBOS = {k: BINOM[22][k] for k in range(MAX_K + 1)}


def bits_cells(bb: int):
    cells = []
    while bb:
        lsb = bb & (-bb)
        cells.append(lsb.bit_length() - 1)
        bb ^= lsb
    return cells


def encode_wolf_cells(cells) -> int:
    c = sorted(cells)
    return BINOM[c[0]][1] + BINOM[c[1]][2] + BINOM[c[2]][3]


def encode_wolf_bb(wolf_bb: int) -> int:
    return encode_wolf_cells(bits_cells(wolf_bb))


def decode_combination(rank: int, n: int, k: int) -> list[int]:
    out = []
    pos = n - 1
    for i in range(k, 0, -1):
        while pos >= i - 1 and BINOM[pos][i] > rank:
            pos -= 1
        out.append(pos)
        rank -= BINOM[pos][i]
        pos -= 1
    out.reverse()
    return out


# 每个狼组合的自由格列表与全局->自由格映射
_WOLF_CELLS = [decode_combination(r, 25, 3) for r in range(WOLF_COMBOS)]
_FREE_LIST: list[tuple[int, ...]] = []
_G2F: list[tuple[int, ...]] = []
for _cells in _WOLF_CELLS:
    ws = set(_cells)
    free = [c for c in range(25) if c not in ws]
    g2f = [-1] * 25
    for i, c in enumerate(free):
        g2f[c] = i
    _FREE_LIST.append(tuple(free))
    _G2F.append(tuple(g2f))
FREE_LIST = tuple(_FREE_LIST)
G2F = tuple(_G2F)


def encode_sheep_cells(cells, wolf_rank: int, k: int) -> int:
    """羊全局格子 -> 该狼组合下的自由格组合序。"""
    if k == 0:
        return 0
    g2f = G2F[wolf_rank]
    mapped = sorted(g2f[p] for p in cells)
    rank = 0
    for i, p in enumerate(mapped):
        rank += BINOM[p][i + 1]
    return rank


def decode_sheep_rank(sr: int, wolf_rank: int, k: int) -> list[int]:
    if k == 0:
        return []
    free = FREE_LIST[wolf_rank]
    mapped = decode_combination(sr, 22, k)
    return sorted(free[i] for i in mapped)


def mirror_cell(c: int) -> int:
    return (c // 5) * 5 + (4 - c % 5)


@lru_cache(maxsize=None)
def mirror_wolf_rank(rank: int) -> int:
    return encode_wolf_cells(mirror_cell(p) for p in _WOLF_CELLS[rank])


# ----------------------------------------------------------------
# 镜像规范索引（与子仓库 CanonIndex 行为一致）
# ----------------------------------------------------------------
class CanonIndex:
    __slots__ = ("k", "canon_pairs", "sheep_combos", "prefix", "self_ranks",
                 "self_bits", "_blocks")

    def __init__(self) -> None:
        self.k = 0
        self.canon_pairs = 0
        self.sheep_combos = 0
        self.prefix = (0,) * 2301
        self.self_ranks: list[int] = []
        self.self_bits: dict[int, int] = {}
        self._blocks = 0

    def parse_aux(self, buf: bytes, k: int) -> bool:
        if len(buf) < 24:
            return False
        magic, aux_len, pairs, n, sc, blocks = struct.unpack_from("<6I", buf, 0)
        if magic != 0x43414E41 or aux_len != len(buf):
            return False
        if sc != SHEEP_COMBOS[k]:
            return False
        self.canon_pairs = pairs
        self.sheep_combos = sc
        self.k = k
        self._blocks = blocks
        off = 24
        self.prefix = struct.unpack_from("<2300I", buf, off) + (pairs,)
        off += 2300 * 4
        per = 12 + 16 * blocks
        self.self_ranks = []
        self.self_bits = {}
        for _ in range(n):
            wr, _lc, _fc = struct.unpack_from("<3I", buf, off)
            off += 12
            bits = int.from_bytes(buf[off:off + 8 * blocks], "little")
            off += 16 * blocks
            self.self_ranks.append(wr)
            self.self_bits[wr] = bits
        return off == len(buf)

    def is_leader(self, wr: int) -> bool:
        return wr < mirror_wolf_rank(wr)

    def is_selfmirror(self, wr: int) -> bool:
        return wr == mirror_wolf_rank(wr)

    def _local_id(self, wr: int, sr: int) -> int:
        b = self.self_bits[wr]
        return (b & ((1 << sr) - 1)).bit_count()

    def pair_id(self, wr: int, sr: int) -> int:
        if self.is_leader(wr):
            return self.prefix[wr] + sr
        if self.is_selfmirror(wr):
            cells = [mirror_cell(p) for p in decode_sheep_rank(sr, wr, self.k)]
            psr = encode_sheep_cells(cells, wr, self.k)
            s = sr if sr <= psr else psr
            return self.prefix[wr] + self._local_id(wr, s)
        mwr = mirror_wolf_rank(wr)
        msr = encode_sheep_cells(
            [mirror_cell(p) for p in decode_sheep_rank(sr, wr, self.k)], mwr, self.k
        )
        return self.prefix[mwr] + msr


# ----------------------------------------------------------------
# 表库访问入口
# ----------------------------------------------------------------
_RESULT_NAME = {WOLF_WIN: "wolf_win", SHEEP_WIN: "sheep_win", DRAW: "draw"}


class Tablebase:
    """多桶表库读取器：lookup(State) -> (result, dist)；result 为绝对胜负。"""

    def __init__(self, data_dir: str) -> None:
        self.dir = data_dir
        self._mm: dict[int, mmap.mmap] = {}
        self._canon: dict[int, CanonIndex | None] = {}
        self._entries: dict[int, int] = {}
        self._chunks: dict[int, list[tuple[int, int]]] = {}
        self._chunk_cache: dict[int, dict[int, bytes]] = {}
        self._lru: dict[int, list[int]] = {}

    # ---- 文件打开 ----
    def _path(self, k: int) -> str:
        return os.path.join(self.dir, f"dtc_k{k:02d}.bin")

    def _open(self, k: int) -> bool:
        if k in self._mm:
            return True
        path = self._path(k)
        try:
            f = open(path, "rb")
        except OSError:
            return False
        with f:
            size = os.fstat(f.fileno()).st_size
            hdr = f.read(64)
            if len(hdr) < 64 or hdr[:4] != b"WSTB":
                return False
            version = hdr[4]
            kk = hdr[5]
            flags = hdr[7]
            total_entries_hdr = struct.unpack_from("<Q", hdr, 16)[0]
            aux_len = struct.unpack_from("<I", hdr, 28)[0]
            cc = struct.unpack_from("<I", hdr, 32)[0]
            cdlen = struct.unpack_from("<I", hdr, 36)[0]
            if kk != k:
                raise ValueError(f"bucket k mismatch: file says {kk}, want {k}")
            compressed = bool(flags & 2) and version == 2 and (flags & 1)
            canonical = version == 2 and (flags & 1)
            if compressed:
                expect = 64 + cdlen + 12 * cc + aux_len
            elif canonical:
                expect = 64 + total_entries_hdr + aux_len
            else:
                expect = 64 + WOLF_COMBOS * SHEEP_COMBOS[k] * 2
            if size != expect:
                raise ValueError(f"bad tb file size for k={k}: {size} != {expect}")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        if canonical:
            if compressed:
                chunks = []
                idx_off = 64 + cdlen
                for i in range(cc):
                    off = struct.unpack_from("<Q", mm, idx_off + 12 * i)[0]
                    sz = struct.unpack_from("<I", mm, idx_off + 12 * i + 8)[0]
                    chunks.append((off, sz))
                self._chunks[k] = chunks
                self._chunk_cache[k] = {}
                self._lru[k] = []
                aux = mm[64 + cdlen + 12 * cc: 64 + cdlen + 12 * cc + aux_len]
                entries = cc * CHUNK_ENTRIES
            else:
                aux = mm[64 + total_entries_hdr: 64 + total_entries_hdr + aux_len]
                entries = total_entries_hdr
            canon = CanonIndex()
            if not canon.parse_aux(aux, k):
                mm.close()
                raise ValueError(f"failed to parse aux segment for k={k}")
            self._canon[k] = canon
            self._entries[k] = entries
        else:
            self._canon[k] = None
            self._entries[k] = WOLF_COMBOS * SHEEP_COMBOS[k] * 2
        self._mm[k] = mm
        return True

    # ---- 条目读取 ----
    def _chunk_bytes(self, k: int, c: int) -> bytes:
        cache = self._chunk_cache[k]
        buf = cache.get(c)
        lru = self._lru[k]
        if buf is not None:
            if c in lru:
                lru.remove(c)
            lru.append(c)
            return buf
        off, sz = self._chunks[k][c]
        buf = zlib.decompress(self._mm[k][off:off + sz])
        cache[c] = buf
        lru.append(c)
        while len(cache) > 256:
            victim = lru.pop(0)
            cache.pop(victim, None)
        return buf

    def _entry_at(self, k: int, slot: int) -> int:
        chunks = self._chunks.get(k)
        if not chunks:
            return self._mm[k][64 + slot]
        c = slot // CHUNK_ENTRIES
        return self._chunk_bytes(k, c)[slot - c * CHUNK_ENTRIES]

    # ---- 查询 ----
    def lookup_ranks(self, wr: int, sr: int, k: int, sheep_to_move: bool):
        """返回 (known, result, dist)。"""
        if k < 4:
            return True, WOLF_WIN, 0
        if not self._open(k):
            return False, -1, 0
        canon = self._canon[k]
        if canon is not None:
            slot = canon.pair_id(wr, sr) * 2 + (1 if sheep_to_move else 0)
        else:
            slot = (wr * SHEEP_COMBOS[k] + sr) * 2 + (1 if sheep_to_move else 0)
        e = self._entry_at(k, slot)
        return True, e & 3, (e >> 2) & 0x3F

    def lookup_state(self, s: State):
        """返回 (known, result, dist)。State.turn: 1=狼回合。"""
        k = s.sheep.bit_count()
        if k < 4:
            return True, WOLF_WIN, 0
        wr = encode_wolf_bb(s.wolf)
        sr = encode_sheep_cells(bits_cells(s.sheep), wr, k)
        return self.lookup_ranks(wr, sr, k, s.turn == 0)

    def value_for_mover(self, s: State) -> int | None:
        """当前回合方视角的精确价值：胜 +1 / 和 0 / 负 -1；未知返回 None。"""
        res = immediate_result(s)
        if res != ONGOING:
            from .game import result_for_mover
            return result_for_mover(res, s)
        known, result, _dist = self.lookup_state(s)
        if not known:
            return None
        mover_is_wolf = s.turn == 1
        if result == DRAW:
            return 0
        mover_wins = (result == WOLF_WIN) == mover_is_wolf
        return 1 if mover_wins else -1

    def close(self) -> None:
        for mm in self._mm.values():
            try:
                mm.close()
            except Exception:
                pass
        self._mm.clear()


def load_tablebase(data_dir: str | None = None) -> Tablebase:
    if data_dir is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(root, "data", "ws_tb_dtc_v2_c")
    return Tablebase(data_dir)
