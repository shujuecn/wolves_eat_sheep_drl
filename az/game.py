"""狼羊棋规则引擎（位棋盘实现）。

与子仓库 wolves_eat_sheep_hard_solve 的 board.cpp / rules.py 规则完全一致：
- 5×5 棋盘；初始 15 羊占上三排（行 0-2），3 羊……3 狼占底排列 1/2/3；狼先行。
- 狼：每方向若相邻格为空可直走一格；且若第 2 格是羊（中间格必须为空）可跳吃。
- 羊：每方向相邻格为空时直走一格；不能吃子。
- 狼胜：羊数 < 4。羊胜：所有狼的四邻全被占据（此时狼也必然无法跳吃，
  因为跳吃同样要求中间格为空——与 C++ any_wolf_can_move 判定等价）。
- 和棋：双方"两点往返"连续计数同时 >= 5（rules.py 两点重复规则，逐条移植），
  或总步数 >= 150。

状态用不可变元组表示，可直接作为 MCTS 树与转置表的键：
    State(sheep, wolf, turn, ply)
    sheep/wolf 为 25 位位板（bit i = 行 r=i//5 列 c=i%5），turn: 1=狼回合 0=羊回合。

动作编码：action = from_cell * 8 + dir * 2 + is_capture
    dir: 0=上(-1,0) 1=下(+1,0) 2=左(0,-1) 3=右(0,+1)，共 25*8=200 个动作槽。
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

BOARD = 5
CELLS = 25
FULL = (1 << CELLS) - 1
MAX_PLY = 150          # 与 board.cpp MAX_MOVES 一致（双方合计步数）
IDLE_LIMIT = 5         # rules.py IDLE_LIMIT：两点往返计数阈值
NUM_ACTIONS = CELLS * 8  # 200

WOLF_WIN = 0   # 与表库结果编码一致
SHEEP_WIN = 1
DRAW = 2
ONGOING = 3


class State(NamedTuple):
    sheep: int
    wolf: int
    turn: int
    ply: int


# ---------------------------------------------------------------
# 预计算表：邻接格 / 跳吃目标格
# 方向顺序与 rules.py DIRECTIONS = ((-1,0),(1,0),(0,-1),(0,1)) 一致
# ---------------------------------------------------------------
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

_NB: list[tuple[int, ...]] = []
_PREY: list[tuple[int, ...]] = []
for _r in range(BOARD):
    for _c in range(BOARD):
        _nbs, _prs = [], []
        for _dr, _dc in _DIRS:
            _nr, _nc = _r + _dr, _c + _dc
            if 0 <= _nr < BOARD and 0 <= _nc < BOARD:
                _nbs.append(_nr * BOARD + _nc)
                _pr, _pc = _r + 2 * _dr, _c + 2 * _dc
                _prs.append(_pr * BOARD + _pc if 0 <= _pr < BOARD and 0 <= _pc < BOARD else -1)
            else:
                _nbs.append(-1)
                _prs.append(-1)
        _NB.append(tuple(_nbs))
        _PREY.append(tuple(_prs))
NB = tuple(_NB)        # NB[cell] = (上格|−1, 下格|−1, 左格|−1, 右格|−1)
PREY = tuple(_PREY)    # PREY[cell] = 跳吃方向上的第 2 格（猎物所在格）

_ROW = tuple(i // BOARD for i in range(CELLS))
_COL = tuple(i % BOARD for i in range(CELLS))


def initial_state() -> State:
    sheep = 0
    for r in range(3):
        for c in range(BOARD):
            sheep |= 1 << (r * BOARD + c)
    wolf = (1 << 21) | (1 << 22) | (1 << 23)  # 行4 列1/2/3
    return State(sheep, wolf, 1, 0)


def sheep_count(s: State) -> int:
    return s.sheep.bit_count()


def legal_actions(s: State) -> tuple[int, ...]:
    """当前回合方全部合法动作（顺序：棋子升序 × 方向，先直走后跳吃）。"""
    occ = s.sheep | s.wolf
    acts: list[int] = []
    if s.turn == 1:
        bb = s.wolf
        while bb:
            lsb = bb & (-bb)
            frm = lsb.bit_length() - 1
            bb ^= lsb
            base = frm << 3
            nbs = NB[frm]
            prys = PREY[frm]
            for d in range(4):
                mid = nbs[d]
                if mid < 0:
                    continue
                if not (occ >> mid) & 1:
                    acts.append(base | (d << 1))
                    prey = prys[d]
                    if prey >= 0 and (s.sheep >> prey) & 1:
                        acts.append(base | (d << 1) | 1)
    else:
        bb = s.sheep
        while bb:
            lsb = bb & (-bb)
            frm = lsb.bit_length() - 1
            bb ^= lsb
            base = frm << 3
            nbs = NB[frm]
            for d in range(4):
                mid = nbs[d]
                if mid < 0:
                    continue
                if not (occ >> mid) & 1:
                    acts.append(base | (d << 1))
    return tuple(acts)


def action_to_move(a: int) -> tuple[int, int, int]:
    """解码动作 -> (from_cell, to_cell, captured_cell|-1)。"""
    frm = a >> 3
    d = (a >> 1) & 3
    if a & 1:
        to = PREY[frm][d]
        return frm, to, to
    return frm, NB[frm][d], -1


def apply_action(s: State, a: int) -> State:
    frm = a >> 3
    fm = 1 << frm
    if a & 1:  # 跳吃：落到猎物格并移除该羊
        to = PREY[frm][(a >> 1) & 3]
        if s.turn == 1:
            return State(s.sheep & ~(1 << to), (s.wolf & ~fm) | (1 << to), 0, s.ply + 1)
        raise ValueError("sheep cannot capture")
    to = NB[frm][(a >> 1) & 3]
    tm = 1 << to
    if s.turn == 1:
        return State(s.sheep, (s.wolf & ~fm) | tm, 0, s.ply + 1)
    return State((s.sheep & ~fm) | tm, s.wolf, 1, s.ply + 1)


def result_for_mover(result: int, s: State) -> int:
    """把绝对结果换算成当前回合方视角：胜 +1 / 和 0 / 负 -1。"""
    if result == DRAW:
        return 0
    mover_is_wolf = s.turn == 1
    mover_wins = (result == WOLF_WIN) == mover_is_wolf
    return 1 if mover_wins else -1


def immediate_result(s: State) -> int:
    """终局判定（不含重复局面规则）：WOLF_WIN/SHEEP_WIN/DRAW 或 ONGOING。

    与硬解求解器语义一致：
    - 羊 <4 → 狼胜；
    - 三狼四邻全占（含跳吃在内均无走法）→ 羊胜；
    - 轮到羊而羊无任何合法走法 → 羊负（狼胜，组合博弈惯例，与逆推求解器
      零后继状态判负一致）。
    """
    if s.sheep.bit_count() < 4:
        return WOLF_WIN
    occ = s.sheep | s.wolf
    bb = s.wolf
    while bb:
        lsb = bb & (-bb)
        w = lsb.bit_length() - 1
        bb ^= lsb
        nbs = NB[w]
        for d in range(4):
            mid = nbs[d]
            if mid >= 0 and not (occ >> mid) & 1:
                # 有狼能动 → 未终局；但若轮到羊且羊也全被堵则判狼胜
                if s.turn == 1:
                    return ONGOING
                sb = s.sheep
                while sb:
                    lsb2 = sb & (-sb)
                    c = lsb2.bit_length() - 1
                    sb ^= lsb2
                    cnbs = NB[c]
                    for d2 in range(4):
                        m2 = cnbs[d2]
                        if m2 >= 0 and not (occ >> m2) & 1:
                            return ONGOING
                return WOLF_WIN
    return SHEEP_WIN


# ---------------------------------------------------------------
# 对局环境：完整规则（含两点重复和棋 + 150 步和棋）
# ---------------------------------------------------------------

def _back_and_forth_count(positions: deque | list) -> int:
    """逐条移植 rules.py::_back_and_forth_count。"""
    n = len(positions)
    if n < 3:
        return 0
    count = 1
    for i in range(n - 3, -1, -1):
        if positions[i] != positions[i + 2]:
            break
        count += 1
    return count if count >= 2 else 0


class Env:
    """带完整终止规则的对局环境（重复规则逻辑与 rules.py 逐条一致）。"""

    def __init__(self) -> None:
        self.state = initial_state()
        self.result = ONGOING  # 终局后为 WOLF_WIN/SHEEP_WIN/DRAW
        self._hist_limit = IDLE_LIMIT + 2
        self._reset_bookkeeping()

    def _reset_bookkeeping(self) -> None:
        s = self.state
        self.piece_ids: dict[int, int] = {}
        self.piece_histories: dict[int, deque] = {}
        for cell in range(CELLS):
            side = 1 if (s.wolf >> cell) & 1 else (0 if (s.sheep >> cell) & 1 else -1)
            if side < 0:
                continue
            pid = (side << 5) | cell
            self.piece_ids[cell] = pid
            self.piece_histories[pid] = deque((cell,), maxlen=self._hist_limit)
        self.last_piece_by_side = {1: None, 0: None}
        self.idle_streaks = {1: 0, 0: 0}

    def reset(self) -> None:
        self.state = initial_state()
        self.result = ONGOING
        self._reset_bookkeeping()

    def legal_actions(self) -> tuple[int, ...]:
        return legal_actions(self.state)

    def step(self, action: int) -> bool:
        """执行一步。返回 True 表示对局已结束（self.result 有效）。"""
        assert self.result == ONGOING
        frm, to, cap = action_to_move(action)
        side = self.state.turn

        pid = self.piece_ids.pop(frm)
        hist = self.piece_histories[pid]
        if self.last_piece_by_side[side] != pid:
            hist.clear()
            hist.append(frm)
        if cap >= 0:
            captured_pid = self.piece_ids.pop(cap, None)
            if captured_pid is not None:
                self.piece_histories.pop(captured_pid, None)

        self.state = apply_action(self.state, action)
        self.piece_ids[to] = pid
        hist.append(to)
        self.last_piece_by_side[side] = pid

        repeat = _back_and_forth_count(hist)
        self.idle_streaks[side] = repeat

        res = immediate_result(self.state)
        if res == ONGOING:
            if all(self.idle_streaks[x] >= IDLE_LIMIT for x in (1, 0)):
                res = DRAW
            elif self.state.ply >= MAX_PLY:
                res = DRAW
        self.result = res
        return res != ONGOING

    # ---- 便捷接口 ----
    def mover_value_if(self, result: int) -> int:
        return result_for_mover(result, self.state)

    def clone_state_key(self) -> State:
        return self.state
