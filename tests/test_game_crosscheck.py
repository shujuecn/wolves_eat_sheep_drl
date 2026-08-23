"""对拍测试：az.game 与子仓库 wolves_eat_sheep_game/rules.py 随机对局一致性。

逐 ply 比对：
1) 双方生成的合法走法集合（(from, to, captured) 规范化后）完全一致；
2) 终局判定一致（狼胜 / 羊胜 / 和棋，含两点重复与 150 步和棋）。

只读引用子仓库代码（sys.path 注入），不做任何写入。

用法：python tests/test_game_crosscheck.py [n_games]
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "wolves_eat_sheep_hard_solve"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SUB / "wolves_eat_sheep_game"))

import rules as ref  # noqa: E402  子仓库规则库（只读）

from az import game  # noqa: E402


def _ref_move_key(mv: ref.Move):
    return mv.destination, (mv.captured if mv.captured else mv.destination)


def _my_move_keys(state: game.State):
    out = {}
    for a in game.legal_actions(state):
        frm, to, cap = game.action_to_move(a)
        out[(to, (cap if cap >= 0 else to))] = a
    return out


def _cell(p) -> int:
    return p[0] * 5 + p[1]


def _ref_legal_keys(gs: ref.GameState):
    """返回 {(to_cell, cap_cell)} -> ((r,c), Move)，键与我方引擎同构。"""
    keys = {}
    for r in range(5):
        for c in range(5):
            if gs.board[r][c] != gs.turn:
                continue
            for mv in gs.legal_moves_from((r, c)):
                key = (_cell(mv.destination), _cell(mv.captured or mv.destination))
                keys[key] = ((r, c), mv)
    return keys


def _apply_ref(gs: ref.GameState, chosen_key, keys):
    (r, c), mv = keys[chosen_key]
    assert gs.move((r, c), mv.destination)


def play_one(seed: int, verbose: bool = False) -> str:
    rng = random.Random(seed)
    mine = game.Env()
    theirs = ref.GameState()
    steps = 0
    while True:
        # 我方已终局：先核对结果再退出（rules.py 终局后不再换边/生成走法）
        if mine.result != game.ONGOING:
            return _finish(mine, theirs, seed, steps)
        # 合法走法集合必须完全一致
        mk = _my_move_keys(mine.state)
        rkeys = _ref_legal_keys(theirs)
        assert set(mk.keys()) == set(rkeys.keys()), (
            f"seed={seed} ply={steps}\nmine={sorted(mk)}\ntheirs={sorted(rkeys)}"
        )
        choice = rng.choice(sorted(mk))
        mine.step(mk[choice])
        _apply_ref(theirs, choice, rkeys)
        steps += 1
        if verbose and steps % 20 == 0:
            print(f"  seed={seed} ply={steps} sheep={game.sheep_count(mine.state)}")


def _finish(mine: game.Env, theirs: ref.GameState, seed: int, steps: int) -> str:
    my_outcome = {
        game.WOLF_WIN: "wolf",
        game.SHEEP_WIN: "sheep",
        game.DRAW: "draw",
    }[mine.result]
    if my_outcome == "draw":
        assert theirs.winner == ref.DRAW, f"seed={seed}: 我判和 {theirs.winner}"
    elif my_outcome == "wolf":
        assert theirs.winning_side == ref.WOLF, f"seed={seed}: 我判狼胜 {theirs.winner}"
    else:
        assert theirs.winning_side == ref.SHEEP, f"seed={seed}: 我判羊胜 {theirs.winner}"
    return my_outcome


def main() -> None:
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    tally: dict[str, int] = {"wolf": 0, "sheep": 0, "draw": 0}
    max_steps = 0
    for i in range(n_games):
        outcome = play_one(i * 7919 + 13)
        tally[outcome] += 1
    # 初始局面与子仓库布局一致性
    s = game.initial_state()
    gs = ref.GameState()
    assert s.sheep == sum(
        1 << (r * 5 + c) for r in range(3) for c in range(5)
    ), "初始羊位板不符"
    assert s.wolf == (1 << 21) | (1 << 22) | (1 << 23), "初始狼位板不符"
    assert s.turn == 1 and gs.turn == ref.WOLF
    print(f"OK: {n_games} 局随机对拍全部一致")
    print(f"结果分布: 狼胜={tally['wolf']} 羊胜={tally['sheep']} 和棋={tally['draw']}")
    assert tally["draw"] > 0, "随机对局应能出现和棋（150步/重复），否则覆盖不足"
    assert tally["wolf"] > 0 and tally["sheep"] > 0


if __name__ == "__main__":
    main()
