# 狼羊棋 AlphaZero 自对弈训练（主仓库）

用 **AlphaZero 自对弈**（MCTS + 策略价值网络，纯自我博弈，不使用任何监督/表库先验）
训练狼羊棋模型，目标：与子仓库 `wolves_eat_sheep_hard_solve` 的**硬解最优解达到 ≥95% 相似度**。

> 子仓库（C++ 全量逆向求解，188 亿状态表库）只读引用，不向其提交任何代码；
> 表库数据目录 `data/ws_tb_dtc_v2_c/` 由使用者上传，仅用于评估。

## 规则（与子仓库逐条一致，已对拍验证）

- 5×5；15 羊占上三排，3 狼在底排 1/2/3 列，狼先行；
- 狼：直走一格（空格），或沿该方向隔一个空格跳吃第 2 格上的羊；
- 羊：直走一格（空格），不能吃子；
- 狼胜：羊 <4；羊胜：三狼四邻全占；轮到羊而羊无路可走 → 狼胜（与逆推求解器零后继判负一致）；
- 和棋：两点往返计数双方同时 ≥5（rules.py 移植）或总步数 ≥150。

## 目录结构

```
az/                    # AlphaZero 包
├── config.py          # 超参数
├── game.py            # 位棋盘规则引擎 + 对局环境（含重复和棋规则）
├── encode.py          # 状态平面编码 / 动作空间 / 水平镜像增广
├── network.py         # ResNet 策略-价值网络（6 块 ×64 通道，约 48 万参数）
├── mcts.py            # 协作式 MCTS 与自对弈单局（纯 Python，无 torch）
├── selfplay.py        # actor 进程（CPU 自对弈，评估走主进程推理服务）
├── train.py           # 训练主控：GPU 批量推理服务 + 回放池 + 训练 + checkpoint
├── evaluate.py        # 对照硬解表库的评估器
└── tablebase.py       # v1 / v2 canonical / v2 compressed 表库读取器（自实现）
tests/
├── test_game_crosscheck.py       # 规则对拍 vs 子仓库 rules.py
└── test_tablebase_crosscheck.py  # 表库读取对拍 vs 子仓库参考实现
scripts/               # 启动脚本
data/models/           # checkpoint（gitignore）
results/               # 评估报告（入库）
```

## 使用

```bash
conda activate torch

# 规则对拍（800 局随机对局与子仓库 rules.py 逐步一致）
python tests/test_game_crosscheck.py 800

# 表库读取对拍（需要 data/ws_tb_dtc_v2_c/ 就绪）
python tests/test_tablebase_crosscheck.py 400

# 训练（默认 20 actors × 24 并发局 × 160 sims；60 iters ≈ 2h @ RTX4090）
python -m az.train --actors 20 --sims 160 --iters 60
python -m az.train --resume          # 续训

# 评估对照硬解最优解
python -m az.evaluate --ckpt data/models/latest.pt --n-pos 2000 --sims 600
```

## 「相似度」口径

对自对弈轨迹采样出的真实可达局面（按羊数分层去重），以硬解表库为真值：

| 指标 | 含义 |
|---|---|
| 最优走法命中率 | 模型首选着法 ∈ {所有保持博弈论价值的着法} |
| 严格 DTM 命中率 | 胜方最快将杀 / 败方最长抵抗的着法集合命中 |
| 价值结论一致率 | 根价值估计与表库胜/和/负结论一致 |
| 自弈结果达成率 | 从局面出发模型自弈至终局的实际结果 = 表库结论 |

**综合相似度 = 0.5×(最优命中率 + 达成率)**，目标 ≥95%。

## 进展

- [x] 规则引擎与 rules.py/C++ 双实现对拍一致（含重复和棋、150 步和棋）
- [x] v2 压缩表库读取器（canonical 镜像索引 + deflate 分块）
- [x] AlphaZero 完整管线（多进程自对弈 + GPU 批推理 + 训练 + 断点续训）
- [ ] 训练收敛 + 评估达标 ≥95%

### 训练里程（同口径评估：1921 局面，sims=600）

| 阶段 | 对局数 | 最优命中 | 严格 DTM | 价值一致 | 自弈达成率 | 综合相似度 |
|---|---:|---:|---:|---:|---:|---:|
| run6 champ iter180 | 69,120 | 98.80% | 66.89% | 90.79% | **73.25%** | **86.03%** |
| run8 iter570（phase7+8 共 +10 万局） | 218,880 | 98.70% | 69.81% | **93.86%** | 70.67% | 84.68% |

结论：价值理解持续变强，自弈达成率（胜面转化）为当前瓶颈；
phase9 以 sims 480→640 提升自弈标签质量续训 20 万局进行中。
详细报告见 `results/*.md`。
