"""AlphaZero 策略-价值网络（小型 ResNet）。

输入  (B,3,5,5) uint8/float32：羊位板、狼位板、行棋方平面
输出  policy logits (B,200)（非法动作由调用方以 mask 屏蔽），value (B,)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class AlphaZeroNet(nn.Module):
    def __init__(self, blocks: int = 6, channels: int = 64, num_actions: int = 200):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        # 策略头
        self.p_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.p_fc = nn.Linear(32 * 25, num_actions)
        # 价值头
        self.v_conv = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.v_fc1 = nn.Linear(16 * 25, 64)
        self.v_fc2 = nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        # 输出层小初始化（策略接近均匀、价值接近 0，训练初期更稳）
        nn.init.normal_(self.p_fc.weight, std=0.01)
        nn.init.zeros_(self.p_fc.bias)
        nn.init.normal_(self.v_fc2.weight, std=0.01)
        nn.init.zeros_(self.v_fc2.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """x: (B,3,5,5) float; mask: (B,200) bool 合法动作。

        返回 masked logits（非法=-inf 等效大负值）与 tanh 价值 (B,)。
        """
        h = self.stem(x)
        h = self.trunk(h)
        p = self.p_conv(h).flatten(1)
        logits = self.p_fc(p).masked_fill(~mask, -1e9)
        v = self.v_conv(h).flatten(1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v)).squeeze(1)
        return logits, v

    @torch.inference_mode()
    def infer_batch(self, planes: torch.Tensor, masks: torch.Tensor):
        """推理辅助：planes uint8 (B,3,5,5)，masks bool (B,200)。"""
        was_training = self.training
        self.eval()
        x = planes.to(next(self.parameters()).device, non_blocking=True).float()
        m = masks.to(x.device)
        logits, v = self.forward(x, m)
        if was_training:
            self.train()
        return logits, v
