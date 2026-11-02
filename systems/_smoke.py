"""
systems/_smoke.py ― GPUで動くかを確かめるためのテスト用モデル（2段の1次元CNN）

使い方: python train.py --system _smoke --modality visual --seed 0 --data_dir data_fake
自分のモデルは systems/{自分のシステム名}.py に作る（このファイルは書き換えない）
"""
import torch
import torch.nn as nn

from train import batches


class CNNModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Conv1d(in_channels=in_dim, out_channels=64, kernel_size=3, padding=1, dilation=1)
        self.sc = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=3, padding=2, dilation=2)
        self.relu = nn.ReLU()

    def forward(self, X, M):
        h = self.fc(X.transpose(1, 2))
        h = self.relu(h)
        h = h * M.unsqueeze(1).float()  # 詰め物の秒を0に戻す
        h = self.sc(h)
        return h.squeeze(1)


def build_model(in_dim):
    return CNNModel(in_dim)


def train_one_epoch(model, data, opt, args, dev, rng):
    model.train()
    total, n = 0.0, 0
    for _, X, Y, M in batches(data, args.batch_size, dev, rng):
        P = model(X, M)
        loss = ((P - Y) ** 2)[M].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item(); n += 1
    return total / n
