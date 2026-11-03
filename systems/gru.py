import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from train import batches


class GRUScorer(nn.Module):
    def __init__(self, in_dim, hidden_size=64):
        super().__init__()

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, X, M):
        # pack_padded_sequenceに渡す長さはCPU上に置く。
        lengths = M.sum(dim=1).cpu()

        # 詰め物を除いて処理する。動画を長い順に並べておく必要はない。
        packed = pack_padded_sequence(
            X,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_states, _ = self.gru(packed)

        # 元の動画の順番と秒数に戻す。
        states, _ = pad_packed_sequence(
            packed_states,
            batch_first=True,
            total_length=X.shape[1],
        )

        # 最後の1秒だけでなく、すべての秒についてスコアを作る。
        scores = self.head(states).squeeze(-1)
        return torch.sigmoid(scores)


def build_model(in_dim):
    return GRUScorer(in_dim)


def train_one_epoch(model, data, opt, args, dev, rng):
    model.train()
    total_squared_error = 0.0
    total_seconds = 0

    for _, X, Y, M in batches(data, args.batch_size, dev, rng):
        opt.zero_grad()
        P = model(X, M)

        # 本物の秒だけで平均二乗誤差（MSE）を計算する。
        squared_errors = (P[M] - Y[M]) ** 2
        loss = squared_errors.mean()
        loss.backward()
        opt.step()

        # バッチごとの有効秒数の違いを考慮して集計する。
        n = squared_errors.numel()
        total_squared_error += loss.item() * n
        total_seconds += n

    if total_seconds == 0:
        raise ValueError("学習データに有効な秒がありません。")

    return total_squared_error / total_seconds
