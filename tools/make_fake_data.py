"""
make_fake_data.py ― 動作確認用の偽データを作る（本物の正解データができるまでの代わり）

使い方: python tools/make_fake_data.py --out data_fake --dim 1024
出力:   train.py が読む形と同じ
          {out}/features/{modality}/{video_id}.npy  形（秒数, 次元）
          {out}/labels/{video_id}.npy               形（秒数,） 0〜1
          {out}/splits/{train,val}.txt

偽データの仕組み：各動画に隠れた「盛り上がり度」（山が1〜3個）があり、それが正解になる。
特徴量の0番目と1番目にだけ盛り上がり度が混ざっていて、残りはただのノイズ。
"""
import argparse, os
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data_fake")
    p.add_argument("--modality", default="visual", choices=["visual", "audio", "text"])
    p.add_argument("--n_train", type=int, default=200)
    p.add_argument("--n_val", type=int, default=50)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--min_sec", type=int, default=60)
    p.add_argument("--max_sec", type=int, default=500)
    p.add_argument("--noise", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    for d in [f"features/{args.modality}", "labels", "splits"]:
        os.makedirs(f"{args.out}/{d}", exist_ok=True)
    ids = [f"fake{i:05d}" for i in range(args.n_train + args.n_val)]
    for vid in ids:
        T = int(rng.integers(args.min_sec, args.max_sec + 1))  # 動画の長さ（秒）
        t = np.arange(T)
        excite = np.zeros(T)  # 隠れた盛り上がり度
        for _ in range(int(rng.integers(1, 4))):
            c, w = rng.uniform(0, T), rng.uniform(3, 10)
            excite += np.exp(-((t - c) / w) ** 2)
        excite = np.clip(excite, 0, 1)
        x = rng.normal(0, args.noise, size=(T, args.dim))
        x[:, 0] += excite * 2
        x[:, 1] += excite
        np.save(f"{args.out}/features/{args.modality}/{vid}.npy", x.astype(np.float32))
        np.save(f"{args.out}/labels/{vid}.npy", excite.astype(np.float32))
    with open(f"{args.out}/splits/train.txt", "w") as f:
        f.write("\n".join(ids[:args.n_train]))
    with open(f"{args.out}/splits/val.txt", "w") as f:
        f.write("\n".join(ids[args.n_train:]))
    print(f"{len(ids)}本を {args.out}/ に作成（次元 {args.dim}）")


if __name__ == "__main__":
    main()
