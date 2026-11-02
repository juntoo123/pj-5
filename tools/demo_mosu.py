"""
demo_mosu.py ― 学習した重みで MoSu の動画を予測し、正解（gt_score）のグラフと重ねて表示する（プロトタイプ）

使い方: python tools/demo_mosu.py --run runs/_smoke_visual_seed0 --n 5 --remote
入力:   {run}/summary.json（どのシステム・モダリティで学習したか）と {run}/best.pt（重み）
        MoSu の特徴量と正解（prepare_mosu.py と同じく mosu/ に置くか、--remote で読む）
出力:   {out}/pred.csv      video_id, t, score（動画ごとに0〜1）, raw_score（正規化前）
        {out}/{video_id}.png  予測と正解を重ねたグラフ
        {out}/all.png         全動画をまとめたグラフ

動画は、学習に使った train / val と test を除いた val の残りから選ぶ（モデルが一度も見ていない動画）。
"""
import argparse, csv, json, os, sys
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")  # 画面のないサーバーでも画像を保存できるようにする
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import load_system
from tools.prepare_mosu import fetch_small, open_features


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)  # runs/{system}_{modality}_seed{n}
    p.add_argument("--n", type=int, default=5)  # 表示する動画の数
    p.add_argument("--video_ids", nargs="*")  # 動画を直接指定する場合
    p.add_argument("--mosu_dir", default="mosu")
    p.add_argument("--remote", action="store_true")
    p.add_argument("--out", default="demo_out")
    p.add_argument("--clip_sec", type=int, default=15)  # 見どころ候補の長さ（秒）
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def normalize(p):
    return (p - p.min()) / (p.max() - p.min() + 1e-8)


def top_clip(score, sec):
    """sec秒の窓で平均スコアが一番高い区間の開始秒"""
    if len(score) <= sec:
        return 0
    return int(np.argmax(np.convolve(score, np.ones(sec) / sec, mode="valid")))


def main():
    args = parse_args()
    summary = json.load(open(f"{args.run}/summary.json"))
    run_args = summary["args"]
    system = load_system(run_args["system"])
    args.modality = run_args["modality"]  # open_features が使う

    split = json.load(open(fetch_small("mosu_split.json", args.mosu_dir)))
    with open(fetch_small("mosu_metadata.csv", args.mosu_dir), encoding="utf-8-sig") as f:
        youtube_of = {r["video_id"]: r["youtube_id"] for r in csv.DictReader(f)}
    if args.video_ids:
        ids = args.video_ids
    else:
        used = set()
        for name in ("train", "val"):  # 学習に使った動画は除く
            path = f"{run_args['data_dir']}/splits/{name}.txt"
            if os.path.exists(path):
                used |= {l.strip() for l in open(path) if l.strip()}
        candidates = sorted(set(split["val_keys"]) - used)
        ids = sorted(np.random.default_rng(args.seed).choice(candidates, size=args.n, replace=False).tolist())
    test = set(split["test_keys"])
    assert not test & set(ids), "testの動画が混ざっている"

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    results = []
    with h5py.File(fetch_small("mosu_gt.h5", args.mosu_dir), "r") as gt, open_features(args) as feat:
        model = None
        for vid in ids:
            x = np.asarray(feat[vid], dtype=np.float32)
            y = np.asarray(gt[vid]["gt_score"], dtype=np.float32)
            T = min(len(x), len(y))
            x, y = x[:T], y[:T]
            if model is None:  # 重みを読み込むのは1回だけ
                model = system.build_model(x.shape[1]).to(dev)
                model.load_state_dict(torch.load(f"{args.run}/best.pt", map_location=dev))
                model.eval()
            with torch.no_grad():
                X = torch.from_numpy(x)[None].to(dev)              # （1, 秒数, 次元）
                M = torch.ones(1, T, dtype=torch.bool, device=dev)  # 詰め物なし
                raw = model(X, M)[0].float().cpu().numpy()
            results.append((vid, raw, normalize(raw), y))
            print(f"{vid}: {T}秒", flush=True)

    with open(f"{args.out}/pred.csv", "w") as f:
        f.write("video_id,t,score,raw_score\n")
        for vid, raw, s, _ in results:
            f.writelines(f"{vid},{t},{v:.6f},{r:.6f}\n" for t, (v, r) in enumerate(zip(s, raw)))

    print(f"\n{'video_id':10s} {'誤差':>7s} {'相関':>6s}  予測の見どころ（{args.clip_sec}秒）")
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 2.6 * len(results)), squeeze=False)
    for ax, (vid, raw, s, y) in zip(axes[:, 0], results):
        err = float(np.mean((s - y) ** 2))  # train.py の val_error と同じ計算
        corr = float(np.corrcoef(s, y)[0, 1])
        st = top_clip(s, args.clip_sec)
        url = f"https://www.youtube.com/watch?v={youtube_of.get(vid, '')}&t={st}s"
        print(f"{vid:10s} {err:7.4f} {corr:6.3f}  {st}〜{st + args.clip_sec}秒  {url}")
        for a in (ax, plt.figure(figsize=(12, 3)).gca()):
            a.plot(y, color="#d0453a", lw=1.5, label="MoSu gt_score")
            a.plot(s, color="#3b6fb6", lw=1.5, label="prediction (normalized)")
            a.axvspan(st, st + args.clip_sec, color="#e08a2c", alpha=0.25, label=f"top {args.clip_sec}s")
            a.set_title(f"{vid}  error={err:.4f}  corr={corr:.3f}", fontsize=10)
            a.set_xlabel("second"); a.set_ylim(-0.05, 1.05); a.legend(loc="upper right", fontsize=8)
            if a is not ax:
                a.figure.tight_layout(); a.figure.savefig(f"{args.out}/{vid}.png", dpi=120); plt.close(a.figure)
    fig.tight_layout(); fig.savefig(f"{args.out}/all.png", dpi=110)
    print(f"\n保存: {args.out}/pred.csv, {args.out}/all.png, {args.out}/<video_id>.png")
    print(f"モデル: {run_args['system']} / {args.modality} / 重み {args.run}/best.pt（{summary['best_epoch']}エポック目）")


if __name__ == "__main__":
    main()
