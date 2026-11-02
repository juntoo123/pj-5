"""
mosu_genres.py ― MoSu のジャンル（cluster_id）ごとに、本数と多いタグを、YouTube-8M の正式な名前で表にする

使い方: python tools/mosu_genres.py
入力:   {mosu_dir}/mosu_metadata.csv、mosu_split.json（prepare_mosu.py が取ってくる）
        {mosu_dir}/vocabulary.csv（YouTube-8M のタグの対応表）
          curl -fL -o mosu/vocabulary.csv https://research.google.com/youtube8m/csv/2/vocabulary.csv
出力:   {out}（Markdown の表）。画面にも同じものを表示する
"""
import argparse, ast, csv, json
from collections import Counter, defaultdict
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mosu_dir", default="mosu")
    p.add_argument("--top", type=int, default=8)  # ジャンルごとに表示するタグの数
    p.add_argument("--out", default="mosu/genres.md")
    return p.parse_args()


def main():
    args = parse_args()
    with open(f"{args.mosu_dir}/vocabulary.csv", encoding="utf-8") as f:
        vocab = {int(r["Index"]): r for r in csv.DictReader(f)}
    with open(f"{args.mosu_dir}/mosu_metadata.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    split = json.load(open(f"{args.mosu_dir}/mosu_split.json"))
    split_of = {v: name for name in ("train", "val", "test") for v in split[f"{name}_keys"]}

    by = defaultdict(list)
    for r in rows:
        by[int(r["cluster_id"])].append(r)

    name = lambda i: vocab[i]["Name"] if i in vocab else f"?{i}"
    lines = ["# MoSu のジャンル（cluster_id）", "",
             "| cluster | 本数 | train / val / test | 平均の長さ | 大分類（Vertical1） | 多いタグ（付いている割合） |",
             "|---|---|---|---|---|---|"]
    for c in sorted(by):
        rs = by[c]
        n = len(rs)
        sp = Counter(split_of.get(r["video_id"]) for r in rs)
        dur = np.mean([float(r["duration"]) for r in rs])
        tags = Counter(t for r in rs for t in ast.literal_eval(r["labels"]))
        # 1本の動画で同じ大分類を2回数えないように、動画ごとに大分類の集合を作る
        vert = Counter(v for r in rs for v in {vocab[t]["Vertical1"] for t in ast.literal_eval(r["labels"])
                                               if t in vocab and vocab[t]["Vertical1"]})
        vtxt = "、".join(f"{v} {k / n:.0%}" for v, k in vert.most_common(3))
        ttxt = "、".join(f"{name(t)} {k / n:.0%}" for t, k in tags.most_common(args.top))
        lines.append(f"| {c} | {n:,} | {sp['train']:,} / {sp['val']:,} / {sp['test']:,} | {dur:.0f}秒 | {vtxt} | {ttxt} |")
    lines += ["", "割合は、そのジャンルの動画のうち、そのタグ（大分類）が付いている動画の割合。"]
    text = "\n".join(lines) + "\n"
    print(text)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print("保存:", args.out)


if __name__ == "__main__":
    main()
