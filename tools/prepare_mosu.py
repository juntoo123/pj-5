"""
prepare_mosu.py ― MoSu の train / val から動画を選んで、train.py が読む形に変換する（testは使わない）

使い方: python tools/prepare_mosu.py --modality visual --n_train 200 --n_val 50 --out data_mosu
        （特徴量の .h5 を手元に置いていない場合は --remote で、必要な動画だけネット越しに読む）
        タグで選ぶ例: --cluster 1 --exclude_tags "Trailer (promotion)" "Movieclips"
                      --include_tags "Call of Duty"
        本数だけ確かめる: --count_only
入力:   {mosu_dir}/mosu_split.json      train_keys / val_keys / test_keys
        {mosu_dir}/mosu_metadata.csv    video_id, duration, cluster_id など
        {mosu_dir}/mosu_gt.h5           動画ごとの gt_score（1秒ごと、0〜1）
        {mosu_dir}/mosu_feat_{modality}_*.h5  動画ごとの特徴量（秒数, 768）
        {mosu_dir}/vocabulary.csv       YouTube-8M のタグの対応表（タグで選ぶときだけ必要）
出力:   {out}/features/{modality}/{video_id}.npy  形（秒数, 768）
        {out}/labels/{video_id}.npy               形（秒数,）
        {out}/splits/{train,val}.txt

注意：正解は MoSu の gt_score をそのまま使う仮のもの。正式な変換方法は評価・統合担当が決める。
配布元: https://huggingface.co/datasets/hminjeong/TripleSumm-MoSu （CC BY 4.0）
"""
import argparse, ast, csv, difflib, json, os, urllib.request
import numpy as np
import h5py

HF = "https://huggingface.co/datasets/hminjeong/TripleSumm-MoSu/resolve/main"
FEATURE_FILES = {"visual": "mosu_feat_visual_clip.h5", "audio": "mosu_feat_audio_ast.h5",
                 "text": "mosu_feat_text_roberta.h5"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mosu_dir", default="mosu")  # MoSu のファイルを置く（ない小さいファイルは自動で取ってくる）
    p.add_argument("--modality", default="visual", choices=["visual", "audio", "text"])
    p.add_argument("--n_train", type=int, default=200)  # 0なら train 全部
    p.add_argument("--n_val", type=int, default=50)     # 0なら val 全部
    p.add_argument("--cluster", type=int, default=None)  # ジャンル（cluster_id 0〜9）で絞る。指定なしなら全部
    p.add_argument("--include_tags", nargs="*", default=[])  # このどれかのタグが付いた動画だけ使う
    p.add_argument("--exclude_tags", nargs="*", default=[])  # このどれかのタグが付いた動画は使わない
    p.add_argument("--count_only", action="store_true")  # 選ばれる本数を表示するだけで、変換はしない
    p.add_argument("--out", default="data_mosu")
    p.add_argument("--seed", type=int, default=0)  # 動画を選ぶときの乱数
    p.add_argument("--remote", action="store_true")  # 特徴量の .h5 をダウンロードせず、ネット越しに読む
    return p.parse_args()


def fetch_small(name, mosu_dir):
    """分割・メタデータ・正解（合計約230MB）は、なければダウンロードする"""
    path = f"{mosu_dir}/{name}"
    if not os.path.exists(path):
        os.makedirs(mosu_dir, exist_ok=True)
        print("ダウンロード:", name)
        urllib.request.urlretrieve(f"{HF}/{name}", path + ".tmp")
        os.rename(path + ".tmp", path)
    return path


def open_features(args):
    path = f"{args.mosu_dir}/{FEATURE_FILES[args.modality]}"
    if os.path.exists(path):
        return h5py.File(path, "r")
    if not args.remote:
        raise SystemExit(f"{path} がない。ダウンロードして置くか、--remote を付けて実行すること")
    import fsspec
    f = fsspec.open(f"{HF}/{FEATURE_FILES[args.modality]}", block_size=2**20,
                    client_kwargs={"trust_env": True}).open()  # trust_env: プロキシの設定を使う
    return h5py.File(f, "r")


def tag_indices(names, mosu_dir):
    """タグの名前（vocabulary.csv の Name）を番号に直す"""
    path = f"{mosu_dir}/vocabulary.csv"
    if not os.path.exists(path):
        raise SystemExit(f"{path} がない。次で取ってくること:\n"
                         f"  curl -fL -o {path} https://research.google.com/youtube8m/csv/2/vocabulary.csv")
    with open(path, encoding="utf-8") as f:
        index_of = {r["Name"]: int(r["Index"]) for r in csv.DictReader(f)}
    out = set()
    for name in names:
        if name not in index_of:
            near = difflib.get_close_matches(name, index_of, n=5)
            raise SystemExit(f"タグ「{name}」が見つからない。近い名前: {near}")
        out.add(index_of[name])
    return out


def pick(keys, n, ok, rng):
    keys = sorted(k for k in keys if ok(k))
    if n and n < len(keys):
        keys = sorted(rng.choice(keys, size=n, replace=False).tolist())
    return keys


def main():
    args = parse_args()
    split = json.load(open(fetch_small("mosu_split.json", args.mosu_dir)))
    with open(fetch_small("mosu_metadata.csv", args.mosu_dir), encoding="utf-8-sig") as f:
        meta = {r["video_id"]: r for r in csv.DictReader(f)}
    include = tag_indices(args.include_tags, args.mosu_dir) if args.include_tags else set()
    exclude = tag_indices(args.exclude_tags, args.mosu_dir) if args.exclude_tags else set()

    def ok(vid):
        if args.cluster is not None and int(meta[vid]["cluster_id"]) != args.cluster:
            return False
        tags = set(ast.literal_eval(meta[vid]["labels"]))
        if include and not tags & include:
            return False
        return not tags & exclude

    rng = np.random.default_rng(args.seed)
    chosen = {"train": pick(split["train_keys"], args.n_train, ok, rng),
              "val": pick(split["val_keys"], args.n_val, ok, rng)}
    test = set(split["test_keys"])
    assert not test & (set(chosen["train"]) | set(chosen["val"])), "testの動画が混ざっている"
    total = {name: sum(ok(v) for v in split[f"{name}_keys"]) for name in ("train", "val")}
    print(f"条件に合う動画: train {total['train']}本 / val {total['val']}本 → 使う: "
          f"train {len(chosen['train'])}本 / val {len(chosen['val'])}本")
    if args.count_only:
        return

    for d in [f"features/{args.modality}", "labels", "splits"]:
        os.makedirs(f"{args.out}/{d}", exist_ok=True)
    with h5py.File(fetch_small("mosu_gt.h5", args.mosu_dir), "r") as gt, open_features(args) as feat:
        for name, keys in chosen.items():
            for i, vid in enumerate(keys):
                x = np.asarray(feat[vid], dtype=np.float32)
                y = np.asarray(gt[vid]["gt_score"], dtype=np.float32)
                np.save(f"{args.out}/features/{args.modality}/{vid}.npy", x)
                np.save(f"{args.out}/labels/{vid}.npy", y)
                print(f"{name} {i + 1}/{len(keys)} {vid} 特徴量{x.shape} 正解{y.shape}", flush=True)
            with open(f"{args.out}/splits/{name}.txt", "w") as f:
                f.write("\n".join(keys))
    print(f"完了: train {len(chosen['train'])}本 / val {len(chosen['val'])}本 → {args.out}/（testは未使用）")


if __name__ == "__main__":
    main()
