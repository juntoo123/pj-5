# highlight-project

動画を読み込み、1秒ごとの「見どころ度」を予測して、見どころを切り抜くプロジェクト。

## 全体の流れ

```
動画 (videos/*.mp4)
  │  features/extract_features.py   動画 → 1秒ごとの特徴量
  ▼
data/features/{visual,audio,text}/{video_id}.npy   形（秒数, 次元）
  │  train.py                        特徴量 + 正解 → モデルを学習
  ▼
runs/{system}_{modality}_seed{n}/best.pt
  │  highlight/predict.py            新しい動画 → 1秒ごとのスコア
  ▼
スコア（秒数,）
  │  highlight/clip.py               スコア → 見どころ区間 → 動画を切り抜く
  ▼
クリップ (outputs/*.mp4)
  │  frontend/                       結果を見るための画面（未着手）
```

## フォルダ構成

| 場所 | 中身 | 状態 |
|---|---|---|
| `train.py` | 学習スクリプト（共通） | 基本は書き換えない |
| `systems/` | 各自のモデルと学習（1人1ファイル） | `_template.py` をコピーして実装。`systems/README.md` 参照 |
| `features/` | 動画 → 特徴量の変換 | 未実装（雛形のみ） |
| `highlight/` | 学習済みモデルで予測し、切り抜く | 未実装（雛形のみ） |
| `frontend/` | 画面 | 未着手（置き場所だけ） |
| `jobs/` | 計算サーバ（Slurm）に投入するジョブスクリプト | |
| `tools/` | MoSu の変換（`prepare_mosu.py`）、偽データ（`make_fake_data.py`）、予測と正解の表示（`demo_mosu.py`） | |
| `docs/` | 計算サーバでの手順（`server_setup.md`） | |
| `data/` | 特徴量・正解・分割（Gitには入れない） | `data/README.md` 参照 |
| `runs/` | 学習結果（Gitには入れない） | 実行すると自動で作られる |
| `logs/` | ジョブの出力 `.out`（Gitには入れない） | |

## 環境構築（IS計算サーバ）

初めての場合（ログイン・鍵の設定・GitHubとの接続など）は [docs/server_setup.md](docs/server_setup.md) を最初から読むこと。

ログインノードで実行する（計算ノードには uv が入っていない）。

```bash
cd ~/highlight-project
uv lock
uv sync --locked
```

## 学習の実行

### GPUテスト（MoSuの一部で、小さいGPUを借りて3エポックだけ学習）

1. ログインノードで、MoSu の train / val から一部を `train.py` の形に変換する（testは使わない）

   ```bash
   source .venv/bin/activate
   python tools/prepare_mosu.py --modality visual --n_train 200 --n_val 50 --remote
   ```

   - `--remote`：40GBある特徴量ファイルをダウンロードせず、選んだ動画だけネット越しに読む
   - 分割・メタデータ・正解（約230MB）は `mosu/` に自動でダウンロードされる
   - `--cluster 0〜9` でジャンル（MoSu の cluster_id）を絞れる
   - `--include_tags "Call of Duty"` / `--exclude_tags "Trailer (promotion)"` でタグで選べる（`mosu/vocabulary.csv` が必要）
   - `--count_only` で、選ばれる本数だけを表示する
   - `--n_train 0 --n_val 0` で、条件に合う動画を全部使う
   - ジャンルとタグの一覧は `python tools/mosu_genres.py`
   - 正解は MoSu の `gt_score`（1秒ごと、0〜1）をそのまま使う仮のもの。正式な変換は評価・統合担当が決める

2. 小さいGPU（A100 MIG 10GB）でジョブを投入する

   ```bash
   sbatch -J gpu_test -p a100_1g --gres=gpu:1g.10gb:1 --qos=low jobs/gpu_test.sbatch
   tail -f logs/gpu_test-<ジョブID>.out
   ```

   - 最後に `gpu: NVIDIA A100 80GB PCIe MIG 1g.10gb` のように出れば成功
   - 自分のモデルで試すときは `--export=ALL,SYSTEM=fast` を付ける
   - 偽データで試すときは `--export=ALL,DATA=data_fake` を付ける（`tools/make_fake_data.py` で自動作成）
   - 結果は `runs_test/` に出る（データ・結果ともGitには入れない）

### 動作確認（対話セッション、30分まで）

```bash
srun -p a100_1g --gres=gpu:1g.10gb:1 --qos=inter --pty bash
source .venv/bin/activate
python train.py --system fast --modality visual --seed 0 --subset 5 --max_epochs 2
exit
```

### 本番（バッチジョブ）

プロジェクトのルートで実行する（ジョブは sbatch を実行した場所で始まるため）。

```bash
sbatch -J train_fast_visual \
  -p a100_3g --gres=gpu:3g.40gb:1 \
  --export=ALL,SYSTEM=fast,MODALITY=visual \
  jobs/train.sbatch
```

- `--qos` は所属によって異なる。サーバ利用ガイド 2.2 を確認すること
- 学習時間を比べるときは、全員同じパーティション（GPU）を使う

### プロトタイプ：予測と正解を重ねて表示（ログインノードで）

```bash
python tools/demo_mosu.py --run runs/_smoke_visual_seed0 --n 5 --remote
```

- `demo_out/all.png` に予測（青）と MoSu の正解（赤）を重ねたグラフ、`demo_out/pred.csv` に予測（video_id, t, score, raw_score）が出る
- 動画は、学習に使った train / val と test を除いた val の残りから選ぶ
- 予測で一番高い15秒の区間と、その場面から再生する YouTube のリンクも表示する

### ジョブの確認

```bash
squeue -u "$USER"
tail -f logs/<ジョブ名>-<ジョブID>.out
scancel <ジョブID>
```

## 出力

`runs/{system}_{modality}_seed{n}/`

- `best.pt` valの誤差が一番小さかった時点のモデル
- `pred_val.csv` valの予測（video_id, t, score）
- `summary.json` エポックごとの記録、最良の成績、学習時間、パラメータ数、GPU
