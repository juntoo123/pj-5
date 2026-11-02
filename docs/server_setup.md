# 計算サーバでの手順（下準備からGPUテストまで）

IS計算サーバ利用ガイドの内容を、このプロジェクト用にまとめたもの。
サーバのアドレスなど細かい値は、利用ガイドを見ること（このファイルは公開リポジトリに置くため書かない）。

## 0. 全体像：3つの場所

```
① 自分のPC
   │  ssh でログイン（学内ネット or VPN が必要）
   ▼
② ログインノード（受付）
   │  ここで：git、uv、データの準備、ジョブの投入
   │  ここでしない：学習（重い処理は禁止）
   │
   │  sbatch / srun で「この仕事をGPUでやって」と申し込む
   ▼
③ 計算ノード（GPUが刺さったマシン）
      ここで：train.py が動く
      ここにない：uv（.venv を有効化して python を直接使う）
```

- ホームディレクトリ（`~`）は②と③で共有されている。②で置いたファイルは、③からそのまま見える
- 今どこにいるかは、プロンプトのホスト名でわかる。計算ノードに入ると `isws-dnnserver02` などに変わる

| やること | 場所 |
|---|---|
| コードを書く・push する | ① 自分のPC（または②で直接編集） |
| `git pull`、`uv lock`、`uv sync` | ② ログインノード |
| MoSu の変換（`tools/prepare_mosu.py`） | ② ログインノード |
| `sbatch`、`squeue`、`scancel`、`tail` | ② ログインノード |
| 学習（`train.py`） | ③ 計算ノード（ジョブの中で自動で動く） |

---

## 1. 最初に1回だけやる下準備

### 1-1. サーバにログインする（①で）

学内ネットにつなぐか、VPNで学内ネットにつないでから、次を実行する。

```bash
ssh s<学籍番号>@<サーバのアドレス>
```

- 初回は、メールで届いた初期パスワードでログインし、新しいパスワードに変更する
- `WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!` と出たら、①で `ssh-keygen -R <サーバのアドレス>` を実行してからやり直す
- ログアウトは `exit`

### 1-2. パスワードなしでログインできるようにする（公開鍵認証）

**①で**鍵を作り、公開鍵を表示してコピーする。

```bash
ssh-keygen -t ed25519        # 聞かれたら Enter でよい（すでに鍵があれば不要）
cat ~/.ssh/id_ed25519.pub    # 表示された1行をコピー
```

**②で**、コピーした1行を登録する。

```bash
mkdir -p ~/.ssh
nano ~/.ssh/authorized_keys  # 貼り付けて保存（Ctrl+O → Enter → Ctrl+X）
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
exit
```

もう一度 `ssh s<学籍番号>@<サーバのアドレス>` して、パスワードを聞かれなければ完了。

- `id_ed25519`（`.pub` が付いていない方）は秘密鍵。サーバにコピーしたり、人に渡したりしない

### 1-3. サーバから GitHub につなげるようにする（②で）

1-2 とは向きが逆（サーバ → GitHub）なので、サーバ上でもう1つ鍵を作る。

```bash
ssh-keygen -t ed25519        # Enter でよい
cat ~/.ssh/id_ed25519.pub    # 表示された1行をコピー
```

GitHub の右上アイコン → **Settings → SSH and GPG keys → New SSH key** に貼り付けて保存する。

```bash
ssh -T git@github.com        # 「Hi ユーザー名!」と出ればOK（初回は yes と入力）
git config --global user.name "自分の名前"
git config --global user.email "GitHubに登録したメールアドレス"
```

### 1-4. プロジェクトを持ってくる（②で）

```bash
cd ~
git clone git@github.com:<オーナー名>/<リポジトリ名>.git
cd <リポジトリ名>
```

以降、`~/<リポジトリ名>` がプロジェクトのフォルダ。

### 1-5. Python の環境を作る（②で）

```bash
uv --version                 # uv が使えるか確認
uv lock                      # uv.lock がまだない、または pyproject.toml を変えたときだけ
uv sync --locked             # uv.lock のとおりに .venv を作る（数分かかる）
```

- `uv.lock` をチームで共有している場合は、`uv lock` はせず `uv sync --locked` だけ
- 自分が `uv lock` で `uv.lock` を作ったり更新したりしたら、コミットしてチームに共有する

  ```bash
  git add uv.lock
  git commit -m "uv.lock を更新"
  git push
  ```

### 1-6. GPU が見えるか確かめる（③に30分だけ入る）

```bash
srun -p a100_1g --gres=gpu:1g.10gb:1 --qos=inter --pty bash   # ③に入る（空くまで待つことがある）
hostname                                                     # isws-dnnserver02 などになっていれば③
cd ~/<リポジトリ名>
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
exit                                                         # ②に戻る
```

`True NVIDIA A100 80GB PCIe MIG 1g.10gb` のように出ればOK。

- `--qos=inter` は対話専用（30分まで、1人1つ）。学習には使わない
- `--qos` は所属で変わる。研究室外のユーザーの場合はガイド2.2を確認すること

---

## 2. MoSu の一部を準備する（②で）

```bash
cd ~/<リポジトリ名>
source .venv/bin/activate
python tools/prepare_mosu.py --modality visual --n_train 200 --n_val 50 --remote
```

- train から200本、val から50本を選んで `data_mosu/` に変換する（test は使わない）
- `--remote`：45GBの特徴量ファイルはダウンロードせず、選んだ動画だけネット越しに読む（1本数秒）
- 分割・メタデータ・正解（約230MB）は `mosu/` に自動でダウンロードされる
- `--cluster 0〜9` でジャンルを絞れる
- 正解は仮のもの（MoSu の `gt_score` をそのまま使う）。正式な変換は評価・統合担当が決める

サーバから Hugging Face につながらない場合は、自分のPC や Colab で同じコマンドを実行して `data_mosu/` を作り、①からサーバへ送る。

```bash
scp -r data_mosu s<学籍番号>@<サーバのアドレス>:~/<リポジトリ名>/
```

---

## 3. GPU テストのジョブを投入する（②で）

```bash
cd ~/<リポジトリ名>              # 必ずプロジェクトのルートで投入する
sbatch -J gpu_test -p a100_1g --gres=gpu:1g.10gb:1 --qos=low jobs/gpu_test.sbatch
```

`Submitted batch job 12345` のようにジョブIDが出る。

| やりたいこと | コマンド |
|---|---|
| 自分のジョブの状態を見る | `squeue -u "$USER"`（`PD` は順番待ち、`R` は実行中、一覧から消えたら終了） |
| 出力をリアルタイムで見る | `tail -f logs/gpu_test-<ジョブID>.out`（終わるときは Ctrl+C） |
| 出力を全部見る | `cat logs/gpu_test-<ジョブID>.out` |
| ジョブを取り消す | `scancel <ジョブID>` |

オプションの付け足し：

- 自分のモデルで試す：`--export=ALL,SYSTEM=fast`（`systems/fast.py` を使う）
- 偽データで試す：`--export=ALL,DATA=data_fake`

### 成功したときの出力の最後

```
gpu: NVIDIA A100 80GB PCIe MIG 1g.10gb
学習時間: ○○ 秒 / パラメータ数: ○○ / 最良のval誤差: ○○
```

結果は `runs_test/<システム名>_visual_seed0/` に、`best.pt`・`pred_val.csv`・`summary.json` として保存される。

---

## 4. 2回目以降の流れ

```bash
# ①で：コードを書いて push
git add systems/fast.py
git commit -m "変更の説明"
git push

# ①から②へ
ssh s<学籍番号>@<サーバのアドレス>
cd ~/<リポジトリ名>
git pull
uv sync --locked                 # pyproject.toml か uv.lock が変わったときだけ
sbatch ...                       # ジョブを投入
squeue -u "$USER"
tail -f logs/<ジョブ名>-<ジョブID>.out
```

- ②で直接コードを編集してもよい（`nano` や、VS Code の Remote-SSH 拡張機能を使うと楽）。その場合は②で `git add` / `commit` / `push` する
- 始める前に `git pull` を忘れない

---

## 5. よくあるつまずき

| 症状 | 原因と対処 |
|---|---|
| `ssh` がつながらない | 学内ネット / VPN につながっているか確認 |
| `Permission denied (publickey)`（GitHub） | 1-3 の鍵の登録を確認。`ssh -T git@github.com` で確かめる |
| `.venv がない` と出てジョブが終わる | ②で `uv sync --locked` をしていない |
| `uv sync --locked` が lock が古いと言って止まる | `pyproject.toml` が変わっている。`uv lock` してから再実行し、`uv.lock` をコミット |
| ジョブが投入時に拒否される | `-p` と `--gres` の組み合わせ、`--qos` を確認（`a100_1g` は `inter` / `low` / `idle` のみ） |
| ずっと `PD`（順番待ち） | GPU が混んでいる。待つ |
| `gpu: cpu` と出る | GPU が見えていない。`.out` の `cuda:` の行と `nvidia-smi` の出力を確認 |
| `CUDA out of memory` | GPU のメモリ不足。`--batch_size` を小さくするか、大きいパーティションを使う |
| `Killed` と出て止まる | メインメモリ不足。動画の本数を減らすか、大きいパーティションを使う（`a100_1g` は30GiB） |
| `.out` に何も出てこない | ジョブがまだ始まっていない（`squeue` で確認） |

`low` / `idle` のジョブは、優先度の高いジョブに中断されることがある。中断されると最初からやり直しになる。
