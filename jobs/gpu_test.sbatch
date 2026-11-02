#!/bin/bash
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --time=00:20:00

# GPUテスト：小さいGPUで数エポックだけ学習して、GPUで動くかを確かめる
# 投入（プロジェクトのルートで）:
#   MoSuの一部で（先にログインノードで tools/prepare_mosu.py を実行して data_mosu/ を作っておく）
#     sbatch -J gpu_test -p a100_1g --gres=gpu:1g.10gb:1 --qos=low jobs/gpu_test.sbatch
#   偽データで
#     sbatch -J gpu_test -p a100_1g --gres=gpu:1g.10gb:1 --qos=low --export=ALL,DATA=data_fake jobs/gpu_test.sbatch
#   自分のモデルで（SYSTEM=自分のシステム名）
#     sbatch -J gpu_test -p a100_1g --gres=gpu:1g.10gb:1 --qos=low --export=ALL,SYSTEM=fast jobs/gpu_test.sbatch

set -euo pipefail

SYSTEM="${SYSTEM:-_smoke}"
DATA="${DATA:-data_mosu}"
MODALITY="${MODALITY:-visual}"

echo "ジョブID: $SLURM_JOB_ID"
echo "実行ノード: $(hostname)"
echo "パーティション: $SLURM_JOB_PARTITION"
nvidia-smi -L

if [[ ! -f .venv/bin/activate ]]; then
  echo "エラー: .venv がない。ログインノードで uv sync --locked を実行してから投入すること"
  exit 1
fi
source .venv/bin/activate

python -u -c "import torch; print('torch', torch.__version__, '/ cuda:', torch.cuda.is_available(), '/', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPUなし')"

if [[ ! -d "$DATA" ]]; then
  if [[ "$DATA" == data_fake ]]; then
    python -u tools/make_fake_data.py --out data_fake --modality "$MODALITY"
  else
    echo "エラー: $DATA がない。ログインノードで python tools/prepare_mosu.py --remote を実行してから投入すること"
    exit 1
  fi
fi
echo "データ: $DATA（train $(grep -c . "$DATA/splits/train.txt")本 / val $(grep -c . "$DATA/splits/val.txt")本）"

python -u train.py --system "$SYSTEM" --modality "$MODALITY" --seed 0 \
  --data_dir "$DATA" --out_dir runs_test --max_epochs 3

echo "--- summary.json ---"
python -u -c "
import json
s = json.load(open('runs_test/${SYSTEM}_${MODALITY}_seed0/summary.json'))
print('gpu:', s['gpu'])
print('学習時間:', s['train_sec'], '秒 / パラメータ数:', s['n_params'], '/ 最良のval誤差:', round(s['best_val_error'], 4))
"
