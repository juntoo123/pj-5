#!/bin/bash
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append

# 本番の学習：seed 0, 1, 2 を順番に実行する
# 投入（プロジェクトのルートで）:
#   sbatch -J train_fast_visual -p a100_3g --gres=gpu:3g.40gb:1 \
#     --export=ALL,SYSTEM=fast,MODALITY=visual jobs/train.sbatch
# データは標準で data/（MoSuの一部で試すときは --export に DATA=data_mosu を足す）

set -euo pipefail

SYSTEM="${SYSTEM:-fast}"
MODALITY="${MODALITY:-visual}"
DATA="${DATA:-data}"

echo "ジョブID: $SLURM_JOB_ID"
echo "実行ノード: $(hostname)"
nvidia-smi -L

source .venv/bin/activate
for seed in 0 1 2; do
  python -u train.py --system "$SYSTEM" --modality "$MODALITY" --seed "$seed" --data_dir "$DATA"
done
