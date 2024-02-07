#!/bin/bash

#SBATCH --job-name=fast
#SBATCH --partition=gpu  # 或者您可以选择其他可用的 GPU 分区
#SBATCH --gres=gpu:4  # 请求 4 个 GPU
#SBATCH --output=job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<yuelin.xu@cispa.de>

# 运行您的 Python 脚本
srun --container-image=pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime \
python ~/CISPA-home/OverfittingInRML/train_fast_fgsm.py  > ~/CISPA-home/OverfittingInRML/fast.txt 2>&1