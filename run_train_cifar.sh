#!/bin/bash

#SBATCH --job-name=train_cifar
#SBATCH --partition=a100  # 或者您可以选择其他可用的 GPU 分区
#SBATCH --gres=gpu:4  # 请求 4 个 GPU
#SBATCH --time=8:00:00  # 或者您需要的时间
#SBATCH --output=job-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<yuelin.xu@cispa.de>

# 运行您的 Python 脚本
srun --container-image=pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime \
python ~/CISPA-home/Overfitting-in-adversarially-robust-deep-learning/train_cifar.py

