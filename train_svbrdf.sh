#!/bin/bash

outdir="${1:-"~/training-runs"}"
dataRoot="${2:-"/home/yyyeh/Datasets/Data_Deschaintre18"}"
gpus="${3:-"1"}"
batch="${4:-"4"}"
gamma="${5:-"40"}"
snap="${6:-"1"}"
workers="${7:-"1"}"

cd eg3d

# L109: SvbrdfFolderDataset
# monoplane.py L45 Superresolution512
# python3 train_svbrdf.py --outdir=~/training-runs --cfg=ffhq --data=/home/yyyeh/Datasets/BRDFOriginDataset \
#   --gpus=1 --batch=4 --gamma=1 --gen_pose_cond=False --cond=False --density_reg 0 --metrics none --snap 1

# L109: SvbrdfLowFolderDataset
# monoplane.py L45 Superresolution256
python3 train_svbrdf.py --outdir=$outdir --cfg=ffhq --data=$dataRoot --workers=$workers\
  --gpus=$gpus --batch=$batch --gamma=$gamma --gen_pose_cond=False --cond=False --density_reg 0 --metrics none --snap $snap
