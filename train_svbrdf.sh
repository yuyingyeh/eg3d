#!/bin/bash

outdir="${1:-"~/training-runs"}"
dataRoot="${2:-"/home/yyyeh/Datasets/Data_Deschaintre18"}"
gpus="${3:-"1"}"
batch="${4:-"8"}"
gamma="${5:-"5"}"
snap="${6:-"10"}"
workers="${7:-"1"}"
glr="${8:-"0.0025"}"
dlr="${9:-"0.002"}"

cd eg3d

# L109: SvbrdfFolderDataset
# monoplane.py L45 Superresolution512
# python3 train_svbrdf.py --outdir=~/training-runs --cfg=ffhq --data=/home/yyyeh/Datasets/BRDFOriginDataset \
#   --gpus=1 --batch=4 --gamma=1 --gen_pose_cond=False --cond=False --density_reg 0 --metrics none --snap 1

# L109: SvbrdfLowFolderDataset
# monoplane.py L45 Superresolution256
# python3 train_svbrdf.py --outdir=$outdir --cfg=ffhq --data=$dataRoot --workers=$workers\
#   --gpus=$gpus --batch=$batch --gamma=$gamma --gen_pose_cond=False --cond=False --density_reg 0 --metrics none --snap $snap

# train_svbrdf.py L268 MonoPlaneNoSRGenerator 
python3 train_svbrdf_noSR.py --outdir=$outdir --cfg=ffhq --data=$dataRoot --workers=$workers\
  --gpus=$gpus --batch=$batch --gamma=$gamma --gen_pose_cond=False --cond=False --density_reg 0 --metrics none --snap $snap\
  --neural_rendering_resolution_initial 256 --glr=$glr --dlr=$dlr
#   --resume=/home/yyyeh/GitRepo/PhotoScene-private/third_party/eg3d/eg3d/network/svbrdf.pkl
