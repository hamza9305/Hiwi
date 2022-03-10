#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh || exit $?
conda activate anynet || exit $?

#current_env=$(conda info | grep "active environment" | cut -f2 -d":" | sed 's/ //g')
#[ "$current_env" != "anynet" ] && echo "Please activate the conda environment \"anynet\" (python=3.6)" && exit 1

[ -z "$SAVE_PATH" ] && SAVE_PATH=results/scenenn
[ -z "$DATASET" ] && DATASET=/mnt/data/seuj/datasets/SceneNN/scenenn_seg_76_raw/fbx/anynet/
[ -z "$MAX_DISP" ] && MAX_DISP=192
[ -z "$EPOCHS" ] && EPOCHS=50
[ -z "$SPN_START" ] && SPN_START=5
[ -z "$TRAIN_BSIZE" ] && TRAIN_BSIZE=24
[ -z "$TEST_BSIZE" ] && TEST_BSIZE=24
[ -z "$CUDA_DEVICE_ORDER" ] && export CUDA_DEVICE_ORDER="PCI_BUS_ID"

python finetune.py --maxdisp ${MAX_DISP} --with_spn --datapath "${DATASET}" \
    --save_path ${SAVE_PATH} --datatype other --pretrained checkpoint/kitti2015_ck/checkpoint.tar \
    --split_file checkpoint/kitti2015_ck/split.txt --train_bsize=${TRAIN_BSIZE} --test_bsize=${TEST_BSIZE} --resume ${SAVE_PATH}/checkpoint.tar \
    --epochs=${EPOCHS} --dontnormgtdisp --dontclip --cosanneal --start_epoch_for_spn=${SPN_START} $@
