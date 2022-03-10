#!/bin/bash

# HINT: First 50% of Theodore+ is THEOStereo
# This script was used for "A Study on the Influence of Omnidirectional Distortion on CNN-based Stereo Vision"

source ~/anaconda3/etc/profile.d/conda.sh || exit $?
conda activate anynet || exit $?

#current_env=$(conda info | grep "active environment" | cut -f2 -d":" | sed 's/ //g')
#[ "$current_env" != "anynet" ] && echo "Please activate the conda environment \"anynet\" (python=3.6)" && exit 1

[ -z "$SAVE_PATH" ] && SAVE_PATH=results/theodore_plus_omni
[ -z "$DATASET" ] && DATASET=/mnt/data/seuj/theodore_plus/splits_symlinks/seed_20200805__numimgs_0062500__train_80__valid_10__test_10
#[ -z "$DATASET" ] && DATASET=/home/seuj/Data/perspective
[ -z "$MAX_DISP" ] && MAX_DISP=192
[ -z "$EPOCHS" ] && EPOCHS=100
[ -z "$SPN_START" ] && SPN_START=2
[ -z "$TRAIN_BSIZE" ] && TRAIN_BSIZE=8
[ -z "$TEST_BSIZE" ] && TEST_BSIZE=6
[ -z "$CUDA_DEVICE_ORDER" ] && export CUDA_DEVICE_ORDER="PCI_BUS_ID"
[ -z "$NUM_LOADER_WORKER" ] && NUM_LOADER_WORKER="4"
#[ -z "$CHKPNT" ] && CHKPNT=checkpoint/kitti2015_ck/checkpoint.tar
[ -z "$CHKPNT" ] && CHKPNT=checkpoint/sceneflow/sceneflow.tar

# THEO+ lite {
MISC=--maximgstrain=25000\ --maximgsval=3125\ --resize=1024x1024
# }

# maybe interesting {
# --num_test_imgs_to_store=10 --num_train_imgs_to_store=10
# }

python finetune.py --maxdisp ${MAX_DISP} --with_spn --datapath "${DATASET}" \
    --save_path ${SAVE_PATH} --datatype theo+ --pretrained ${CHKPNT} \
    --split_file checkpoint/kitti2015_ck/split.txt --train_bsize=${TRAIN_BSIZE} --test_bsize=${TEST_BSIZE} \
    --resume ${SAVE_PATH}/checkpoint.tar \
    --epochs=${EPOCHS} --start_epoch_for_spn=${SPN_START} --cosanneal \
    --numloaderworker ${NUM_LOADER_WORKER} --dontnormgtdisp $MISC $@
