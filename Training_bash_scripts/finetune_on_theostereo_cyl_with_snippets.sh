#!/bin/bash

CONDA_ENV_NAME="anynet"

source ~/anaconda3/etc/profile.d/conda.sh || exit $?
conda activate "$CONDA_ENV_NAME" || exit $?

#export LD_LIBRARY_PATH=~/anaconda3/envs/anynet/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
#export LD_LIBRARY_PATH=~/anaconda3/pkgs/pytorch-1.9.0-py3.9_cuda11.1_cudnn8.0.5_0/lib/python3.9/site-packages/torch/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
#export LD_LIBRARY_PATH=/opt/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} # for arch only

export LD_PRELOAD=/usr/lib/libstdc++.so${LD_PRELOAD:+:${LD_PRELOAD}}
#export LD_PRELOAD=~/anaconda3/envs/anynet/lib/libpython3.9.so${LD_PRELOAD:+:${LD_PRELOAD}}
#ldd models/spn_t1/gate_lib/gaterecurrent2dnoind_cuda.cpython-39-x86_64-linux-gnu.so | grep libstdc


[ -z "$SAVE_PATH" ] && SAVE_PATH=results/theostereo_cyl_with_snippets
[ -z "$DATASET" ] && DATASET=/mnt/data_on_nvme2/home/theostereo_Pano
[ -z "$MAX_DISP" ] && MAX_DISP=192
[ -z "$EPOCHS" ] && EPOCHS=200
[ -z "$SPN_START" ] && SPN_START=2
[ -z "$TRAIN_BSIZE" ] && TRAIN_BSIZE=48 # @haahm: you can try to increase the training batch size
[ -z "$TEST_BSIZE" ] && TEST_BSIZE=6 # @haahm: you can try to increase the testing/validation batch size
[ -z "$CUDA_DEVICE_ORDER" ] && export CUDA_DEVICE_ORDER="PCI_BUS_ID" 
[ -z "$NUM_LOADER_WORKER" ] && NUM_LOADER_WORKER="20" # @haahm: increase or decrease the CPU dataloader threads as you wish
[ -z "$CHKPNT" ] && CHKPNT=checkpoint/sceneflow/sceneflow.tar
[ -z "$RESUME" ] && RESUME=results/theostereo_cyl_with_snippets


#TODO:
export CUDA_VISIBLE_DEVICES=2

if [ -z "$CUDA_VISIBLE_DEVICES" ];
then
    nvidia-smi
    YELLOW='\033[1;33m'
    NC='\033[0m'
    echo -e "$YELLOW" 2>&1
    echo "    ---------------------------------------------------------------------" 2>&1
    echo "    Please select which of the GPUs above you want to use." 2>&1
    echo "    E.g. If you use GPU 2 and 3, you can select them by:" 2>&1
    echo "    export CUDA_VISIBLE_DEVICES=2,3" 2>&1
    echo "    Include this command into the script in $0" 2>&1
    echo "    ---------------------------------------------------------------------" 2>&1
    echo -e "$NC" 2>&1
    exit 1
fi


# @haahm: Do we still need to resize? If so, please use:
#MISC= --resize=1024x1024

# maybe interesting {
#MISC=${MISC}\ --num_test_imgs_to_store=10 --num_train_imgs_to_store=10
# }

nice python finetune.py --maxdisp ${MAX_DISP} --with_spn --datapath "${DATASET}" --no_tensorboard --resize=1024x1024 \
    --save_path ${SAVE_PATH} --datatype theo+ --pretrained ${CHKPNT} --resume ${RESUME} --num_test_imgs_to_store 10 \
    --train_bsize=${TRAIN_BSIZE} --test_bsize=${TEST_BSIZE} --num_train_imgs_to_store 10 \
    --resume ${SAVE_PATH}/checkpoint.tar \
    --epochs=${EPOCHS} --start_epoch_for_spn=${SPN_START} --cosanneal \
    --numloaderworker ${NUM_LOADER_WORKER} --dontnormgtdisp $MISC $@

# DATASET structure:
# .
# ├── test
# │   ├── disp_occ_0_webp
# │   ├── img_stereo_webp
# │   └── img_webp
# ├── train
# │   ├── disp_occ_0_webp
# │   ├── img_stereo_webp
# │   └── img_webp
# └── valid
#     ├── disp_occ_0_webp
#     ├── img_stereo_webp
#     └── img_webp
# 
# WHEREAS:
# img_webp contains the images of the left camera
# img_stereo_webp contains the images of the right camera
# disp_occ_0_webp contains the disparity maps
