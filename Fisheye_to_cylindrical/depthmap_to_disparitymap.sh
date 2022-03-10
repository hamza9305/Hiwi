#!/bin/bash

CONDA_ENV_NAME="anynet"

source ~/anaconda3/etc/profile.d/conda.sh || exit $?
conda activate "$CONDA_ENV_NAME" || exit $?


[ -z "$SAVE_PATH" ] && SAVE_PATH=//mnt/data/users/haahm/theostereo/valid/disp_occ_0_webp
[ -z "$DATASET" ] && DATASET=/mnt/data/users/haahm/theostereo/valid/depth_exr_abs

#TODO:
export CUDA_VISIBLE_DEVICES=2,3

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


nice python depthmap_to_disparity.py --indir "${DATASET}" --outdir ${SAVE_PATH} 
