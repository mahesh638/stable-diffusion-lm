#!/bin/bash
set -u

DSET=${1:-cc_news/cc_news}

GPU=${2:-0}
INIT_PRETRAINED_MODEL=${3:-"False"}
USE_PRETRAINED_EMBEDDINGS=${4:-"False"}
FREEZE_EMBEDDINGS=${5:-"False"}

LR_ANNEAL_STEPS=${6:-10000}
LR=${7:-0.0001}
DIFFUSION_STEPS=${8:-2000}
NOISE_SCHEDULE=${9:-pw_lin}
BATCH_SIZE=${10:-64}
SEQ_LEN=${11:-32}

CHECKPOINT_PATH=${12:-"ckpts/${DSET}"}
TRAIN_TXT_PATH=${13:-data/${DSET}-train.txt}
VAL_TXT_PATH=${14:-data/${DSET}-test.txt}
IN_CHANNELS=${15:-128}
WEIGHT_DECAY=${16:-0.0008}
SEED=${17:-42}
DROPOUT=${18:-0.1}
NUM_HEADS=${19:-4}
CONFIG_NAME=${20:-"bert-base-uncased"}


NOTES=${18:-"Pre-trained models, pre-trained embeddings, embeddings not frozen"}

mkdir -p ${CHECKPOINT_PATH}


ARGS=(--checkpoint_path ${CHECKPOINT_PATH}
    --save_interval 2000 --lr ${LR}
    --batch_size ${BATCH_SIZE}
    --diffusion_steps ${DIFFUSION_STEPS}
    --noise_schedule ${NOISE_SCHEDULE}
    --sequence_len ${SEQ_LEN} --seed ${SEED}
    --dropout ${DROPOUT} --in_channel ${IN_CHANNELS}
    --out_channel ${IN_CHANNELS}
    --weight_decay ${WEIGHT_DECAY}
    --predict_xstart True
    --train_txt_path ${TRAIN_TXT_PATH}
    --dataset ${DSET}
    --val_txt_path ${VAL_TXT_PATH}
    --num_heads ${NUM_HEADS}
    --config_name ${CONFIG_NAME}
    --init_pretrained ${INIT_PRETRAINED_MODEL}
    --freeze_embeddings ${FREEZE_EMBEDDINGS}
    --use_pretrained_embeddings ${USE_PRETRAINED_EMBEDDINGS}
    --notes \""${NOTES}"\")

ARGS+=(--lr_anneal_steps $LR_ANNEAL_STEPS)


export CUDA_VISIBLE_DEVICES=$GPU && python -m src.train_infer.train "${ARGS[@]}"
