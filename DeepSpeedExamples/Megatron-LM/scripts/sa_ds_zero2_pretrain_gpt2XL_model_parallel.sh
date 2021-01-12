#! /bin/bash

# DP = 1, VMP = 8
# DP = 2, VMP = 4
# DP = 4, VMP = 2
# DP = 8, VMP = 1
# Change for multinode config
MP_SIZE=${MP_SIZE:-1}
SEQ_LENGTH=${SEQ_LENGTH:-1024}
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=6

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/sa_ds_zero2_pretrain_gpt2XL_model_parallel_config.json"
gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 48 \
       --hidden-size 1536 \
       --num-attention-heads 24 \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters 20 \
       --resume-dataloader \
       --train-data webtext \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --no-load-optim \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --fp16 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --deepspeed \
       --deepspeed_config ${config_json} \
       --deepspeed_sparse_attention
    "


run_cmd="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
