#! /bin/bash

# DP = 1, VMP = 8
# DP = 2, VMP = 4
# DP = 4, VMP = 2
# DP = 8, VMP = 1
# Change for multinode config
NUM_WORKERS=4
NUM_GPUS_PER_WORKER=16

MP_SIZE=${MP_SIZE:-16}
SEQ_LENGTH=${SEQ_LENGTH:-2048}

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

hostfile="/job/hostfile"
config_json="$script_dir/multinode_ds_zero2_pretrain_gpt2XL_model_parallel_config.json"
gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 1 \
       --hidden-size 12288 \
       --num-attention-heads 96 \
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
       --deepspeed_config ${config_json}
    "


run_cmd="deepspeed --hostfile=${hostfile} --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
