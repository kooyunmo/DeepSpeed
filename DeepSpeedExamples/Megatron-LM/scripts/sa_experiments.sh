#!/bin/bash

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
config_json="$script_dir/sa_ds_zero2_pretrain_gpt2XL_model_parallel_config.json"
NUM_GPUS_PER_WORKER=16
BATCH_SIZE=1

for SEQ_LEN in 256 512 1024 2048
  do
    for MP_SIZE in 1 2 4 8
      do
        DP_SIZE=$((NUM_GPU_PER_WORKERS / MP_SIZE))
        TRAIN_BATCH_SIZE=$((BATCH_SIZE "*" DP_SIZE))
        echo ${TRAIN_BACH_SIZE}

cat > ${config_json} <<-JSON
{
  "train_batch_size": ${TRAIN_BATCH_SIZE},
  "gradient_accumulation_steps": 1,
  "steps_per_print": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.00015,
      "weight_decay": 1e-2
    }
  },
  "zero_optimization": {
    "stage": 2,
    "cpu_offload": false,
    "contiguous_gradients": false,
    "overlap_comm": false
  },
  "zero_allow_untested_optimizer": true,
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": false
  },
  "wall_clock_breakdown": true,
  "sparse_attention": {
    "mode": "variable",
    "attention": "unidirectional"
  }
}
JSON

        BATCH_SIZE=${BATCH_SIZE} MP_SIZE=${MP_SIZE} SEQ_LENGTH=${SEQ_LEN} bash scripts/sa_ds_zero2_pretrain_gpt2XL_model_parallel.sh |& tee mp${MP_SIZE}_sl${SEQ_LEN}.log
        gzip trace-*.json mp${MP_SIZE}_sl${SEQ_LEN}.gz
      done
  done
