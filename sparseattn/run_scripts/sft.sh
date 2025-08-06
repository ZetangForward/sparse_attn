#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --output=./joblog/%x-%A_%a.out
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=680G
#SBATCH --time=0-10:00:00

# !!! Activate conda environment here !!!
# >>> conda activate prulong

# Model and training configuration
model=${MODEL:-"meta-llama/Llama-3.1-8B"} # !!! -> should point to a prulong model checkpoint
bsz=${BSZ:-64}
seq=${SEQ:-1}
lr=${LR:-2e-5}
steps=${STEPS:-250}
save_steps=${SAVE:-250}
warmup=${WARMUP:-0.05}
suffix=${SUFFIX:-""}
overrides=${OVERRIDES:-""}

# FSDP configuration
# 0=Disable, 1=FULL_SHARD, 2=SHARD_GRAD_OP, 3=NO_SHARD, 4=HYBRID_SHARD, 5=HYBRID_SHARD_ZERO2
fsdp=${FSDP:-"1"}
gc=${GC:-"1"}

# SFT-specific arguments
max_toks=${MAX_TOKS:-131072}
start_head_sparsity=${START_HEAD_SPARSITY:-0.0} # !!! -> should be sparsity of checkpoint
end_head_sparsity=${END_HEAD_SPARSITY:-0.0} # !!! -> should be sparsity of checkpoint (same as start)
mask_learning_rate=${MASK_LEARNING_RATE:-0} 
reg_learning_rate=${REG_LEARNING_RATE:-0} 
warmup_type=${WARMUP_TYPE:-"linear"}
sparsity_warmup_ratio=${SPARSITY_WARMUP_RATIO:-0.8}
disable_linear_reg_term=${DISABLE_LINEAR_REG_TERM:-false}
context_window_if_toggled=${CONTEXT_WINDOW_IF_TOGGLED:-1024}
freeze_weights=${FREEZE_WEIGHTS:-false}
freeze_masks=${FREEZE_MASKS:-true}
min_lr_ratio=${MIN_LR_RATIO:-0.1}
seq_parallel_size=${SEQ_PARALLEL_SIZE:-2}

# Streaming configuration
toggle_type=${TOGGLE_TYPE:-"streaming"}
sink_size=${SINK_SIZE:-128}

# Dataset configuration
dataset=${DATASET:-"datasets/sft"}

# SFT domain configuration (space-separated list)
domains=${DOMAINS:-"ultrachat@1.0"}

# Create run name
extra_name=""
if [[ $freeze_weights == "true" ]]; then
    extra_name="${extra_name}_wfrozen"
fi
if [[ $freeze_masks == "true" ]]; then
    extra_name="${extra_name}_mfrozen"
fi

timestamp=$(date +%Y-%m-%d_%H-%M-%S)
run_name="sft_$(basename $model)_bsz${bsz}_steps${steps}_lr${lr}_warmup${warmup}_sp${end_head_sparsity}_cw${context_window_if_toggled}_mlr${mask_learning_rate}_rlr${reg_learning_rate}${suffix}${extra_name}_${timestamp}"

out_dir="checkpoints/$run_name"
mkdir -p $out_dir
nvidia-smi

# Calculate GPU and node configuration
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    num_gpus=$(nvidia-smi -L | wc -l)
else
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
num_gpus=${NUM_GPUS:-$num_gpus}

num_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | wc -l)
if [ $num_nodes == 0 ]; then
    num_nodes=1
fi
num_nodes=${NUM_NODES:-$num_nodes}

# Setup distributed training
if [ $num_nodes -gt 1 ]; then
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    master_addr=${MASTER_ADDR:-$master_addr}

    header="srun torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$master_addr:56321 \
    --nnodes=$num_nodes \
    --nproc-per-node=$num_gpus \
    -m training.lh_train_language_model"
else
    master_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

    header="torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:$master_port \
    --nnodes=1 \
    --nproc-per-node=$num_gpus \
    -m training.lh_train_language_model"
fi

accu=$(($bsz / $seq / $num_gpus / $num_nodes))

echo "num_nodes=${num_nodes} master_addr=${master_addr} master_port=${master_port} num_gpus=${num_gpus}"

# Environment variables
export OMP_NUM_THREADS=$num_gpus
export WANDB_PROJECT="prulong"
export WANDB_DIR=$out_dir
export WANDB_MODE="offline"
export TOKENIZERS_PARALLELISM=true
export LOGIT_BLOCK_SIZE=2048

# Training arguments
base_arguments=(
    --report_to wandb
    --do_train

    --model_name $model
    --tokenizer_name $model

    --run_name $run_name
    --output_dir $out_dir
    --config_overrides_json "$overrides"
    --gradient_accumulation_steps $accu
    --per_device_train_batch_size $seq
    --per_device_eval_batch_size $seq

    --bf16
    --learning_rate $lr
    --min_lr_ratio $min_lr_ratio
    --lr_scheduler_type cosine
    --max_grad_norm 1.0
    --adam_beta1 0.9
    --adam_beta2 0.95
    --weight_decay 0.1
    --warmup_ratio $warmup
    --optim adamw_torch

    --logging_steps 1
    --log_level info

    --max_steps $steps
    --save_steps $save_steps
    --dataloader_num_workers 1

    --disable_tqdm true
    --use_fast_tokenizer false
    --remove_unused_columns false
    --ddp_find_unused_parameters false

    --cuda_empty_cache

    # SFT-specific arguments
    --per_device_max_tokens $max_toks
    --seq_parallel_size $seq_parallel_size
    --start_head_sparsity $start_head_sparsity
    --end_head_sparsity $end_head_sparsity
    --mask_learning_rate $mask_learning_rate
    --reg_learning_rate $reg_learning_rate
    --warmup_type $warmup_type
    --sparsity_warmup_ratio $sparsity_warmup_ratio
    --disable_linear_regularization_term $disable_linear_reg_term
    --context_window_if_toggled $context_window_if_toggled
    --freeze_non_mask_parameters $freeze_weights
    --freeze_mask_parameters $freeze_masks
    --should_log_loss true
    --save_total_limit 3

    # Streaming configuration
    --toggle_type $toggle_type
    --sink_size $sink_size

    # SFT-specific configuration
    --apply_instruct_masks
    --token_scaled_loss
    --load_masks_sparsity 0.0
)

# FSDP configuration
if [ $fsdp -ne 0 ]; then
    export FSDP_SHARDING_STRATEGY=$fsdp
    base_arguments+=( --fsdp "auto_wrap" )
    export FSDP_STATE_DICT_TYPE="FULL_STATE_DICT"
fi

# Gradient checkpointing
if [ $gc -ne 0 ]; then
    base_arguments+=( --gradient_checkpointing )
fi

# Add dataset paths
base_arguments+=( --tokenized_mds_train )
for domain in $domains; do
    base_arguments+=( $dataset/$domain )
done

base_arguments+=( $@ )

echo "Command: ${header} ${base_arguments[@]}"
${header} "${base_arguments[@]}" 2>&1 | tee -a $out_dir/log.out 