#!/bin/bash -l

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_NAME="prulong_32k"
DEFAULT_NUM_GPUS=1
DEFAULT_TIME="15:00:00"
DEFAULT_MEM="60G"
SLURM_PARTITION="pli-c"

# Common tasks across all scripts
COMMON_TASKS=(
    "longproc_addon/configs/html_to_tsv.yaml"
    "longproc_addon/configs/travel_planning.yaml"
    "configs/recall.yaml"
    "configs/rerank.yaml"
    "configs/rag.yaml"
    "configs/icl.yaml"
    "configs/longqa.yaml"
    "configs/summ.yaml"
)
# ============================================================================
# Job submission function
# ============================================================================
submit_job() {
# Extract names for job organization
TASK_NAME=$(basename $TASK .yaml)   
MODEL_NAME=$(basename $MODEL)

# Set up output directory and completion tracking
OUT_DIR=outputs/${MODEL_NAME}/outputs_sp${SPARSITY}_pf${PREFILL}_tg
if [ -f ${OUT_DIR}/.${TASK_NAME}.completed ]; then
    echo "Skipping completed: ${MODEL_NAME} - ${TASK_NAME} - sp${SPARSITY}"
    return
fi

# Resource allocation - some tasks need more GPUs
NUM_GPUS=${NUM_GPUS:-$DEFAULT_NUM_GPUS}
TIME=${TIME:-$DEFAULT_TIME}
MEM=${MEM:-$DEFAULT_MEM}

# Some tasks may need more resources
if [ "$TASK_NAME" == "rag" ]; then
    NUM_GPUS=2
fi

# Build command and job name
CMD="python eval.py --config ${TASK} --model_name_or_path $MODEL --tokenizer_name $MODEL --output_dir $OUT_DIR $EXTRA"
JOB_NAME="${SCRIPT_NAME}_${TASK_NAME}_sp${SPARSITY}_${MODEL_NAME}"

# Check for existing jobs to avoid duplicates
# if squeue -h --me -n ${JOB_NAME} | grep -q .; then
#     echo "Job already queued: ${JOB_NAME}"
#     return
# fi

echo "!!! Submitting: ${JOB_NAME}"

# Submit SLURM job
# sbatch<<EOT
#!/bin/bash -l
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --output=./joblog/%x-%A_%a.out                          
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}

# !!! activate the correct environment here !!!
# >>> conda activate prulong

echo "Command: $CMD"
echo "Resources: ${NUM_GPUS} GPUs, ${MEM} memory, ${TIME} time"
CUDA_LAUNCH_BLOCKING=1 $CMD && echo "${MODEL_NAME} - ${TASK_NAME}" > ${OUT_DIR}/.${TASK_NAME}.completed
# EOT

}

# ============================================================================
# Main execution
# ============================================================================
export OUTLINES_CACHE_DIR=/data/qqt/project/PruLong-main/tmp/outlines

# Script-specific configuration
prulonged_models=( # !!! this should point to one or more prulong model checkpoints
    "/data/qqt/project/PruLong-main/prulong/checkpoints/masksonly_Meta-Llama-3.1-8B-Instruct_bsz16_steps1000_lr1e-5_warmup0.1_sp0.7_cw1024_mlr1.0_rlr1.0_wfrozen"
)
PREFILL=32768

for SPARSITY in 0.7; do # 0.0 -> original model
    for TASK in ${COMMON_TASKS[@]}; do
        for MODEL in ${prulonged_models[@]}; do 
            MASKS=${MODEL}/masks_sp${SPARSITY}.tsv
            EXTRA="--no_torch_compile --duoattn $MASKS --duoattn_sparsity $SPARSITY --duoattn_chunk_prefilling $PREFILL"

            TASK=$TASK MODEL=$MODEL EXTRA=$EXTRA SPARSITY=$SPARSITY submit_job
        done
    done 
done