#!/bin/bash -l

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_NAME="l2_sweep"
DEFAULT_NUM_GPUS=1
DEFAULT_TIME="23:00:00"
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
OUT_DIR=outputs/${MODEL_NAME}_l2/L2_loc${LOCAL}_max${MAX_CAPACITY}_outputs${SUFFIX}
if [ -f ${OUT_DIR}/.${TASK_NAME}.completed ]; then
    echo "Skipping completed: ${MODEL_NAME} - ${TASK_NAME} - ${METHOD} - max${MAX_CAPACITY}"
    return
fi

# Resource allocation (can be overridden by caller)
NUM_GPUS=${NUM_GPUS:-$DEFAULT_NUM_GPUS}
TIME=${TIME:-$DEFAULT_TIME}
MEM=${MEM:-$DEFAULT_MEM}

# Build command and job name
CMD="python eval.py --config ${TASK} --model_name_or_path $MODEL --tokenizer_name $MODEL --minference $METHOD --output_dir $OUT_DIR $EXTRA"
JOB_NAME="${SCRIPT_NAME}_${TASK_NAME}_${METHOD}_${LOCAL}_${MODEL_NAME}"

# Check for existing jobs to avoid duplicates
if squeue -h --me -n ${JOB_NAME} | grep -q .; then
    echo "Job already queued: ${JOB_NAME}"
    return
fi

echo "!!! Submitting: ${JOB_NAME}"

# Submit SLURM job
sbatch<<EOT
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
EOT

}

# ============================================================================
# Main execution
# ============================================================================
export OUTLINES_CACHE_DIR=/tmp/outlines

# Script-specific configuration
MODEL=meta-llama/Llama-3.1-8B-Instruct
PREFILL_SIZE=32768
METHOD="l2"
LOCAL=64

for TASK in ${COMMON_TASKS[@]}; do 
    for MAX_CAPACITY in 512 1024 2048 4096 8192 16384; do             
        # Build extra arguments
        EXTRA="--no_torch_compile --minference_max_capacity_prompt $MAX_CAPACITY --minference_window_size $LOCAL --minference_chunk_prefilling $PREFILL_SIZE"
        SUFFIX="_local${LOCAL}_tg"
    
        TASK=$TASK MODEL=$MODEL SUFFIX=$SUFFIX EXTRA=$EXTRA METHOD=$METHOD LOCAL=$LOCAL MAX_CAPACITY=$MAX_CAPACITY submit_job
    done
done