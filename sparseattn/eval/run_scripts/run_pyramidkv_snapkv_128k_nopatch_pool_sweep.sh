#!/bin/bash -l

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_NAME="pyramidkv_snapkv_128k_nopatch_pool_sweep"
DEFAULT_NUM_GPUS=1
DEFAULT_TIME="15:00:00"
DEFAULT_MEM="50G"
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
OUT_DIR=outputs/${MODEL_NAME}/NOPATCH_POOL${POOL_SIZE}_outputs${SUFFIX}__${METHOD}
if [ -f ${OUT_DIR}/.${TASK_NAME}.completed ]; then
    echo "Skipping completed: ${MODEL_NAME} ${METHOD} - ${TASK_NAME} - sp${SPARSITY} - pool${POOL_SIZE}"
    return
fi

# Resource allocation - special handling for RAG task
NUM_GPUS=${NUM_GPUS:-$DEFAULT_NUM_GPUS}
TIME=${TIME:-$DEFAULT_TIME}
MEM=${MEM:-$DEFAULT_MEM}

# RAG task needs more resources
if [ "$TASK_NAME" == "rag" ]; then
    NUM_GPUS=2
fi

# Build command and job name
CMD="python eval.py --config ${TASK} --model_name_or_path $MODEL --tokenizer_name $MODEL --minference $METHOD --output_dir $OUT_DIR $EXTRA"
JOB_NAME="${SCRIPT_NAME}_${TASK_NAME}_${METHOD}_sp${SPARSITY}_pool${POOL_SIZE}_${MODEL_NAME}"

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
TOTAL_CAPACITY=131072
MINFERENCE_WINDOW_SIZE=64
SPARSITY=70

for TASK in ${COMMON_TASKS[@]}; do 
    for METHOD in pyramidkv snapkv; do
        for POOL_SIZE in 1024 2048 4096 8192 16384 32768; do
            # Convert sparsity to fraction
            SPARSITY_FRAC=$(echo "$SPARSITY / 100.0" | bc -l)
            
            # Build extra arguments
            EXTRA="--no_torch_compile --minference_sparsity $SPARSITY_FRAC --minference_window_size $MINFERENCE_WINDOW_SIZE --minference_chunk_prefilling $TOTAL_CAPACITY --minference_max_capacity_prompt $POOL_SIZE"
            SUFFIX="_sp${SPARSITY}_tg"
        
            TASK=$TASK MODEL=$MODEL SUFFIX=$SUFFIX EXTRA=$EXTRA METHOD=$METHOD SPARSITY=$SPARSITY POOL_SIZE=$POOL_SIZE submit_job
        done
    done
done

    
