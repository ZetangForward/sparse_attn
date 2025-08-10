#!/bin/bash -l
set m
set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

export OUTLINES_CACHE_DIR="/data/qqt/project/PruLong-main/tmp/outlines"
export HF_ENDPOINT=https://hf-mirror.com

# 项目路径
PROJECT_DIR="/data/qqt/project/SparseAttn/sparseattn/eval"
cd "$PROJECT_DIR" || { echo "❌ Project directory not found!"; exit 1; }

# 模型配置
MODEL="/data/hf_models/Meta-Llama-3.1-8B-Instruct"
SPARSITY=0
PREFILL=32768

# 所有任务列表
TASKS=(
    "longproc_addon/configs/html_to_tsv.yaml"
    "longproc_addon/configs/travel_planning.yaml"
    "configs/recall.yaml"
    "configs/rerank.yaml"
    "configs/rag.yaml"
    "configs/icl.yaml"
    "configs/longqa.yaml"
    "configs/summ.yaml"
)

OUTPUT_LOGS_DIR="joblog-xattn"

METRIC="xattn"

# 输出和日志目录
mkdir -p outputs "$OUTPUT_LOGS_DIR"

# ============================================================================
# 全局变量：记录启动的 worker PIDs
# ============================================================================
WORKER_PIDS=()

# ============================================================================
# 清理函数：清理指定 GPU 上的 Python 进程
# ============================================================================
cleanup_gpu() {
    local gpu_id=$1
    # echo "🧹 Cleaning up processes on GPU $gpu_id..."
    local pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i $gpu_id 2>/dev/null | xargs 2>/dev/null) || true
    if [ -n "$pids" ] && [ "$pids" != " " ]; then
        for pid in $pids; do
            # 尽量判断是否为 python 进程
            if ps -p $pid --no-headers -o comm= 2>/dev/null | grep -q "python"; then
                echo "💀 Killing Python process on GPU $gpu_id: PID $pid"
                kill -9 $pid 2>/dev/null || true
            fi
        done
    fi
}

# ============================================================================
# 信号处理：捕获 Ctrl+C，终止所有 worker 和 GPU 进程
# ============================================================================
trap '{
    echo -e "\n🛑 Received interrupt signal (Ctrl+C). Shutting down gracefully..."
    # 先尝试正常 kill 所有 worker 进程
    kill "${WORKER_PIDS[@]}" 2>/dev/null || true
    sleep 2
    # 再强制 kill 所有 worker（防止 hang）
    kill -9 "${WORKER_PIDS[@]}" 2>/dev/null || true
    # 清理所有使用的 GPU
    for gpu in "${GPUS[@]}"; do
        cleanup_gpu $gpu
    done
    echo "✅ Cleanup completed. Exiting."
    exit 1
}' SIGINT SIGTERM

# ============================================================================
# Worker 函数：每个 GPU 执行任务
# ============================================================================
worker() {
    local gpu_id=$1
    shift
    local tasks=("$@")

    # ✅ 为每个 worker 设置独立的 trap，确保能响应中断
    local worker_pid=$$
    trap '{
        echo "🛑 [GPU $gpu_id] Worker interrupted. Cleaning up..."
        cleanup_gpu $gpu_id
        exit 1
    }' SIGINT SIGTERM

    for TASK_PATH in "${tasks[@]}"; do
        if [ ! -f "$TASK_PATH" ]; then
            echo "❌ [GPU $gpu_id] Task config not found: $TASK_PATH"
            continue
        fi

        TASK_NAME=$(basename "$TASK_PATH" .yaml)
        MODEL_NAME=$(basename "$MODEL")
        OUT_DIR="outputs/${MODEL_NAME}_${METRIC}/outputs_sp${SPARSITY}_pf${PREFILL}_tg"
        COMPLETED_FLAG="$OUT_DIR/.${TASK_NAME}.completed"
        LOGFILE="./${OUTPUT_LOGS_DIR}/${TASK_NAME}_gpu${gpu_id}.log"

        mkdir -p "$OUT_DIR"

        if [ -f "$COMPLETED_FLAG" ]; then
            echo "🟩 [GPU $gpu_id] Skipping completed: $TASK_NAME"
            continue
        fi

        MASKS="$MODEL/masks_sp${SPARSITY}.tsv"
        EXTRA="--no_torch_compile --fastprefill_metric $METRIC"
        CMD="python eval.py --config $TASK_PATH --model_name_or_path $MODEL --tokenizer_name $MODEL --output_dir $OUT_DIR $EXTRA"

        echo "🚀 [GPU $gpu_id] Running: $TASK_NAME"
        echo "   Model: $MODEL_NAME"
        echo "   Command: $CMD"
        echo "   Log: $LOGFILE"
        echo "   Output: $OUT_DIR"

        # 执行命令
        CUDA_VISIBLE_DEVICES=$gpu_id \
            $CMD >> "$LOGFILE" 2>&1

        if [ $? -eq 0 ]; then
            echo "${MODEL_NAME} - ${TASK_NAME}" > "$COMPLETED_FLAG"
            echo "✅ [GPU $gpu_id] Success: $TASK_NAME"
        else
            echo "❌ [GPU $gpu_id] Failed: $TASK_NAME (check log: $LOGFILE)"
        fi

        sleep 2
    done

    # worker 结束时清理 GPU
    cleanup_gpu $gpu_id
    echo "🏁 [GPU $gpu_id] Worker finished."
}

# ============================================================================
# 任务分配并启动
# ============================================================================

echo "📋 Total tasks: ${#TASKS[@]}"
echo "🚀 Using GPUs: ${GPUS[*]}"
echo "🧠 Distributing tasks in round-robin across $NUM_GPUS workers..."
echo "📝 Logs will be saved to ./joblog/<task>_gpu<id>.log"

for idx in "${!GPUS[@]}"; do
    gpu=${GPUS[idx]}
    worker_tasks=()

    for ((i = idx; i < ${#TASKS[@]}; i += NUM_GPUS)); do
        worker_tasks+=("${TASKS[i]}")
    done

    if [ ${#worker_tasks[@]} -eq 0 ]; then
        continue
    fi

    echo "📎 GPU $gpu assigned ${#worker_tasks[@]} tasks: $(printf '%s ' "${worker_tasks[@]##*/}")"

    # 启动 worker 并记录 PID
    worker "$gpu" "${worker_tasks[@]}" &
    WORKER_PIDS+=($!)
done

# ============================================================================
# 安全等待所有 worker（非阻塞轮询）
# ============================================================================
echo "⏳ Waiting for all workers to complete... (Press Ctrl+C to interrupt)"

while [ ${#WORKER_PIDS[@]} -gt 0 ]; do
    NEW_PIDS=()
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            NEW_PIDS+=("$pid")  # 仍在运行
        else
            wait "$pid" 2>/dev/null || true
            echo "🔽 Worker PID $pid finished."
        fi
    done
    WORKER_PIDS=("${NEW_PIDS[@]}")
    sleep 3
done

echo "🎉 All tasks completed. Script finished."