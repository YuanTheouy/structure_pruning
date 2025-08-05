#!/bin/bash
# =================================================================================
#    AMC-LLM 剪枝搜索脚本 - OPT-1.3B PPO训练版本 (增强版)
# =================================================================================
#
#   用法:
#       1. 基础用法: ./scripts/searchPPO13.sh
#       2. 指定GPU: ./scripts/searchPPO13.sh --gpu-id 0
#       3. 指定实验配置: ./scripts/searchPPO13.sh --gpu-id 1 --exp-id 2
#       4. 指定保留比例: ./scripts/searchPPO13.sh --preserve-ratio 0.8
#       5. 指定训练轮数: ./scripts/searchPPO13.sh --train-episodes 5000
#
#   脚本将在前台运行，您可以直接看到训练输出。
#
# =================================================================================

# --- 1. 参数解析 ---
GPU_ID=0           # 默认使用GPU 0
EXP_ID=0           # 默认实验配置ID
PRESERVE_RATIO=0.7 # 默认保留比例
TRAIN_EPISODES=3000 # 默认训练轮数
ENABLE_DOWNSTREAM=false # 默认关闭下游任务评估
STATE_MODE=0       # 默认使用特征提取的状态 (0=全局剪枝率, 1=特征提取状态)
SHOW_HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --exp-id)
            EXP_ID="$2"
            shift 2
            ;;
        --preserve-ratio)
            PRESERVE_RATIO="$2"
            shift 2
            ;;
        --train-episodes)
            TRAIN_EPISODES="$2"
            shift 2
            ;;
        --enable-downstream)
            ENABLE_DOWNSTREAM=true
            shift
            ;;
        --disable-downstream)
            ENABLE_DOWNSTREAM=false
            shift
            ;;
        --state-mode)
            STATE_MODE="$2"
            shift 2
            ;;
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看用法"
            exit 1
            ;;
    esac
done

if [ "$SHOW_HELP" = true ]; then
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --gpu-id ID         指定使用的GPU ID (默认: 0)"
    echo "  --exp-id ID         指定实验配置ID (默认: 0)"
    echo "  --preserve-ratio R  指定保留比例 (默认: 0.7)"
    echo "  --train-episodes N  指定训练轮数 (默认: 3000)"
    echo "  --enable-downstream 开启下游任务评估 (默认: 开启)"
    echo "  --disable-downstream 关闭下游任务评估"
    echo "  --state-mode MODE   指定Agent状态模式 (0=全局剪枝率, 1=特征提取状态, 默认: 1)"
    echo "  --help, -h          显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                              # 使用默认设置(特征提取状态)"
    echo "  $0 --gpu-id 1                   # 使用GPU 1"
    echo "  $0 --state-mode 0               # 使用全局剪枝率作为状态"
    echo "  $0 --state-mode 1               # 使用特征提取状态(默认)"
    echo "  $0 --disable-downstream         # 关闭下游任务评估"
    echo "  $0 --exp-id 2 --preserve-ratio 0.8 # 实验配置2，保留比例80%"
    echo "  $0 --train-episodes 5000 --state-mode 0 # 训练5000轮，使用全局剪枝率状态"
    exit 0
fi

# --- 2. 实验配置池 ---
#                   ID:     0      1      2      3      4
learning_rates=(          1e-4   5e-4   1e-4   5e-4)
entropy_coeffs=(          0.01   0.01   0.05   0.05)
learning_epochs=(         10     10     15     15)
clip_params=(             0.2    0.1    0.2    0.1)
seeds=(                   2025   2026   2027   2028)
num_collects=(            15     15     20     20)

# --- 3. 固定参数配置 ---
MODEL_PATH="/home/theo/data/yx_repository/01_Models/opt-1.3b"
MODEL_NAME="opt-1.3b"
PRUNE_TYPE="para"
LBOUND=0.2
RBOUND=1.0
N_SAMPLES=64

# --- 4. 根据实验ID选择超参数 ---
LEARNING_RATE=${learning_rates[EXP_ID]}
ENTROPY_COEF=${entropy_coeffs[EXP_ID]}
LEARNING_EPOCHS=${learning_epochs[EXP_ID]}
CLIP_PARAM=${clip_params[EXP_ID]}
SEED=${seeds[EXP_ID]}
NUM_COLLECT=${num_collects[EXP_ID]}

# 根据状态模式设置USE_NEW_INPUT标志
# 注意：现在不需要传递--use_new_input参数，因为它由--state_mode控制
if [ "$STATE_MODE" = "1" ]; then
    STATE_MODE_FLAG="--state_mode=1"
else
    STATE_MODE_FLAG="--state_mode=0"
fi

# 验证实验ID有效性
if [ -z "${LEARNING_RATE}" ]; then
    echo "  错误: 实验ID '${EXP_ID}' 无效。请确保ID在 0-$((${#learning_rates[@]}-1)) 范围内。"
    exit 1
fi

# 验证状态模式有效性
if [ "$STATE_MODE" != "0" ] && [ "$STATE_MODE" != "1" ]; then
    echo "  错误: 状态模式 '${STATE_MODE}' 无效。请使用 0(全局剪枝率) 或 1(特征提取状态)。"
    exit 1
fi

# --- 5. 路径与文件命名 ---
BASE_OUTPUT_DIR="./logs/ppo_experiments"
mkdir -p ${BASE_OUTPUT_DIR}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUFFIX="exp${EXP_ID}_gpu${GPU_ID}_ratio${PRESERVE_RATIO}_lr${LEARNING_RATE}_epoch${LEARNING_EPOCHS}_clip${CLIP_PARAM}_entropy${ENTROPY_COEF}_seed${SEED}_${TIMESTAMP}"
EXPORT_PATH="./checkpoints/opt13b_ppo_${SUFFIX}.pth.tar"

# --- 6. 环境变量设置 ---
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# export HF_HOME="/path/to/your/huggingface_cache" 

# --- 7. 显示实验配置信息 ---
echo "=================================================================="
echo "   启动 AMC-LLM 剪枝搜索 - OPT-1.3B 实验 #${EXP_ID}"
echo "=================================================================="
echo "    实验配置:"
echo "     - GPU设备:          ${GPU_ID}"
echo "     - 模型路径:         ${MODEL_PATH}"
echo "     - 保留比例:         ${PRESERVE_RATIO}"
echo "     - 剪枝类型:         ${PRUNE_TYPE}"
echo "     - 训练轮数:         ${TRAIN_EPISODES}"
echo "     - 下游任务评估:     $([ "$ENABLE_DOWNSTREAM" = true ] && echo "开启" || echo "关闭")"
echo "     - 状态模式:         $([ "$STATE_MODE" = "0" ] && echo "全局剪枝率" || echo "特征提取状态")"
echo ""
echo "     PPO超参数:"
echo "     - 学习率:           ${LEARNING_RATE}"
echo "     - 熵系数:           ${ENTROPY_COEF}"
echo "     - 学习周期:         ${LEARNING_EPOCHS}"
echo "     - Clip参数:         ${CLIP_PARAM}"
echo "     - 样本收集数:       ${NUM_COLLECT}"
echo "     - 随机种子:         ${SEED}"
echo ""
echo "    输出路径:"
echo "     - 日志目录:         ${BASE_OUTPUT_DIR}"
echo "     - 最优模型:         ${EXPORT_PATH}"
echo "------------------------------------------------------------------"
echo "  开始训练... (按 Ctrl+C 可中断)"
echo ""

# --- 8. 执行训练命令 ---
python -u amc_searchPPO.py \
    --job=train \
    --model="${MODEL_PATH}" \
    --model_name="${MODEL_NAME}" \
    --preserve_ratio=${PRESERVE_RATIO} \
    --prune="${PRUNE_TYPE}" \
    --lbound=${LBOUND} \
    --rbound=${RBOUND} \
    --n_samples=${N_SAMPLES} \
    --num_collect=${NUM_COLLECT} \
    --learning_epoch=${LEARNING_EPOCHS} \
    --use_real_val \
    --acc_metric=acc1 \
    --reward=reward_ppl \
    --train_episode=${TRAIN_EPISODES} \
    --seed=${SEED} \
    --structure \
    --export_path="${EXPORT_PATH}" \
    --lr_a=${LEARNING_RATE} \
    --clip_param=${CLIP_PARAM} \
    --entropy_coef=${ENTROPY_COEF} \
    --gpu_id=${GPU_ID} \
    --output="${BASE_OUTPUT_DIR}" \
    --enable_downstream="${ENABLE_DOWNSTREAM}" \
    ${STATE_MODE_FLAG}

# --- 9. 训练完成提示 ---
TRAINING_EXIT_CODE=$?
echo ""
echo "=================================================================="
if [ ${TRAINING_EXIT_CODE} -eq 0 ]; then
    echo "实验 #${EXP_ID} 训练完成！"
    echo "最优模型已保存至: ${EXPORT_PATH}"
    echo "日志文件位于: ${BASE_OUTPUT_DIR}"
else
    echo "实验 #${EXP_ID} 训练失败 (退出码: ${TRAINING_EXIT_CODE})"
fi
echo "=================================================================="
    