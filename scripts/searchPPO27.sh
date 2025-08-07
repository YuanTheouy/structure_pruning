#!/bin/bash
# =================================================================================
#    AMC-LLM 剪枝搜索脚本 - OPT-1.3B PPO训练版本 (增强版)
# =================================================================================
#
#   用法:
#       1. 基础用法: ./scripts/searchPPO27.sh
#       2. 指定GPU: ./scripts/searchPPO27.sh --gpu-id 0
#       3. 指定实验配置: ./scripts/searchPPO27.sh --gpu-id 1 --exp-id 2
#       4. 指定保留比例: ./scripts/searchPPO27.sh --preserve-ratio 0.8
#       5. 指定训练轮数: ./scripts/searchPPO27.sh --train-episodes 5000
#
#   脚本将在前台运行，您可以直接看到训练输出。
#
# =================================================================================

# --- 1. 参数解析 ---
GPU_ID=0           # 默认使用GPU 0
EXP_ID=0           # 默认实验配置ID
PRESERVE_RATIO=0.7 # 默认保留比例
TRAIN_EPISODES=5000 # 默认训练轮数
ENABLE_DOWNSTREAM=false # 默认关闭下游任务评估
STATE_MODE=1       # 默认使用特征提取的状态 (0=全局剪枝率, 1=特征提取状态)
FEATURE_CONFIG="default" # 默认使用默认特征配置 (default/basic/attention/comprehensive/minimal/activation)
USE_GUMBEL_SOFTMAX=false # 默认关闭Gumbel-Softmax功能
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
        --feature-config)
            FEATURE_CONFIG="$2"
            shift 2
            ;;
        --use-gumbel-softmax)
            USE_GUMBEL_SOFTMAX=true
            shift
            ;;
        --disable-gumbel-softmax)
            USE_GUMBEL_SOFTMAX=false
            shift
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
    echo "  --feature-config CFG 指定特征配置 (default/basic/attention/comprehensive/minimal/activation, 默认: default)"
    echo "  --use-gumbel-softmax 启用Gumbel-Softmax离散动作空间方法"
    echo "  --disable-gumbel-softmax 禁用Gumbel-Softmax，使用传统连续动作空间方法 (默认)"
    echo "  --help, -h          显示此帮助信息"
    echo ""
    echo "特征配置说明:"
    echo "  default      - 启用所有特征模块 (9维)"
    echo "  basic        - 基础特征 (4维: 层索引+模块类型+激活范数+稀疏度)"
    echo "  attention    - 注意力导向 (5维: 基础+注意力特征)"
    echo "  comprehensive- 全面特征 (同default)"
    echo "  minimal      - 最小特征 (2维: 仅层索引+模块类型)"
    echo "  activation   - 激活导向 (6维: 基础+激活特征+门控特征)"
    echo ""
    echo "Gumbel-Softmax功能说明:"
    echo "  Gumbel-Softmax是一种将连续动作空间离散化的技术，通过可微分的软采样实现稳定的策略学习。"
    echo "  启用后将使用4个离散动作bins，温度从5.0退火到0.1，提高训练稳定性。"
    echo ""
    echo "示例:"
    echo "  $0                              # 使用默认设置"
    echo "  $0 --gpu-id 1                   # 使用GPU 1"
    echo "  $0 --state-mode 0               # 使用全局剪枝率作为状态"
    echo "  $0 --state-mode 1 --feature-config basic # 使用基础特征配置"
    echo "  $0 --feature-config attention   # 使用注意力导向特征"
    echo "  $0 --disable-downstream         # 关闭下游任务评估"
    echo "  $0 --exp-id 2 --preserve-ratio 0.8 # 实验配置2，保留比例80%"
    echo "  $0 --train-episodes 5000 --feature-config comprehensive # 训练5000轮，使用全面特征"
    echo "  $0 --use-gumbel-softmax         # 启用Gumbel-Softmax离散动作空间"
    echo "  $0 --use-gumbel-softmax --state-mode 0 # 启用Gumbel-Softmax + 全局剪枝率状态"
    exit 0
fi

# --- 2. 实验配置池 ---
#                   ID:     0      1      2      3      4
learning_rates=(          5e-4   5e-4   5e-4   5e-4)
entropy_coeffs=(          0.01   0.01   0.05   0.05)
learning_epochs=(         10     10     15     15)
clip_params=(             0.2    0.1    0.2    0.1)
seeds=(                   2025   2026   2027   2028)
num_collects=(            15     15     20     20)

# --- 3. 固定参数配置 ---
MODEL_PATH="/home/theo/data/yx_repository/01_Models/opt-2.7b"
MODEL_NAME="opt-2.7b"
PRUNE_TYPE="para"
LBOUND=0.2
RBOUND=1.0
N_SAMPLES=64

# Gumbel-Softmax 相关参数
NUM_ACTION_BINS=4          # 动作离散化的bins数量
GUMBEL_TAU_INITIAL=2.0     # Gumbel-Softmax初始温度
GUMBEL_TAU_FINAL=0.2       # Gumbel-Softmax最终温度
GUMBEL_ANNEAL_EPISODES=1000  # 温度退火的episode数量

# --- 4. 根据实验ID选择超参数 ---
LEARNING_RATE=${learning_rates[EXP_ID]}
ENTROPY_COEF=${entropy_coeffs[EXP_ID]}
LEARNING_EPOCHS=${learning_epochs[EXP_ID]}
CLIP_PARAM=${clip_params[EXP_ID]}
SEED=${seeds[EXP_ID]}
NUM_COLLECT=${num_collects[EXP_ID]}

# 根据状态模式设置STATE_MODE_FLAG标志
if [ "$STATE_MODE" = "1" ]; then
    STATE_MODE_FLAG="--state_mode=1"
    # 添加特征配置参数
    STATE_MODE_FLAG="$STATE_MODE_FLAG --feature_config=$FEATURE_CONFIG"
else
    STATE_MODE_FLAG="--state_mode=0"
fi

# 根据Gumbel-Softmax设置添加相关参数
if [ "$USE_GUMBEL_SOFTMAX" = true ]; then
    GUMBEL_FLAGS="--use_gumbel_softmax --num_action_bins=$NUM_ACTION_BINS --gumbel_tau_initial=$GUMBEL_TAU_INITIAL --gumbel_tau_final=$GUMBEL_TAU_FINAL --gumbel_anneal_episodes=$GUMBEL_ANNEAL_EPISODES"
else
    GUMBEL_FLAGS=""
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

# 验证特征配置有效性(仅在状态模式1时)
if [ "$STATE_MODE" = "1" ]; then
    VALID_CONFIGS=("default" "basic" "attention" "comprehensive" "minimal" "activation")
    VALID_CONFIG=false
    for config in "${VALID_CONFIGS[@]}"; do
        if [ "$FEATURE_CONFIG" = "$config" ]; then
            VALID_CONFIG=true
            break
        fi
    done
    
    if [ "$VALID_CONFIG" = false ]; then
        echo "  错误: 特征配置 '${FEATURE_CONFIG}' 无效。"
        echo "  可用配置: ${VALID_CONFIGS[*]}"
        exit 1
    fi
fi

# --- 5. 路径与文件命名 ---
BASE_OUTPUT_DIR="./logs/ppo_experiments"
mkdir -p ${BASE_OUTPUT_DIR}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUFFIX="exp${EXP_ID}_gpu${GPU_ID}_ratio${PRESERVE_RATIO}_lr${LEARNING_RATE}_epoch${LEARNING_EPOCHS}_clip${CLIP_PARAM}_entropy${ENTROPY_COEF}_seed${SEED}_${TIMESTAMP}"
EXPORT_PATH="./checkpoints/opt27b_ppo_${SUFFIX}.pth.tar"

# --- 6. 环境变量设置 ---
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# export HF_HOME="/path/to/your/huggingface_cache" 

# 强制GPU绑定 - 确保每个训练进程使用不同GPU
echo "=> [GPU绑定] 为此进程强制绑定GPU ${GPU_ID}"
export CUDA_VISIBLE_DEVICES=${GPU_ID}
echo "=> [GPU绑定] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" 

# --- 7. 显示实验配置信息 ---
echo "=================================================================="
echo "   启动 AMC-LLM 剪枝搜索 - OPT-2.7B 实验 #${EXP_ID}"
echo "=================================================================="
echo "    实验配置:"
echo "     - GPU设备:          ${GPU_ID}"
echo "     - 模型路径:         ${MODEL_PATH}"
echo "     - 保留比例:         ${PRESERVE_RATIO}"
echo "     - 剪枝类型:         ${PRUNE_TYPE}"
echo "     - 训练轮数:         ${TRAIN_EPISODES}"
echo "     - 下游任务评估:     $([ "$ENABLE_DOWNSTREAM" = true ] && echo "开启" || echo "关闭")"
echo "     - 状态模式:         $([ "$STATE_MODE" = "0" ] && echo "全局剪枝率" || echo "特征提取状态")"
if [ "$STATE_MODE" = "1" ]; then
echo "     - 特征配置:         ${FEATURE_CONFIG}"
fi
echo "     - Gumbel-Softmax:   $([ "$USE_GUMBEL_SOFTMAX" = true ] && echo "启用" || echo "禁用")"
if [ "$USE_GUMBEL_SOFTMAX" = true ]; then
echo "       * 动作Bins数:     ${NUM_ACTION_BINS}"
echo "       * 初始温度:       ${GUMBEL_TAU_INITIAL}"
echo "       * 最终温度:       ${GUMBEL_TAU_FINAL}"
echo "       * 退火Episodes:   ${GUMBEL_ANNEAL_EPISODES}"
fi
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
    ${STATE_MODE_FLAG} \
    ${GUMBEL_FLAGS}

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
    