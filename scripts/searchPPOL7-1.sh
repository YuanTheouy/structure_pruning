#!/bin/bash
# =================================================================================
#    AMC-LLM 剪枝搜索脚本 - Llama-2-7B PPO训练版本 (增强版)
# =================================================================================
#
#   用法:
#       1. 基础用法: ./scripts/searchPPOL7.sh
#       2. 指定GPU: ./scripts/searchPPOL7.sh --gpu-id "0,1"
#       3. 指定实验配置: ./scripts/searchPPOL7.sh --gpu-id "1,2" --exp-id 2
#       4. 指定目标稀疏度: ./scripts/searchPPOL7.sh --target-sparsity 0.2
#       5. 指定训练轮数: ./scripts/searchPPOL7.sh --train-episodes 8000
#       6. 兼容旧参数: ./scripts/searchPPOL7.sh --preserve-ratio 0.8  (会自动转换为稀疏度0.2)
#
# =================================================================================

# --- 1. 参数解析 ---
GPU_ID="0,1"       # 默认使用GPU 0,1 (支持多GPU)
EXP_ID=0           # 默认实验配置ID
TARGET_SPARSITY=0.2 # 默认目标稀疏度
TRAIN_EPISODES=8000 # 默认训练轮数
ENABLE_DOWNSTREAM=false # 默认关闭下游任务评估
STATE_MODE=0       # 默认使用特征提取的状态 (0=全局剪枝率, 1=特征提取状态)
FEATURE_CONFIG="default" # 默认使用默认特征配置 (default/basic/attention/comprehensive/minimal/activation)
USE_GUMBEL_SOFTMAX=false # 默认关闭Gumbel-Softmax功能
USE_GRADUAL_PRUNING=false  # 默认开启渐进式剪枝
GRADUAL_PRUNING_END_EPISODE=1000 # 渐进式剪枝结束的episode
GRADUAL_INITIAL_SPARSITY=0.05 # 渐进式剪枝的初始稀疏度

# --- 评估数据集配置 ---
USE_DATASET_GROWTH=true     # 默认关闭数据集渐进增长功能
DATASET_INITIAL_RATIO=0.05     # 数据集初始使用比例 (1.0 = 使用全部验证集)
DATASET_FINAL_RATIO=1.0       # 数据集最终使用比例 (1.0 = 使用全部验证集)
DATASET_GROWTH_START_EPISODE=0 # 数据集增长开始的episode
DATASET_GROWTH_END_EPISODE=1000 # 数据集增长结束的episode

SHOW_HELP=false

# ★★★ 添加这行 ★★★
RESUME_CHECKPOINT_PATH=""

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
            TARGET_SPARSITY=$(awk "BEGIN {printf \"%.6f\", 1 - $2}")
            echo "=> [转换] 保留比例 $2 -> 目标稀疏度 $TARGET_SPARSITY"
            shift 2
            ;;
        --target-sparsity)
            TARGET_SPARSITY="$2"
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
        --use-gradual-pruning)
            USE_GRADUAL_PRUNING=true
            shift
            ;;
        --disable-gradual-pruning)
            USE_GRADUAL_PRUNING=false
            shift
            ;;
        --gradual-pruning-end-episode)
            GRADUAL_PRUNING_END_EPISODE="$2"
            shift 2
            ;;
        --gradual-initial-sparsity)
            GRADUAL_INITIAL_SPARSITY="$2"
            shift 2
            ;;
        --use-dataset-growth)
            USE_DATASET_GROWTH=true
            shift
            ;;
        --disable-dataset-growth)
            USE_DATASET_GROWTH=false
            shift
            ;;
        --dataset-initial-ratio)
            DATASET_INITIAL_RATIO="$2"
            shift 2
            ;;
        --dataset-final-ratio)
            DATASET_FINAL_RATIO="$2"
            shift 2
            ;;
        --dataset-growth-start-episode)
            DATASET_GROWTH_START_EPISODE="$2"
            shift 2
            ;;
        --dataset-growth-end-episode)
            DATASET_GROWTH_END_EPISODE="$2"
            shift 2
            ;;
        --resume)
            RESUME_CHECKPOINT_PATH="$2"
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
    echo "  --gpu-id ID         指定使用的GPU ID (默认: 0,1，支持单个或多个GPU如 0 或 0,1,2)"
    echo "  --exp-id ID         指定实验配置ID (默认: 0)"
    echo "  --target-sparsity S 指定目标稀疏度 (默认: 0.2)"
    echo "  --preserve-ratio R  指定保留比例 (会自动转换为稀疏度)"
    echo "  --train-episodes N  指定训练轮数 (默认: 8000)"
    echo "  --enable-downstream 开启下游任务评估"
    echo "  --disable-downstream 关闭下游任务评估 (默认)"
    echo "  --state-mode MODE   指定Agent状态模式 (0=全局剪枝率, 1=特征提取状态, 默认: 1)"
    echo "  --feature-config CFG 指定特征配置 (default/basic/attention/comprehensive/minimal/activation, 默认: default)"
    echo "  --use-gumbel-softmax 启用Gumbel-Softmax离散动作空间方法"
    echo "  --disable-gumbel-softmax 禁用Gumbel-Softmax (默认)"
    echo "  --use-gradual-pruning 启用渐进式剪枝 (默认)"
    echo "  --disable-gradual-pruning 禁用渐进式剪枝"
    echo "  --gradual-pruning-end-episode N 渐进式剪枝结束的episode (默认: 2000)"
    echo "  --gradual-initial-sparsity S 渐进式剪枝的初始稀疏度 (默认: 0.05)"
    echo "  --use-dataset-growth      启用数据集渐进增长功能"
    echo "  --disable-dataset-growth  禁用数据集渐进增长功能 (默认)"
    echo "  --dataset-initial-ratio R 数据集初始使用比例 (0.0-1.0, 默认: 0.05)"
    echo "  --dataset-final-ratio R   数据集最终使用比例 (0.0-1.0, 默认: 1.0)"
    echo "  --dataset-growth-start-episode N  数据集开始增长的episode (默认: 0)"
    echo "  --dataset-growth-end-episode N    数据集增长结束的episode (默认: 1000)"
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
    echo "  启用后将使用20个离散动作bins，温度从2.0退火到0.2，提高训练稳定性。"
    echo ""
    echo "渐进式剪枝功能说明:"
    echo "  渐进式剪枝从较低的稀疏度开始，在训练过程中逐步增加到目标稀疏度。"
    echo "  这种方法可以帮助模型更好地适应剪枝过程，提高最终的性能。"
    echo ""
    echo "数据集渐进增长功能说明:"
    echo "  数据集渐进增长从较小的数据集比例开始，在训练过程中逐步增加到完整数据集。"
    echo "  这种课程学习方法可以帮助模型在训练初期快速收敛，后期使用更多数据提高性能。"
    echo ""
    echo "示例:"
    echo "  $0                              # 使用默认设置(GPU 0,1)"
    echo "  $0 --gpu-id 1                   # 使用单个GPU 1"
    echo "  $0 --gpu-id 0,1,2               # 使用多个GPU 0,1,2"
    echo "  $0 --target-sparsity 0.3 --use-gradual-pruning --gradual-initial-sparsity 0.1"
    echo "  $0 --use-dataset-growth --dataset-initial-ratio 0.1 --dataset-growth-end-episode 2000"
    echo "  $0 --use-gradual-pruning --use-dataset-growth # 同时启用两种渐进策略"
    exit 0
fi

# --- 2. 实验配置池 ---
#                   ID:     0      1      2      3
learning_rates=(          5e-4   5e-4   1e-4   5e-4)
entropy_coeffs=(          0.01   0.01   0.05   0.05)
learning_epochs=(         10     10     15     15)
clip_params=(             0.2    0.1    0.2    0.1)
seeds=(                   2025   2026   2027   2028)
num_collects=(            15     15     20     20)

# --- 3. 固定参数配置 ---
# MODEL_PATH="/home/wuyanming/yx_repository/01_Models/llama1-13b"
MODEL_PATH="/public/home/weijiateng2023/model/llama1-7b"
# MODEL_PATH="/home/lisiqi/amc-LLM/model/llama-2-7b-hf"
# MODEL_PATH="../../01_Models/llama-2-7b-hf"
MODEL_NAME="llama1-7b"
PRUNE_TYPE="para"
LBOUND=0.1
RBOUND=1.0
N_SAMPLES=64

# Gumbel-Softmax 相关参数
NUM_ACTION_BINS=20
GUMBEL_TAU_INITIAL=2.0
GUMBEL_TAU_FINAL=0.2
GUMBEL_ANNEAL_EPISODES=1000

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

# 根据渐进式剪枝设置添加相关参数
if [ "$USE_GRADUAL_PRUNING" = true ]; then
    # 直接使用目标稀疏度，不需要转换
    GRADUAL_FLAGS="--use_gradual_pruning --gradual_final_sparsity=$TARGET_SPARSITY --gradual_initial_sparsity=$GRADUAL_INITIAL_SPARSITY --gradual_pruning_end_episode=$GRADUAL_PRUNING_END_EPISODE"
else
    GRADUAL_FLAGS=""
fi

# 根据数据集渐进增长设置添加相关参数
if [ "$USE_DATASET_GROWTH" = true ]; then
    DATASET_GROWTH_FLAGS="--use_dataset_growth --dataset_initial_ratio=$DATASET_INITIAL_RATIO --dataset_final_ratio=$DATASET_FINAL_RATIO --dataset_growth_start_episode=$DATASET_GROWTH_START_EPISODE --dataset_growth_end_episode=$DATASET_GROWTH_END_EPISODE"
else
    DATASET_GROWTH_FLAGS=""
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
MAIN_GPU_ID=$(echo ${GPU_ID} | cut -d',' -f1)
SUFFIX="exp${EXP_ID}_gpu${MAIN_GPU_ID}_sparsity${TARGET_SPARSITY}_lr${LEARNING_RATE}_epoch${LEARNING_EPOCHS}_clip${CLIP_PARAM}_entropy${ENTROPY_COEF}_seed${SEED}_${TIMESTAMP}"

# ★ Bug修复：创建一个变量来保存完整、唯一的输出目录
FULL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_NAME}_${PRUNE_TYPE}_search_${SUFFIX}"
mkdir -p ${FULL_OUTPUT_DIR}

# ★ Bug修复：导出路径应该在新的输出目录中
EXPORT_PATH="${FULL_OUTPUT_DIR}/llama1-7b_ppo_best_model.pth.tar"


# --- 6. 环境变量设置 ---
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="../huggingface" 

# 强制GPU绑定 - 确保每个训练进程使用指定GPU
echo "=> [GPU绑定] 为此进程绑定GPU ${GPU_ID}"
export CUDA_VISIBLE_DEVICES=${GPU_ID}
echo "=> [GPU绑定] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# 检查GPU数量
GPU_COUNT=$(echo ${GPU_ID} | tr ',' '\n' | wc -l)
echo "=> [GPU信息] 将使用 ${GPU_COUNT} 个GPU: ${GPU_ID}"
echo "=> [GPU信息] 主GPU ID: ${MAIN_GPU_ID}" 

# --- 7. 显示实验配置信息 ---
echo "=================================================================="
echo "   启动 AMC-LLM 剪枝搜索 - llama1-7b 实验 #${EXP_ID}"
echo "=================================================================="
echo "    实验配置:"
echo "     - GPU设备:          ${GPU_ID} (${GPU_COUNT}个GPU)"
echo "     - 主GPU ID:         ${MAIN_GPU_ID}"
echo "     - 模型路径:         ${MODEL_PATH}"
echo "     - 目标稀疏度:       ${TARGET_SPARSITY} (剪枝 $(awk "BEGIN {printf \"%.1f\", $TARGET_SPARSITY * 100}")% 参数)"
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
echo "     - 数据集使用模式:   $([ "$USE_DATASET_GROWTH" = true ] && echo "渐进增长" || echo "全集使用")"
if [ "$USE_DATASET_GROWTH" = true ]; then
echo "       * 初始使用比例:   ${DATASET_INITIAL_RATIO} ($(awk "BEGIN {printf \"%.1f\", $DATASET_INITIAL_RATIO * 100}")% 验证集)"
echo "       * 最终使用比例:   ${DATASET_FINAL_RATIO} ($(awk "BEGIN {printf \"%.1f\", $DATASET_FINAL_RATIO * 100}")% 验证集)"
echo "       * 增长开始Episode: ${DATASET_GROWTH_START_EPISODE}"
echo "       * 增长结束Episode: ${DATASET_GROWTH_END_EPISODE}"
else
echo "       * 使用验证集比例: 1.0 (100% 验证集)"
fi
echo "     - 渐进式剪枝:       $([ "$USE_GRADUAL_PRUNING" = true ] && echo "启用" || echo "禁用")"
if [ "$USE_GRADUAL_PRUNING" = true ]; then
echo "       * 初始稀疏度:     ${GRADUAL_INITIAL_SPARSITY} (剪枝 $(awk "BEGIN {printf \"%.1f\", $GRADUAL_INITIAL_SPARSITY * 100}")% 参数)"
echo "       * 目标稀疏度:     ${TARGET_SPARSITY} (剪枝 $(awk "BEGIN {printf \"%.1f\", $TARGET_SPARSITY * 100}")% 参数)"
echo "       * 结束Episode:    ${GRADUAL_PRUNING_END_EPISODE}"
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
echo "     - 日志目录:         ${FULL_OUTPUT_DIR}"
echo "     - 最优模型:         ${EXPORT_PATH}"

# ★ 添加续训模式的提示信息
if [ -n "$RESUME_CHECKPOINT_PATH" ]; then
    echo "=> [模式] 断点续训：将从 '${RESUME_CHECKPOINT_PATH}' 加载状态"
    echo "=> [模式] 新的输出将保存到全新目录: ${FULL_OUTPUT_DIR}"
else
    echo "=> [模式] 全新训练，输出目录: ${FULL_OUTPUT_DIR}"
fi

echo "------------------------------------------------------------------"
echo "  开始训练... (按 Ctrl+C 可中断)"
echo ""

# --- 8. 执行训练命令 ---

# ★ 添加一个flag来传递续训参数
RESUME_FLAG=""
if [ -n "$RESUME_CHECKPOINT_PATH" ]; then
    RESUME_FLAG="--resume_from_checkpoint ${RESUME_CHECKPOINT_PATH}"
fi

python -u amc_searchPPO.py \
    --job=train \
    --model="${MODEL_PATH}" \
    --model_name="${MODEL_NAME}" \
    --preserve_ratio=$(awk "BEGIN {printf \"%.6f\", 1 - $TARGET_SPARSITY}") \
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
    --gpu_id=${MAIN_GPU_ID} \
    --output="${FULL_OUTPUT_DIR}" \
    --enable_downstream="${ENABLE_DOWNSTREAM}" \
    ${STATE_MODE_FLAG} \
    ${GUMBEL_FLAGS} \
    ${GRADUAL_FLAGS} \
    ${DATASET_GROWTH_FLAGS} \
    ${RESUME_FLAG}

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
