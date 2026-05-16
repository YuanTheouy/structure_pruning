#!/bin/bash
# =================================================================================
#    AMC-LLM 模型导出脚本 - Qwen2.5-7B 剪枝模型导出 (增强版)
# =================================================================================
#
#   用法:
#       该脚本用于导出经过剪枝的 Qwen2.5-7B 模型。
#
#   核心功能:
#       1. 支持通过 --ratios 参数传入自定义剪枝比例。
#       2. 支持在脚本内预定义多套剪枝比例，通过 --preserve-ratio 方便地切换，
#          非常适合开发和调试。
#
#   示例:
#       # 使用脚本内为 0.7 预设的剪枝比例
#       ./exportQwen25.sh --preserve-ratio 0.7 --enable-recon
#
#       # 临时覆盖，使用命令行传入的剪枝比例
#       ./exportQwen25.sh --ratios "0.9,0.8,..." --enable-recon
#
#       # 指定使用GPU 0和1进行重建
#       ./exportQwen25.sh --gpu-id 0,1 --enable-recon
#
#       # 组合使用：指定GPU、启用重构和下游评估
#       ./exportQwen25.sh --gpu-id 1,2,3 --enable-recon --enable-downstream
#
# =================================================================================

# --- 1. 参数解析 ---
PRESERVE_RATIO=0.7 # 默认保留比例，用于查找预设或命名
ENABLE_RECON=false
ENABLE_DOWNSTREAM=false
DATASET_NAME="wikitext2"
EXPORT_PATH=""
RATIOS="" # 优先使用命令行传入的比例
GPU_ID=""  # 新增：GPU ID参数
SHOW_HELP=false
USER_PROVIDED_RATIOS=false # 标记是否由用户在命令行指定

while [[ $# -gt 0 ]]; do
    case $1 in
        --preserve-ratio)
            PRESERVE_RATIO="$2"
            shift 2
            ;;
        --enable-recon)
            ENABLE_RECON=true
            shift
            ;;
        --enable-downstream)
            ENABLE_DOWNSTREAM=true
            shift
            ;;
        --dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        --export-path)
            EXPORT_PATH="$2"
            shift 2
            ;;
        --ratios)
            RATIOS="$2"
            USER_PROVIDED_RATIOS=true
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo "错误: 未知参数: $1"
            echo "使用 --help 查看用法"
            exit 1
            ;;
    esac
done

if [ "$SHOW_HELP" = true ]; then
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --preserve-ratio R    指定保留比例 (默认: 0.7)，用于查找预设或命名"
    echo "  --ratios RATIOS       (可选) 指定自定义剪枝比例，优先级高于内部预设"
    echo "  --enable-recon        启用重构模式 (默认: 禁用)"
    echo "  --enable-downstream   启用下游任务评估 (默认: 禁用)"
    echo "  --dataset NAME        指定用于PPL评估的数据集 (默认: wikitext2)"
    echo "  --export-path PATH    指定导出路径 (默认: 自动生成)"
    echo "  --gpu-id IDS          指定使用的GPU ID (如: 0 或 0,1,2,3，默认: 自动选择)"
    echo "  --help, -h            显示此帮助信息"
    exit 0
fi

# --- 2. 预定义剪枝比例配置 (调试与开发用) ---
# !!! 警告: 以下剪枝比例是为 Qwen2.5-7B 模型结构设计的占位符 !!!
# !!! 它们的数值是随机生成的，仅用于演示和方便调试，*绝不能*用于实际的模型剪枝 !!!
# !!! 在实际使用前，您必须用通过搜索得到的、针对您的任务优化的真实比例替换它们 !!!
#
# Qwen2.5-7B Instruct 有 40 个 Transformer 层。
# 假设剪枝策略针对每层的 gate_proj 和 down_proj (FFN部分)，总计 40*2=80 个可剪枝参数。
# 因此，下面的字符串包含 80 个逗号分隔的数值。
declare -A ratio_configs
ratio_configs[0.7]="1.0, 0.504222972972973, 1.0, 0.2282516891891892, 1.0, 1.0, 1.0, 0.33292863175675674, 1.0, 0.5825063344594594, 1.0, 0.5981313344594594, 1.0, 0.37811444256756754, 1.0, 1.0, 1.0, 1.0, 1.0, 0.32849451013513514, 1.0, 0.44256756756756754, 0.4642857142857143, 0.6009818412162162, 1.0, 1.0, 1.0, 1.0, 0.9285714285714286, 1.0, 1.0, 0.6026182432432432, 1.0, 0.4096283783783784, 1.0, 0.7765519425675675, 0.5357142857142857, 0.23965371621621623, 1.0, 0.4552364864864865, 0.9285714285714286, 0.8490815033783784, 1.0, 0.5503061655405406, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5763830236486487, 1.0, 0.36776815878378377, 1.0, 0.7350612331081081, 1.0, 0.9953019425675675"


# --- 3. 固定参数配置 ---
# !!! 请根据您的环境修改以下路径 !!!
# MODEL_PATH="/home/theo/data/yx_repository/01_Models/Qwen2.5-7B" # 修改为您的 Qwen2.5-7B 模型路径
MODEL_PATH="/home/yx/yx_repository/01_Models/Qwen2.5-7B" # 修改为您的 Qwen2.5-7B 模型路径
MODEL_NAME="qwen2.5-7b" # 模型标识符
PRUNE_TYPE="para"
LBOUND=0.1
RBOUND=1.0
N_SAMPLES=64
RECON_SAMPLE=32
SEED=2025

# --- 4. 自动选择剪枝比例配置 ---
if [ "$USER_PROVIDED_RATIOS" = false ]; then
    RATIOS=${ratio_configs[$PRESERVE_RATIO]}
    if [ -z "$RATIOS" ]; then
        echo "错误: 在脚本中没有为保留比例 ${PRESERVE_RATIO} 定义预设剪枝配置。"
        echo "请在 'ratio_configs' 中添加该配置，或使用 --ratios 参数直接提供。"
        exit 1
    fi
fi

# --- 5. 自动生成导出路径 ---
if [ -z "$EXPORT_PATH" ]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    RATIO_SUFFIX=$(echo "$PRESERVE_RATIO" | tr '.' '_')
    RECON_SUFFIX=""
    if [ "$ENABLE_RECON" = true ]; then
        RECON_SUFFIX="_recon"
    fi
    EXPORT_PATH="./checkpoints/${MODEL_NAME}_${RATIO_SUFFIX}_${DATASET_NAME}${RECON_SUFFIX}_${TIMESTAMP}_export.pth.tar"
fi

# --- 6. 环境与执行器配置 ---
CONDA_ENV_NAME="amc_llm" # 修改为您的Conda环境名称
PYTHON_EXECUTABLE=$(conda run -n ${CONDA_ENV_NAME} which python 2>/dev/null || which python)
if ! command -v $PYTHON_EXECUTABLE &> /dev/null; then
    echo "错误: 找不到 Python 解释器。请激活 Conda 环境或检查路径。"
    exit 1
fi
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "${CONDA_PREFIX}/lib" | tr '\n' ':' | sed 's/:$//')

# --- 6.1. GPU设置 ---
if [ -n "$GPU_ID" ]; then
    echo "=> [GPU配置] 用户指定GPU: ${GPU_ID}"
    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    GPU_COUNT=$(echo ${GPU_ID} | tr ',' '\n' | wc -l)
    MAIN_GPU_ID=$(echo ${GPU_ID} | cut -d',' -f1)
    echo "=> [GPU配置] 将使用 ${GPU_COUNT} 个GPU: ${GPU_ID}"
    echo "=> [GPU配置] 主GPU: ${MAIN_GPU_ID}"
else
    echo "=> [GPU配置] 使用自动GPU分配"
    GPU_ID="auto"
    MAIN_GPU_ID="auto"
fi

# --- 6.2. 内存优化配置 ---
# 启用PyTorch内存分段以减少内存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 设置CUDA内存管理策略
export CUDA_LAUNCH_BLOCKING=0
# 强制垃圾回收和内存清理
export PYTHONHASHSEED=0

# --- 7. 重构与下游任务参数配置 ---
RECON_FLAG=""
RECON_STATUS="禁用 (速度更快，精度稍低)"
if [ "$ENABLE_RECON" = true ]; then
    RECON_FLAG="--recon"
    RECON_STATUS="启用 (精度更高，速度较慢)"
fi

DOWNSTREAM_FLAG="--enable_downstream=false"
DOWNSTREAM_STATUS="禁用 (仅评估PPL，速度较快)"
DELAYED_EVAL_FLAG=""
if [ "$ENABLE_DOWNSTREAM" = true ]; then
    DOWNSTREAM_FLAG="--enable_downstream=true"
    DOWNSTREAM_STATUS="启用 (评估下游任务性能，耗时较长)"
    DELAYED_EVAL_FLAG="--delayed_downstream_eval"
fi

# --- 8. 确保导出目录存在 ---
mkdir -p "$(dirname "$EXPORT_PATH")"

# --- 9. 显示配置信息 ---
echo "=================================================================="
echo "   AMC-LLM 模型导出 - Qwen2.5-7B 剪枝模型"
echo "=================================================================="
echo "    导出配置:"
echo "     - 模型路径:         ${MODEL_PATH}"
echo "     - 数据集 (PPL):     ${DATASET_NAME}"
echo "     - 保留比例 (命名):  ${PRESERVE_RATIO}"
echo "     - 重构模式:         ${RECON_STATUS}"
echo "     - 下游任务评估:     ${DOWNSTREAM_STATUS}"
echo ""
echo "    GPU配置:"
if [ "$GPU_ID" = "auto" ]; then
echo "     - GPU使用:          自动分配"
else
echo "     - GPU使用:          ${GPU_ID} (${GPU_COUNT}个GPU)"
echo "     - 主GPU:            ${MAIN_GPU_ID}"
fi
echo ""
echo "    剪枝比例来源:"
if [ "$USER_PROVIDED_RATIOS" = true ]; then
    echo "     - 由 --ratios 命令行参数提供"
else
    echo "     - 从脚本内预设 'ratio_configs[${PRESERVE_RATIO}]' 加载"
    echo "     - (警告: 请确认预设值是否为真实有效比例，而非占位符)"
fi
echo "------------------------------------------------------------------"
echo "  准备执行导出命令... "
echo ""

# --- 10. 执行导出命令 ---
# 在开始前显示当前GPU状态
echo "=> 当前GPU状态检查:"
if command -v nvidia-smi &> /dev/null; then
    if [ "$GPU_ID" != "auto" ]; then
        echo "=> 指定GPU ${GPU_ID} 状态:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv,noheader,nounits -i ${GPU_ID} 2>/dev/null || echo "   无法查询指定GPU状态"
    else
        echo "=> 所有GPU状态:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "   无法查询GPU状态"
    fi
else
    echo "   nvidia-smi 不可用，跳过GPU状态检查"
fi
echo ""

${PYTHON_EXECUTABLE} -u amc_searchPPO.py \
    --job=export \
    --model="${MODEL_PATH}" \
    --model_name="${MODEL_NAME}" \
    --dataset_name="${DATASET_NAME}" \
    --preserve_ratio=${PRESERVE_RATIO} \
    --ratios="${RATIOS}" \
    --use_real_val \
    --prune="${PRUNE_TYPE}" \
    --structure \
    --state_mode=0 \
    ${RECON_FLAG} \
    ${DOWNSTREAM_FLAG} \
    ${DELAYED_EVAL_FLAG} \
    --lbound=${LBOUND} \
    --rbound=${RBOUND} \
    --n_samples=${N_SAMPLES} \
    --recon_sample=${RECON_SAMPLE} \
    --acc_metric=acc1 \
    --reward=reward_ppl \
    --seed=${SEED} \
    --export_path="${EXPORT_PATH}"

# --- 11. 导出完成提示 ---
EXPORT_EXIT_CODE=$?
echo ""
echo "=================================================================="
if [ ${EXPORT_EXIT_CODE} -eq 0 ]; then
    echo "模型导出完成！"
    echo "导出文件位于: ${EXPORT_PATH}"
    if [ -f "${EXPORT_PATH}" ]; then
        FILE_SIZE=$(du -h "${EXPORT_PATH}" | cut -f1)
        echo "文件大小: ${FILE_SIZE}"
    fi
else
    echo "模型导出失败 (退出码: ${EXPORT_EXIT_CODE})"
fi
echo "=================================================================="