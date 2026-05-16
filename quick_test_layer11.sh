#!/bin/bash
# =================================================================================
#    快速测试脚本 - 验证Layer 11 Head维度匹配修复
# =================================================================================
#
#   用法:
#       ./quick_test_layer11.sh [选项]
#
#   核心功能:
#       1. 专门测试Layer 11 Head的维度匹配问题修复
#       2. 使用极少的样本数和层数，快速验证
#       3. 如果通过则说明修复有效
#
#   示例:
#       ./quick_test_layer11.sh --gpu-id 0
#       ./quick_test_layer11.sh --gpu-id 1,2 --enable-recon
#
# =================================================================================

# --- 1. 参数解析 ---
GPU_ID="0"  # 默认使用GPU 0
ENABLE_RECON=false
SHOW_HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --enable-recon)
            ENABLE_RECON=true
            shift
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
    echo "  --gpu-id IDS          指定使用的GPU ID (如: 0 或 0,1，默认: 0)"
    echo "  --enable-recon        启用重构模式 (默认: 禁用)"
    echo "  --help, -h            显示此帮助信息"
    echo ""
    echo "说明:"
    echo "  这是一个快速测试脚本，专门验证Layer 11 Head的维度匹配问题是否修复。"
    echo "  使用极少的样本数和简化的参数，可以在1-2分钟内完成测试。"
    exit 0
fi

# --- 2. 固定配置 - 快速测试用 ---
MODEL_PATH="../../01_Models/Qwen2.5-7B"  # 使用问题模型
MODEL_NAME="qwen2.5-7b"
PRESERVE_RATIO=0.7  # 使用指定的0.7剪枝比例
DATASET_NAME="c4"   # 使用c4数据集
PRUNE_TYPE="para"
LBOUND=0.2
RBOUND=1.0
N_SAMPLES=4         # 极少的样本数 - 快速测试
RECON_SAMPLE=2      # 极少的重构样本数
SEED=2025

# --- 3. 使用指定的0.7剪枝比例配置 ---
# 这是用户指定的真实剪枝比例，专门测试Layer 11 Head的维度匹配问题
QUICK_RATIOS="1.0, 0.504222972972973, 1.0, 0.2282516891891892, 1.0, 1.0, 1.0, 0.33292863175675674, 1.0, 0.5825063344594594, 1.0, 0.5981313344594594, 1.0, 0.37811444256756754, 1.0, 1.0, 1.0, 1.0, 1.0, 0.32849451013513514, 1.0, 0.44256756756756754, 0.4642857142857143, 0.6009818412162162, 1.0, 1.0, 1.0, 1.0, 0.9285714285714286, 1.0, 1.0, 0.6026182432432432, 1.0, 0.4096283783783784, 1.0, 0.7765519425675675, 0.5357142857142857, 0.23965371621621623, 1.0, 0.4552364864864865, 0.9285714285714286, 0.8490815033783784, 1.0, 0.5503061655405406, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5763830236486487, 1.0, 0.36776815878378377, 1.0, 0.7350612331081081, 1.0, 0.9953019425675675"

echo "使用指定的0.7剪枝比例配置，专门测试Layer 11 Head维度匹配问题"

# --- 4. 生成时间戳和导出路径 ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RECON_SUFFIX=""
if [ "$ENABLE_RECON" = true ]; then
    RECON_SUFFIX="_recon"
fi
EXPORT_PATH="./checkpoints/quick_test_${MODEL_NAME}_layer11${RECON_SUFFIX}_${TIMESTAMP}.pth.tar"

# --- 5. 环境配置 ---
CONDA_ENV_NAME="amc_LLM"
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# --- 6. 重构参数配置 ---
RECON_FLAG=""
RECON_STATUS="禁用 (速度更快)"
if [ "$ENABLE_RECON" = true ]; then
    RECON_FLAG="--recon"
    RECON_STATUS="启用 (精度更高，但较慢)"
fi

# --- 7. 确保目录存在 ---
mkdir -p "$(dirname "$EXPORT_PATH")"

# --- 8. 显示配置信息 ---
echo "=================================================================="
echo "   🚀 快速测试 - Layer 11 Head维度匹配修复验证"
echo "=================================================================="
echo "    测试配置:"
echo "     - 目标问题:         Layer 11 Head维度匹配错误"
echo "     - 测试模型:         ${MODEL_PATH}"
echo "     - 数据集:           ${DATASET_NAME}"
echo "     - 剪枝比例:         0.7 (用户指定的真实配置)"
echo "     - 样本数:           ${N_SAMPLES} (快速模式)"
echo "     - 重构样本:         ${RECON_SAMPLE}"
echo "     - 重构模式:         ${RECON_STATUS}"
echo ""
echo "    GPU配置:"
echo "     - 使用GPU:          ${GPU_ID}"
echo ""
echo "    预期结果:"
echo "     - ✅ 成功: Layer 11 Head维度匹配问题已修复"
echo "     - ❌ 失败: 仍然出现 'mat1 and mat2 shapes cannot be multiplied' 错误"
echo "------------------------------------------------------------------"

# --- 9. GPU状态检查 ---
echo "=> 当前GPU状态检查:"
if command -v nvidia-smi &> /dev/null; then
    echo "=> GPU ${GPU_ID} 状态:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv,noheader,nounits -i ${GPU_ID} 2>/dev/null || echo "   无法查询GPU状态"
else
    echo "   nvidia-smi 不可用，跳过GPU状态检查"
fi
echo ""

# --- 10. 开始测试 ---
echo "🚀 开始快速测试..."
echo "📝 测试命令预览:"
echo "python amc_searchPPO.py --job=export --model=\"${MODEL_PATH}\" --preserve_ratio=${PRESERVE_RATIO} --n_samples=${N_SAMPLES} ${RECON_FLAG}"
echo ""

# 激活conda环境并执行测试
conda activate ${CONDA_ENV_NAME} 2>/dev/null || echo "⚠️  无法激活conda环境，使用系统Python"

python -u amc_searchPPO.py \
    --job=export \
    --model="${MODEL_PATH}" \
    --model_name="${MODEL_NAME}" \
    --dataset_name="${DATASET_NAME}" \
    --preserve_ratio=${PRESERVE_RATIO} \
    --ratios="${QUICK_RATIOS}" \
    --use_real_val \
    --prune="${PRUNE_TYPE}" \
    --structure \
    --state_mode=0 \
    ${RECON_FLAG} \
    --enable_downstream=false \
    --delayed_downstream_eval \
    --lbound=${LBOUND} \
    --rbound=${RBOUND} \
    --n_samples=${N_SAMPLES} \
    --recon_sample=${RECON_SAMPLE} \
    --acc_metric=acc1 \
    --reward=reward_ppl \
    --seed=${SEED} \
    --export_path="${EXPORT_PATH}"

# --- 11. 测试结果分析 ---
TEST_EXIT_CODE=$?
echo ""
echo "=================================================================="
if [ ${TEST_EXIT_CODE} -eq 0 ]; then
    echo "🎉 测试成功完成！"
    echo ""
    echo "✅ Layer 11 Head维度匹配问题修复验证通过！"
    echo "📁 测试导出文件: ${EXPORT_PATH}"
    if [ -f "${EXPORT_PATH}" ]; then
        FILE_SIZE=$(du -h "${EXPORT_PATH}" | cut -f1)
        echo "📊 文件大小: ${FILE_SIZE}"
    fi
    echo ""
    echo "🚀 现在可以安全地进行完整的模型剪枝实验了！"
    echo "💡 建议: 使用你的完整export脚本进行正式剪枝"
else
    echo "❌ 测试失败 (退出码: ${TEST_EXIT_CODE})"
    echo ""
    echo "🔍 可能的原因:"
    echo "   1. Layer 11 Head维度匹配问题仍未解决"
    echo "   2. 环境配置问题"
    echo "   3. 模型路径不正确"
    echo ""
    echo "🛠️  建议检查:"
    echo "   1. 确认模型路径: ${MODEL_PATH}"
    echo "   2. 检查conda环境: ${CONDA_ENV_NAME}"
    echo "   3. 查看错误日志中的具体错误信息"
    echo ""
    echo "📧 如果仍然出现Layer 11 Head的维度匹配错误，请反馈具体错误信息"
fi
echo "=================================================================="
