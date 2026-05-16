#!/bin/bash
# =================================================================================
#    AMC-LLM 剪枝模型重建与评估脚本 - 基于已导出的模型
# =================================================================================
#
#   用法:
#       该脚本用于对已导出的剪枝模型进行重建（重构）和全面评估。
#
#   功能:
#       1. 重新运行导出过程，但启用重构模式
#       2. 启用下游任务评估
#       3. 对比重构前后的性能差异
#
#   示例:
#       # 重建您刚才导出的Qwen2.5-7B模型
#       ./rebuild_and_evaluate.sh
#
# =================================================================================

# --- 配置参数 ---
MODEL_PATH="/home/yx/yx_repository/01_Models/Qwen2.5-7B"  # 根据您的实际路径修改
MODEL_NAME="qwen2.5-7b"
DATASET_NAME="wikitext2"
PRESERVE_RATIO=0.7
ENABLE_RECON=true
ENABLE_DOWNSTREAM=true
N_SAMPLES=64
RECON_SAMPLE=32
SEED=2025

# 您之前导出的剪枝比例（从您的导出日志中复制）
RATIOS="1.0, 0.504222972972973, 1.0, 0.2282516891891892, 1.0, 1.0, 1.0, 0.33292863175675674, 1.0, 0.5825063344594594, 1.0, 0.5981313344594594, 1.0, 0.37811444256756754, 1.0, 1.0, 1.0, 1.0, 1.0, 0.32849451013513514, 1.0, 0.44256756756756754, 0.4642857142857143, 0.6009818412162162, 1.0, 1.0, 1.0, 1.0, 0.9285714285714286, 1.0, 1.0, 0.6026182432432432, 1.0, 0.4096283783783784, 1.0, 0.7765519425675675, 0.5357142857142857, 0.23965371621621623, 1.0, 0.4552364864864865, 0.9285714285714286, 0.8490815033783784, 1.0, 0.5503061655405406, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5763830236486487, 1.0, 0.36776815878378377, 1.0, 0.7350612331081081, 1.0, 0.9953019425675675"

# --- 自动生成导出路径 ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPORT_PATH="./checkpoints/${MODEL_NAME}_${PRESERVE_RATIO//./_}_${DATASET_NAME}_recon_${TIMESTAMP}_export.pth.tar"

# --- 环境配置 ---
CONDA_ENV_NAME="amc_llm"
PYTHON_EXECUTABLE=$(conda run -n ${CONDA_ENV_NAME} which python 2>/dev/null || which python)

if ! command -v $PYTHON_EXECUTABLE &> /dev/null; then
    echo "错误: 找不到Python解释器。请确认Conda环境 '${CONDA_ENV_NAME}' 存在。"
    exit 1
fi

# --- 确保导出目录存在 ---
mkdir -p "$(dirname "$EXPORT_PATH")"

echo "=================================================================="
echo "   AMC-LLM 剪枝模型重建与评估"
echo "=================================================================="
echo "    配置信息:"
echo "     - 模型路径:         ${MODEL_PATH}"
echo "     - 数据集:           ${DATASET_NAME}"
echo "     - 保留比例:         ${PRESERVE_RATIO}"
echo "     - 重构模式:         启用 (精度更高)"
echo "     - 下游任务评估:     启用 (完整评估)"
echo "     - 输出路径:         ${EXPORT_PATH}"
echo ""
echo "    重构参数:"
echo "     - 重构样本数:       ${RECON_SAMPLE}"
echo "     - 特征样本数:       ${N_SAMPLES}"
echo "     - 随机种子:         ${SEED}"
echo "=================================================================="
echo ""

echo "=> 开始执行剪枝、重构和评估..."
echo ""

# --- 执行重建命令 ---
${PYTHON_EXECUTABLE} -u amc_searchPPO.py \
    --job=export \
    --model="${MODEL_PATH}" \
    --model_name="${MODEL_NAME}" \
    --dataset_name="${DATASET_NAME}" \
    --preserve_ratio=${PRESERVE_RATIO} \
    --ratios="${RATIOS}" \
    --use_real_val \
    --prune=para \
    --structure \
    --state_mode=0 \
    --recon \
    --enable_downstream=true \
    --delayed_downstream_eval \
    --lbound=0.1 \
    --rbound=1.0 \
    --n_samples=${N_SAMPLES} \
    --recon_sample=${RECON_SAMPLE} \
    --acc_metric=acc1 \
    --reward=reward_ppl \
    --seed=${SEED} \
    --export_path="${EXPORT_PATH}"

# --- 检查执行结果 ---
EXPORT_EXIT_CODE=$?
echo ""
echo "=================================================================="
if [ ${EXPORT_EXIT_CODE} -eq 0 ]; then
    echo "✅ 模型重建与评估完成！"
    echo ""
    echo "输出文件信息:"
    echo "  - 重构后模型: ${EXPORT_PATH}"
    if [ -f "${EXPORT_PATH}" ]; then
        FILE_SIZE=$(du -h "${EXPORT_PATH}" | cut -f1)
        echo "  - 文件大小: ${FILE_SIZE}"
    fi
    echo ""
    echo "对比说明:"
    echo "  - 此次运行启用了重构模式，模型精度应该比之前的导出版本更高"
    echo "  - 同时进行了完整的下游任务评估，可以看到模型在各个任务上的表现"
    echo ""
    echo "下一步建议:"
    echo "  1. 比较重构前后的PPL差异"
    echo "  2. 分析下游任务的性能表现"
    echo "  3. 根据需要调整剪枝比例或重构参数"
else
    echo "❌ 模型重建失败 (退出码: ${EXPORT_EXIT_CODE})"
    echo "请检查错误信息并重试"
fi
echo "=================================================================="
