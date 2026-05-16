#!/bin/bash
# =================================================================================
#    剪枝模型转换脚本 - 将.pth.tar文件转换为Hugging Face格式
# =================================================================================
#
#   用法:
#       1. 基础用法: ./scripts/convert_pruned_model.sh
#       2. 指定输入文件: ./scripts/convert_pruned_model.sh --checkpoint ./checkpoints/my_model.pth.tar
#       3. 指定输出目录: ./scripts/convert_pruned_model.sh --output ./my_converted_model
#       4. 指定基础模型: ./scripts/convert_pruned_model.sh --base-model /path/to/llama-2-7b-hf
#       5. 组合使用: ./scripts/convert_pruned_model.sh --checkpoint ./checkpoints/my_model.pth.tar --output ./my_output
#
#   脚本将把剪枝模型转换为标准的Hugging Face格式，便于使用lm-evaluation-harness评估。
#
# =================================================================================

# --- 1. 参数解析 ---
BASE_MODEL="../../01_Models/llama-2-7b-hf"
CHECKPOINT_FILE="./checkpoints/llama2_7b_0_7_wikitext2_20250912_141938_export.pth.tar"
OUTPUT_DIR="./converted_models/llama2_7b_pruned_0_7"
SHOW_HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
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
    echo "  --base-model PATH     指定基础模型路径 (默认: /home/theo/data/yx_repository/01_Models/llama-2-7b-hf)"
    echo "  --checkpoint PATH     指定剪枝模型检查点文件 (默认: ./checkpoints/llama2_7b_0_7_wikitext2_20250912_141938_export.pth.tar)"
    echo "  --output PATH         指定输出目录 (默认: ./converted_models/llama2_7b_pruned_0_7)"
    echo "  --help, -h            显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                                    # 使用默认设置"
    echo "  $0 --checkpoint ./checkpoints/my_model.pth.tar        # 指定检查点文件"
    echo "  $0 --output ./my_converted_model                      # 指定输出目录"
    echo "  $0 --base-model /path/to/model --output ./output      # 指定基础模型和输出"
    exit 0
fi

# --- 2. 环境配置 ---
CONDA_ENV_NAME="amc_llm" # 修改为您的Conda环境名称
PYTHON_EXECUTABLE=$(conda run -n ${CONDA_ENV_NAME} which python 2>/dev/null || which python)
if ! command -v $PYTHON_EXECUTABLE &> /dev/null; then
    echo "错误: 找不到 Python 解释器。请激活 Conda 环境或检查路径。"
    exit 1
fi

# --- 3. 验证输入文件 ---
if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "错误: 检查点文件不存在: $CHECKPOINT_FILE"
    exit 1
fi

if [ ! -d "$BASE_MODEL" ]; then
    echo "错误: 基础模型目录不存在: $BASE_MODEL"
    exit 1
fi

# --- 4. 确保输出目录的父目录存在 ---
mkdir -p "$(dirname "$OUTPUT_DIR")"

# --- 5. 显示配置信息 ---
echo "=================================================================="
echo "   剪枝模型转换 - .pth.tar 转 Hugging Face 格式"
echo "=================================================================="
echo "    转换配置:"
echo "     - 基础模型路径:     ${BASE_MODEL}"
echo "     - 检查点文件:       ${CHECKPOINT_FILE}"
echo "     - 输出目录:         ${OUTPUT_DIR}"
echo ""
echo "    文件信息:"
if [ -f "${CHECKPOINT_FILE}" ]; then
    CHECKPOINT_SIZE=$(du -h "${CHECKPOINT_FILE}" | cut -f1)
    echo "     - 检查点大小:       ${CHECKPOINT_SIZE}"
fi
echo "------------------------------------------------------------------"
echo "  开始转换模型... "
echo ""

# --- 6. 执行转换命令 ---
echo "=> 开始转换剪枝模型..."

${PYTHON_EXECUTABLE} convert_model.py \
    --base_model_path "${BASE_MODEL}" \
    --checkpoint_path "${CHECKPOINT_FILE}" \
    --output_path "${OUTPUT_DIR}"

# --- 7. 转换完成提示 ---
CONVERT_EXIT_CODE=$?
echo ""
echo "=================================================================="
if [ ${CONVERT_EXIT_CODE} -eq 0 ]; then
    echo "剪枝模型转换完成！"
    echo "转换后的模型位于: ${OUTPUT_DIR}"
    
    # 显示转换后的目录信息
    if [ -d "${OUTPUT_DIR}" ]; then
        echo ""
        echo "转换后的文件列表:"
        ls -la "${OUTPUT_DIR}"
        echo ""
        TOTAL_SIZE=$(du -sh "${OUTPUT_DIR}" | cut -f1)
        echo "转换后模型总大小: ${TOTAL_SIZE}"
        
        echo ""
        echo "您现在可以使用以下命令来评估转换后的模型:"
        echo "lm_eval --model hf \\"
        echo "    --model_args pretrained=${OUTPUT_DIR},trust_remote_code=True \\"
        echo "    --tasks piqa,winogrande,hellaswag \\"
        echo "    --device cuda:0 \\"
        echo "    --batch_size auto"
    fi
else
    echo "剪枝模型转换失败 (退出码: ${CONVERT_EXIT_CODE})"
fi
echo "=================================================================="
