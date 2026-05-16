#!/bin/bash
# =================================================================================
#    AMC-LLM 模型导出脚本 - OPT-2.7B 剪枝模型导出 (增强版)
# =================================================================================
#
#   用法:
#       1. 基础用法: ./scripts/export13b70.sh
#       2. 指定保留比例: ./scripts/export13b70.sh --preserve-ratio 0.8
#       3. 启用重构: ./scripts/export13b70.sh --enable-recon
#       4. 指定导出路径: ./scripts/export13b70.sh --export-path ./my_model.pth.tar
#       5. 指定数据集: ./scripts/export13b70.sh --dataset wikitext2
#       6. 自定义剪枝比例: ./scripts/export13b70.sh --ratios "1.0,0.5,0.3,..."
#
#   脚本将导出剪枝后的模型到指定路径。
#
# =================================================================================

# --- 1. 参数解析 ---
PRESERVE_RATIO=0.7
ENABLE_RECON=false
DATASET_NAME="wikitext2"
EXPORT_PATH=""
RATIOS=""
SHOW_HELP=false

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
    echo "  --preserve-ratio R    指定保留比例 (默认: 0.7)"
    echo "  --enable-recon        启用重构模式 (默认: 禁用)"
    echo "  --dataset NAME        指定数据集名称 (默认: wikitext2)"
    echo "  --export-path PATH    指定导出路径 (默认: 自动生成)"
    echo "  --ratios RATIOS       指定自定义剪枝比例 (逗号分隔)"
    echo "  --help, -h            显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认设置"
    echo "  $0 --preserve-ratio 0.8               # 保留80%参数"
    echo "  $0 --enable-recon                     # 启用重构模式"
    echo "  $0 --dataset piqa                     # 使用PIQA数据集"
    echo "  $0 --export-path ./my_model.pth.tar   # 指定导出路径"
    exit 0
fi

# --- 2. 预定义剪枝比例配置 ---
# 根据保留比例选择对应的剪枝配置
declare -A ratio_configs
ratio_configs[0.7]="1.0, 0.21568586, 1.0, 0.16064769, 1.0, 0.110139616, 1.0, 0.1, 0.30651495, 0.8920768, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.12824744, 0.8500398, 0.92951375, 0.8426913, 0.34870517, 1.0, 0.3668836, 0.9013828, 0.52173406, 0.9559938, 1.0, 1.0, 1.0, 0.2526906, 1.0, 0.6476018, 0.35733142, 1.0, 0.81383276, 0.57449895, 0.63679165, 0.8023593, 0.7445575, 1.0, 0.47160417, 0.8109849, 0.41409576, 1.0, 1.0, 0.16067912, 0.12299981, 0.27442575, 0.45425782, 1.0, 1.0, 0.16074733, 0.75198144, 0.7685735, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8561277, 0.2455811, 1.0, 0.32607928, 0.9751147, 0.1, 1.0, 1.0, 1.0, 1.0, 0.62583965, 1.0, 1.0, 0.57398236, 1.0, 0.6125404, 1.0, 0.4644709, 0.6135012, 1.0, 0.67438716, 0.792971, 0.51503074, 1.0, 0.4005825, 1.0, 0.500716, 0.6640672, 0.2605796, 1.0, 0.85817933, 0.7441333, 0.122389935, 0.78818, 0.64338136, 0.4360937, 1.0, 1.0, 1.0, 0.32079586, 1.0, 0.2557542, 0.514575, 1.0, 0.4166619, 1.0, 1.0, 0.39750668, 1.0, 0.42414522, 0.6948418, 0.15652226, 0.8909981, 0.70801705, 1.0, 0.74540555, 0.46731043, 1.0, 0.24474807, 0.3350446, 1.0, 0.94691163, 0.26644287, 0.57649404, 1.0, 0.22243112, 0.13302772, 0.1, 1.0, 0.6309164, 0.4571139, 0.6699764, 0.27616164, 0.31397647, 0.64414763, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.54292494, 0.28206664, 1.0, 0.26999402, 0.46170014, 0.25166333, 0.3404555, 1.0, 0.24020813, 1.0, 0.89060426, 1.0, 0.1, 0.104174666, 0.69369614, 0.53600293, 0.5071527, 0.12427671, 0.91327435, 0.8837042, 0.9677308, 0.7363731, 1.0, 1.0, 0.7131211, 0.828337, 1.0, 0.83528286, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.75263494, 1.0, 0.49410814, 1.0, 0.52026385, 0.26465872, 1.0, 0.68830657, 0.57877296, 1.0, 1.0, 0.4301718, 0.21617532, 1.0, 0.87953985"
# ratio_configs[0.7]="1.0, 1.0, 0.2, 0.6986084, 1.0, 0.23278809, 1.0, 0.4937744, 1.0, 0.6965332, 0.84375, 0.45922852, 1.0, 0.2, 1.0, 0.3630371, 1.0, 0.8062744, 1.0, 0.29541016, 1.0, 1.0, 1.0, 0.82592773, 1.0, 0.9406738, 1.0, 0.48132324, 1.0, 0.89746094, 1.0, 0.2, 1.0, 0.2, 1.0, 0.8847656, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 0.2, 0.48449707, 1.0, 0.37390137, 1.0, 0.34182602"
# ratio_configs[0.7]="1.0, 1.0, 0.2, 0.6986084, 1.0, 0.23278809, 1.0, 0.4937744, 1.0, 0.6965332, 1.0, 0.45922852, 1.0, 0.15, 1.0, 0.3630371, 1.0, 0.8062744, 1.0, 0.29541016, 1.0, 1.0, 1.0, 0.82592773, 1.0, 0.9406738, 1.0, 0.48132324, 1.0, 0.89746094, 1.0, 0.15, 1.0, 0.15, 1.0, 0.8847656, 1.0, 1.0, 1.0, 0.15, 1.0, 1.0, 0.2, 0.48449707, 1.0, 0.37390137, 1.0, 0.34182602"

# --- 3. 固定参数配置 ---
MODEL_PATH="/home/theo/data/yx_repository/01_Models/opt-2.7b"
MODEL_NAME="opt-2.7b"
PRUNE_TYPE="para"
LBOUND=0.1
RBOUND=1.0
N_SAMPLES=64
RECON_SAMPLE=32
SEED=2024

# --- 4. 自动选择剪枝比例配置 ---
if [ -z "$RATIOS" ]; then
    RATIOS=${ratio_configs[$PRESERVE_RATIO]}
    if [ -z "$RATIOS" ]; then
        echo "警告: 没有为保留比例 ${PRESERVE_RATIO} 预定义剪枝配置，将使用0.7的配置"
        RATIOS=${ratio_configs[0.7]}
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
    EXPORT_PATH="./checkpoints/opt27b_${RATIO_SUFFIX}_${DATASET_NAME}${RECON_SUFFIX}_${TIMESTAMP}_export.pth.tar"
fi

# --- 6. 环境变量设置 ---
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Force use of CUDA 12 libraries and exclude all other CUDA library paths
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/targets/x86_64-linux/lib:/usr/local/cuda/lib64
# Remove conda env lib path that might have conflicting CUDA libraries  
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "/home/theo/data/anaconda3/envs/amc_LLM/lib" | tr '\n' ':' | sed 's/:$//')
# Set CUPY to use specific CUDA installation
export CUDA_PATH=/usr/local/cuda-12.9

# --- 7. 重构参数配置 ---
if [ "$ENABLE_RECON" = true ]; then
    RECON_FLAG="--recon"
    RECON_STATUS="启用 (精度更高，速度较慢)"
else
    RECON_FLAG=""
    RECON_STATUS="禁用 (速度更快，精度稍低)"
fi

# --- 8. 确保导出目录存在 ---
mkdir -p "$(dirname "$EXPORT_PATH")"

# --- 9. 显示配置信息 ---
echo "=================================================================="
echo "   AMC-LLM 模型导出 - OPT-2.7B 剪枝模型"
echo "=================================================================="
echo "    导出配置:"
echo "     - 模型路径:         ${MODEL_PATH}"
echo "     - 数据集:           ${DATASET_NAME}"
echo "     - 保留比例:         ${PRESERVE_RATIO}"
echo "     - 剪枝类型:         ${PRUNE_TYPE}"
echo "     - 重构模式:         ${RECON_STATUS}"
echo "     - 随机种子:         ${SEED}"
echo ""
echo "    技术参数:"
echo "     - 样本数量:         ${N_SAMPLES}"
echo "     - 重构样本:         ${RECON_SAMPLE}"
echo "     - 下边界:           ${LBOUND}"
echo "     - 上边界:           ${RBOUND}"
echo ""
echo "    输出路径:"
echo "     - 导出文件:         ${EXPORT_PATH}"
echo "------------------------------------------------------------------"
echo "  开始导出模型... "
echo ""

# --- 10. 执行导出命令 ---
python -u amc_searchPPO.py \
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
    echo "保留比例: ${PRESERVE_RATIO} | 重构模式: $([ "$ENABLE_RECON" = true ] && echo "已启用" || echo "已禁用")"
    
    # 显示文件大小信息
    if [ -f "${EXPORT_PATH}" ]; then
        FILE_SIZE=$(du -h "${EXPORT_PATH}" | cut -f1)
        echo "文件大小: ${FILE_SIZE}"
    fi
else
    echo "模型导出失败 (退出码: ${EXPORT_EXIT_CODE})"
fi
echo "=================================================================="
