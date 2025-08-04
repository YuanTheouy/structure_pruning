#!/bin/bash
# =================================================================================
#    FeatureExtractor 快速测试脚本
# =================================================================================
#
#   用法:
#       ./test_feature_extractor.sh              # 基础测试
#       ./test_feature_extractor.sh --gpu-id 1   # 指定GPU
#       ./test_feature_extractor.sh --cpu        # 强制使用CPU
#       ./test_feature_extractor.sh --verbose    # 详细输出
#
# =================================================================================

# 默认配置
GPU_ID=0
FORCE_CPU=false
VERBOSE=false

# 参数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --cpu)
            FORCE_CPU=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "FeatureExtractor 测试脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --gpu-id ID    指定GPU ID (默认: 0)"
            echo "  --cpu          强制使用CPU"
            echo "  --verbose      显示详细输出"
            echo "  --help, -h     显示帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                    # 默认GPU测试"
            echo "  $0 --gpu-id 1         # 使用GPU 1"
            echo "  $0 --cpu              # CPU测试"
            echo "  $0 --verbose          # 详细输出"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看用法"
            exit 1
            ;;
    esac
done

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 输出函数
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示测试配置
echo "=================================================================="
echo "           FeatureExtractor 快速测试"
echo "=================================================================="
print_info "测试配置:"
print_info "  GPU ID: ${GPU_ID}"
print_info "  强制CPU: $([ "$FORCE_CPU" = true ] && echo "是" || echo "否")"
print_info "  详细输出: $([ "$VERBOSE" = true ] && echo "是" || echo "否")"
echo ""

# 检查Python环境
print_info "检查Python环境..."
if ! command -v python3 &> /dev/null; then
    print_error "未找到python3命令"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
print_info "Python版本: ${PYTHON_VERSION}"

# 检查必要的依赖
print_info "检查依赖包..."
REQUIRED_PACKAGES=("torch" "transformers" "tqdm")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import ${package}" &> /dev/null; then
        MISSING_PACKAGES+=("${package}")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    print_error "缺少依赖包: ${MISSING_PACKAGES[*]}"
    print_info "请运行: pip install ${MISSING_PACKAGES[*]}"
    exit 1
fi

print_success "所有依赖包检查通过"

# 检查模型路径
MODEL_PATH="/home/theo/data/yx_repository/01_Models/opt-1.3b"
if [ ! -d "${MODEL_PATH}" ]; then
    print_warning "模型路径不存在: ${MODEL_PATH}"
    print_info "测试将尝试自动下载模型（可能需要较长时间）"
fi

# 设置环境变量
if [ "$FORCE_CPU" = true ]; then
    export CUDA_VISIBLE_DEVICES=""
    print_info "强制使用CPU模式"
else
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    print_info "使用GPU: ${GPU_ID}"
fi

# 设置离线模式（使用与searchPPO13.sh相同的设置）
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 检查测试文件
TEST_SCRIPT="test_feature_extractor.py"
if [ ! -f "${TEST_SCRIPT}" ]; then
    print_error "测试脚本不存在: ${TEST_SCRIPT}"
    exit 1
fi

FEATURE_EXTRACTOR="feature_extractor.py"
if [ ! -f "${FEATURE_EXTRACTOR}" ]; then
    print_error "FeatureExtractor模块不存在: ${FEATURE_EXTRACTOR}"
    exit 1
fi

print_success "所有文件检查通过"

# 创建日志目录
LOG_DIR="./logs/feature_extractor_tests"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/test_${TIMESTAMP}.log"

print_info "日志文件: ${LOG_FILE}"

echo ""
echo "------------------------------------------------------------------"
print_info "开始测试..."
echo ""

# 执行测试
if [ "$VERBOSE" = true ]; then
    # 详细模式：显示所有输出
    python3 -u "${TEST_SCRIPT}" 2>&1 | tee "${LOG_FILE}"
    TEST_EXIT_CODE=${PIPESTATUS[0]}
else
    # 简洁模式：只显示关键信息
    python3 -u "${TEST_SCRIPT}" > "${LOG_FILE}" 2>&1
    TEST_EXIT_CODE=$?
    
    # 提取并显示关键信息
    if [ -f "${LOG_FILE}" ]; then
        echo "测试摘要:"
        grep -E "(🔍|✅|❌|⚠️|🎉|💡)" "${LOG_FILE}" | head -20
        echo ""
        echo "完整日志请查看: ${LOG_FILE}"
    fi
fi

echo ""
echo "=================================================================="

# 检查测试结果
if [ ${TEST_EXIT_CODE} -eq 0 ]; then
    print_success "🎉 FeatureExtractor 测试通过！"
    print_info "✨ 测试结果文件: test_feature_extraction_results.pt"
    print_info "📊 详细日志: ${LOG_FILE}"
    echo ""
    print_info "🚀 下一步建议:"
    print_info "  1. 检查测试结果文件确认特征提取正确"
    print_info "  2. 可以将 FeatureExtractor 集成到训练流程中"
    print_info "  3. 根据需要调整特征提取参数"
else
    print_error "❌ FeatureExtractor 测试失败 (退出码: ${TEST_EXIT_CODE})"
    print_info "📋 请检查日志文件获取详细错误信息: ${LOG_FILE}"
    echo ""
    print_info "🔧 常见问题排查:"
    print_info "  1. 检查模型路径是否正确"
    print_info "  2. 确认GPU内存是否足够"
    print_info "  3. 验证依赖包版本是否兼容"
    print_info "  4. 检查网络连接（如需下载模型）"
fi

echo "=================================================================="

exit ${TEST_EXIT_CODE}
