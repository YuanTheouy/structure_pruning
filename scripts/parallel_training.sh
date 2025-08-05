#!/bin/bash
# =================================================================================
#    多GPU并行训练启动脚本 - 自动在不同GPU上启动多个实验
# =================================================================================

set -e  # 遇到错误立即退出

# 配置区域
GPUS=(0 1 2 3)  # 可用的GPU列表
FEATURE_CONFIGS=("minimal" "basic" "attention" "comprehensive")  # 要测试的特征配置
PRESERVE_RATIO=0.7
TRAIN_EPISODES=1000  # 测试用较短训练
BASE_EXP_ID=0

echo "=================================================================="
echo "           多GPU并行训练启动器"
echo "=================================================================="
echo "GPU列表: ${GPUS[*]}"
echo "特征配置: ${FEATURE_CONFIGS[*]}"
echo "保留比例: ${PRESERVE_RATIO}"
echo "训练轮数: ${TRAIN_EPISODES}"
echo "=================================================================="

# 确保脚本目录存在
if [ ! -f "./scripts/searchPPO13.sh" ]; then
    echo "错误: 找不到 ./scripts/searchPPO13.sh"
    echo "请确保在项目根目录运行此脚本"
    exit 1
fi

# 创建日志目录
LOG_DIR="./logs/parallel_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${LOG_DIR}
echo "并行训练日志目录: ${LOG_DIR}"

# 启动进程数组
declare -a pids=()

# 启动并行训练
for i in "${!FEATURE_CONFIGS[@]}"; do
    gpu_id=${GPUS[$i % ${#GPUS[@]}]}  # 循环使用GPU
    feature_config=${FEATURE_CONFIGS[$i]}
    exp_id=$((BASE_EXP_ID + i))
    
    log_file="${LOG_DIR}/gpu${gpu_id}_${feature_config}_exp${exp_id}.log"
    
    echo "启动实验 #${exp_id}: GPU ${gpu_id}, 特征配置 ${feature_config}"
    echo "  日志文件: ${log_file}"
    
    # 在后台启动训练，重定向输出到日志文件
    (
        echo "========================================================================================"
        echo "实验 #${exp_id} 开始: GPU ${gpu_id}, 特征配置 ${feature_config}"
        echo "开始时间: $(date)"
        echo "========================================================================================"
        
        ./scripts/searchPPO13.sh \
            --gpu-id ${gpu_id} \
            --exp-id ${exp_id} \
            --feature-config ${feature_config} \
            --preserve-ratio ${PRESERVE_RATIO} \
            --train-episodes ${TRAIN_EPISODES} \
            --disable-downstream
            
        echo "========================================================================================"
        echo "实验 #${exp_id} 完成: GPU ${gpu_id}, 特征配置 ${feature_config}"
        echo "结束时间: $(date)"
        echo "========================================================================================"
    ) > "${log_file}" 2>&1 &
    
    # 记录进程ID
    pid=$!
    pids+=($pid)
    echo "  进程ID: $pid"
    
    # 稍微延迟以避免同时启动冲突
    sleep 10
done

echo ""
echo "=================================================================="
echo "所有训练进程已启动！"
echo "=================================================================="
echo "活跃进程:"
for i in "${!pids[@]}"; do
    gpu_id=${GPUS[$i % ${#GPUS[@]}]}
    feature_config=${FEATURE_CONFIGS[$i]}
    exp_id=$((BASE_EXP_ID + i))
    echo "  实验 #${exp_id} (GPU ${gpu_id}, ${feature_config}): PID ${pids[$i]}"
done

echo ""
echo "监控命令:"
echo "  查看GPU使用情况: watch -n 2 nvidia-smi"
echo "  查看实时日志: tail -f ${LOG_DIR}/gpu*_*.log"
echo "  查看进程状态: ps -p $(IFS=,; echo \"${pids[*]}\")"
echo ""

# 等待所有进程完成（可选）
read -p "是否等待所有训练完成？(y/N): " wait_for_completion

if [[ $wait_for_completion =~ ^[Yy]$ ]]; then
    echo "等待所有训练进程完成..."
    for i in "${!pids[@]}"; do
        gpu_id=${GPUS[$i % ${#GPUS[@]}]}
        feature_config=${FEATURE_CONFIGS[$i]}
        exp_id=$((BASE_EXP_ID + i))
        pid=${pids[$i]}
        
        echo "等待实验 #${exp_id} (GPU ${gpu_id}, ${feature_config}) 完成..."
        wait $pid
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "✅ 实验 #${exp_id} 成功完成"
        else
            echo "❌ 实验 #${exp_id} 失败 (退出码: $exit_code)"
        fi
    done
    
    echo ""
    echo "=================================================================="
    echo "🎉 所有并行训练已完成！"
    echo "=================================================================="
    echo "日志目录: ${LOG_DIR}"
    echo "查看结果: ls -la ${LOG_DIR}/"
else
    echo ""
    echo "=================================================================="
    echo "并行训练已在后台运行"
    echo "=================================================================="
    echo "监控命令:"
    echo "  nvidia-smi                                    # 查看GPU状态"
    echo "  tail -f ${LOG_DIR}/gpu*_*.log                # 查看训练日志"
    echo "  ps -p $(IFS=,; echo \"${pids[*]}\")         # 查看进程状态"
    echo "  kill $(IFS=' '; echo \"${pids[*]}\")        # 停止所有训练(如需要)"
fi
