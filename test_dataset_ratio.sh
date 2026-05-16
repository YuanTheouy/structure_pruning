#!/bin/bash
# 测试数据集渐进增长功能（比例制）

echo "=== 测试数据集渐进增长功能（比例制）==="
echo ""

echo "1. 测试帮助信息："
./scripts/searchPPO13.sh --help | grep -A 20 "数据集渐进增长"

echo ""
echo "2. 测试启用数据集渐进增长（从5%到100%验证集）："
echo "   命令: ./scripts/searchPPO13.sh --use-dataset-growth --dataset-initial-ratio 0.05 --dataset-final-ratio 1.0 --dataset-growth-end-episode 50 --train-episodes 10"

echo ""
echo "3. 验证参数传递..."
# 只运行很短的时间来验证参数正确传递
timeout 30s ./scripts/searchPPO13.sh \
    --use-dataset-growth \
    --dataset-initial-ratio 0.05 \
    --dataset-final-ratio 1.0 \
    --dataset-growth-start-episode 0 \
    --dataset-growth-end-episode 50 \
    --train-episodes 10 \
    --target-sparsity 0.2 2>&1 | head -50

echo ""
echo "=== 测试完成 ==="
