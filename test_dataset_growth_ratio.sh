#!/bin/bash

echo "测试数据集渐进增长功能（比例制）..."

# 测试1：启用数据集渐进增长，从5%到100%
echo "=== 测试1: 从5%验证集增长到100% ==="
./scripts/searchPPO13.sh \
    --use-dataset-growth \
    --dataset-initial-ratio 0.05 \
    --dataset-final-ratio 1.0 \
    --dataset-growth-end-episode 50 \
    --train-episodes 100 \
    --target-sparsity 0.2

echo ""
echo "测试完成！"
