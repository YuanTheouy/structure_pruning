#!/bin/bash
# 测试数据集渐进增长功能

echo "=== 测试数据集渐进增长功能 ==="
echo "运行少量episode来验证功能是否正常..."

# 运行带有数据集增长的训练，只执行很少的episode进行测试
./scripts/searchPPO13.sh \
    --train-episodes 5 \
    --use-dataset-growth \
    --dataset-initial-size 16 \
    --dataset-final-size 32 \
    --dataset-growth-end-episode 3 \
    --use-gradual-pruning \
    --gradual-pruning-end-episode 2 \
    --gradual-initial-sparsity 0.1 \
    --target-sparsity 0.2

echo "=== 测试完成 ==="
