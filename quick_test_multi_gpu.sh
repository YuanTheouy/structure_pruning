#!/bin/bash

echo "=== Layer 11 Head 剪枝维度修复测试 ==="
echo "测试时间: $(date)"
echo ""
echo "目标: 验证Layer 11 Head剪枝时的'mat1 and mat2 shapes cannot be multiplied'错误是否修复"
echo ""

# 设置环境
export CUDA_VISIBLE_DEVICES=0,1

# 运行精确的Layer 11测试，不进行完整训练
python test_layer11_fix.py 2>&1 | tee layer11_fix_test_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=== 测试完成 ==="
echo "如果看到'🎉 所有测试通过！'，说明修复成功！"
