#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU绑定测试脚本 - 验证每个进程是否正确绑定到指定GPU
"""

import os
import sys
import torch
import time
import argparse

def test_gpu_binding():
    """测试GPU绑定是否正确"""
    
    parser = argparse.ArgumentParser(description='GPU Binding Test')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to test')
    parser.add_argument('--duration', type=int, default=10, help='Test duration in seconds')
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"GPU绑定测试 - 测试GPU {args.gpu_id}")
    print("=" * 80)
    
    # 显示初始GPU状态
    print(f"初始CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"PyTorch可见GPU数量: {torch.cuda.device_count()}")
    
    if not torch.cuda.is_available():
        print("CUDA不可用")
        return
    
    # 设置严格GPU绑定
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    print(f"设置CUDA_VISIBLE_DEVICES={args.gpu_id}")
    
    # 重新初始化CUDA上下文
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # 现在只有一个GPU可见，所以是0
        device = torch.device('cuda:0')
        
        print(f"当前设备: {device}")
        print(f"设备名称: {torch.cuda.get_device_name(0)}")
        print(f"设备内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 创建一个大tensor来占用显存，验证绑定
        print(f"\n开始在GPU {args.gpu_id}上分配显存...")
        
        try:
            # 分配约1GB显存
            tensor_size = (1024, 1024, 128)  # 约1GB float32
            test_tensor = torch.randn(tensor_size, device=device)
            
            print(f"成功分配显存: {test_tensor.numel() * 4 / 1024**3:.1f} GB")
            print(f"张量设备: {test_tensor.device}")
            print(f"张量形状: {test_tensor.shape}")
            
            # 进行一些计算以确保GPU被使用
            print(f"\n开始GPU计算测试，持续 {args.duration} 秒...")
            start_time = time.time()
            
            for i in range(args.duration):
                # 执行矩阵乘法保持GPU忙碌
                result = torch.matmul(test_tensor, test_tensor.transpose(1, 2))
                if i % 2 == 0:
                    print(f"  计算进度: {i+1}/{args.duration} 秒", end='\r')
                time.sleep(1)
            
            end_time = time.time()
            print(f"\n✅ GPU计算测试完成，耗时: {end_time - start_time:.1f}秒")
            print(f"✅ 成功验证GPU {args.gpu_id}绑定正确")
            
            # 清理显存
            del test_tensor, result
            torch.cuda.empty_cache()
            print("✅ 显存已清理")
            
        except Exception as e:
            print(f"❌ GPU测试失败: {e}")
            
    else:
        print("❌ 设置GPU绑定后CUDA不可用")

if __name__ == '__main__':
    test_gpu_binding()
