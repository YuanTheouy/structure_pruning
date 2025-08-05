#!/usr/bin/env python3
"""
简单的GPU绑定测试脚本
验证CUDA_VISIBLE_DEVICES是否正确限制了GPU可见性
"""

import os
import sys
import torch
import argparse

def test_gpu_binding():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', type=int, required=True, help='GPU ID to test')
    args = parser.parse_args()
    
    print("="*50)
    print("GPU绑定测试")
    print("="*50)
    
    # 1. 检查环境变量
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT_SET')
    print(f"1. CUDA_VISIBLE_DEVICES = {cuda_visible}")
    
    # 2. 强制设置CUDA_VISIBLE_DEVICES（模拟主程序行为）
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    print(f"2. 强制设置 CUDA_VISIBLE_DEVICES = {args.gpu_id}")
    
    # 3. 检查PyTorch可见的GPU数量
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"3. PyTorch检测到的GPU数量: {device_count}")
        
        if device_count == 1:
            print("✅ GPU绑定成功！只有1个GPU可见")
            
            # 4. 测试GPU设备
            torch.cuda.set_device(0)
            device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device)
            print(f"4. 当前设备: cuda:{device}")
            print(f"5. 设备名称: {device_name}")
            
            # 5. 分配内存测试
            try:
                test_tensor = torch.randn(1000, 1000, device='cuda:0')
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
                print(f"6. 内存分配测试通过，已分配: {memory_allocated:.1f} MB")
                
                # 清理
                del test_tensor
                torch.cuda.empty_cache()
                print("7. 内存清理完成")
                print("✅ 所有测试通过！GPU绑定正常")
                
            except Exception as e:
                print(f"❌ 内存分配测试失败: {e}")
                return False
                
        else:
            print(f"❌ GPU绑定失败！检测到{device_count}个GPU，应该只有1个")
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                print(f"   GPU {i}: {name}")
            return False
            
    else:
        print("❌ CUDA不可用")
        return False
    
    return True

if __name__ == "__main__":
    success = test_gpu_binding()
    sys.exit(0 if success else 1)
