#!/usr/bin/env python3
"""
测试优化的Ridge回归实现
测试单GPU优化的Ridge回归性能和内存使用情况
"""

import torch
import numpy as np
import time
import gc
from lib.Ridge import Ridge_Regression
from tqdm import tqdm

def test_optimized_ridge_regression():
    """测试优化的Ridge回归实现"""
    print("🔬 Testing Optimized Ridge Regression Implementation")
    print("=" * 60)
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        
        # 显示GPU内存信息
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"📊 Total GPU Memory: {total_memory:.1f} GB")
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        gc.collect()
        
        allocated_before = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"📊 GPU Memory Before Test: {allocated_before:.3f} GB")
    else:
        device = torch.device('cpu')
        print("⚠️ CUDA not available, using CPU")
        return
    
    # 模拟真实的特征矩阵大小 (类似于神经网络层的特征)
    n_samples = 1000      # 样本数量
    n_features = 4096     # 特征维度 (模拟hidden_size)
    n_targets = 512       # 目标数量 (模拟要重建的神经元数)
    
    print(f"📐 Test Matrix Dimensions:")
    print(f"   - Samples: {n_samples}")
    print(f"   - Features: {n_features}")
    print(f"   - Targets: {n_targets}")
    
    # 生成模拟数据
    print("\n🔄 Generating test data...")
    torch.manual_seed(42)
    
    # 使用bfloat16以节省内存 (与实际模型一致)
    A = torch.randn(n_samples, n_features, dtype=torch.bfloat16, device=device)
    B_full = torch.randn(n_samples, n_targets, dtype=torch.bfloat16, device=device)
    
    # 模拟选择的神经元索引
    selected_indices = torch.randperm(n_targets)[:n_targets//2].sort()[0]
    A_selected = A[:, selected_indices]
    
    print(f"✅ Test data generated")
    print(f"   - A shape: {A.shape}")
    print(f"   - A_selected shape: {A_selected.shape}")
    print(f"   - B_full shape: {B_full.shape}")
    
    allocated_after_data = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"📊 GPU Memory After Data Generation: {allocated_after_data:.3f} GB")
    
    # 测试Ridge回归性能
    print("\n🧮 Testing Ridge Regression Performance...")
    
    ridge_times = []
    memory_peaks = []
    
    # 测试多个神经元的重建
    test_neurons = min(50, n_targets)  # 测试前50个神经元
    
    print(f"🔍 Testing reconstruction for {test_neurons} neurons...")
    
    pbar = tqdm(range(test_neurons), desc="Ridge Regression Test")
    
    for i in pbar:
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated(0) / (1024**3)
        
        start_time = time.time()
        
        # 模拟实际的Ridge回归调用
        B_target = B_full[:, i]
        
        try:
            # 创建Ridge回归实例
            ridge_reg = Ridge_Regression(
                A_selected.to(dtype=torch.float32), 
                B_target.to(dtype=torch.float32), 
                alpha=0.9, 
                fit_intercept=False, 
                device=device
            )
            
            # 执行拟合
            scale_factors = ridge_reg.fit()
            
            end_time = time.time()
            ridge_time = end_time - start_time
            ridge_times.append(ridge_time)
            
            # 记录内存峰值
            peak_memory = torch.cuda.max_memory_allocated(0) / (1024**3)
            memory_peaks.append(peak_memory)
            torch.cuda.reset_peak_memory_stats(0)
            
            # 更新进度条
            pbar.set_postfix({
                'Time': f'{ridge_time:.3f}s',
                'Memory': f'{peak_memory:.3f}GB'
            })
            
        except Exception as e:
            print(f"\n❌ Error at neuron {i}: {e}")
            break
        
        # 每10次迭代清理一次内存
        if i % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    pbar.close()
    
    # 计算统计信息
    if ridge_times:
        avg_time = np.mean(ridge_times)
        std_time = np.std(ridge_times)
        min_time = np.min(ridge_times)
        max_time = np.max(ridge_times)
        
        avg_memory = np.mean(memory_peaks)
        max_memory = np.max(memory_peaks)
        
        print(f"\n📈 Performance Statistics:")
        print(f"   - Average time per neuron: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"   - Time range: {min_time:.3f}s - {max_time:.3f}s")
        print(f"   - Average memory usage: {avg_memory:.3f} GB")
        print(f"   - Peak memory usage: {max_memory:.3f} GB")
        
        # 估算全规模性能
        estimated_total_time = avg_time * n_targets
        print(f"\n🔮 Estimated Performance for Full Scale:")
        print(f"   - Total time for {n_targets} neurons: {estimated_total_time:.1f}s ({estimated_total_time/60:.1f} min)")
        print(f"   - Throughput: {1/avg_time:.1f} neurons/second")
        
        # 内存效率分析
        final_memory = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"\n💾 Memory Analysis:")
        print(f"   - Memory before test: {allocated_before:.3f} GB")
        print(f"   - Memory after data: {allocated_after_data:.3f} GB")
        print(f"   - Memory after test: {final_memory:.3f} GB")
        print(f"   - Memory overhead: {final_memory - allocated_before:.3f} GB")
        
        if avg_time < 0.1:
            print("\n✅ EXCELLENT: Ridge regression is very fast!")
        elif avg_time < 0.5:
            print("\n✅ GOOD: Ridge regression performance is acceptable")
        else:
            print("\n⚠️ SLOW: Ridge regression might need further optimization")
            
        if max_memory < 2.0:
            print("✅ EXCELLENT: Memory usage is very efficient!")
        elif max_memory < 4.0:
            print("✅ GOOD: Memory usage is reasonable")
        else:
            print("⚠️ HIGH: Memory usage might be concerning for large models")
    
    else:
        print("❌ No successful Ridge regression tests completed")
    
    # 最终清理
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\n🏁 Test completed!")

if __name__ == "__main__":
    test_optimized_ridge_regression()
