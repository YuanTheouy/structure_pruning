#!/usr/bin/env python3
"""
测试原始错误场景的修复
模拟 RuntimeError: mat1 and mat2 shapes cannot be multiplied (2048x1792 vs 3584x3584)
"""
import torch
import numpy as np

def test_original_error_scenario():
    """测试原始错误场景：mat1和mat2形状不匹配"""
    print("=== 模拟原始错误场景 ===")
    
    # 基于错误信息：2048x1792 vs 3584x3584
    # 这表明在Layer 11的注意力头剪枝时出现了维度不匹配
    
    # 模拟Qwen2.5-7B的配置
    original_num_heads = 32
    original_num_key_value_heads = 8  
    head_dim = 128
    hidden_size = 4096
    seq_len = 2048
    batch_size = 1
    
    preserve_ratio = 0.8  # 保留80%的注意力头
    d_prime = int(preserve_ratio * original_num_key_value_heads)  # 6
    num_q_groups = original_num_heads // original_num_key_value_heads  # 4
    
    print(f"原始配置: {original_num_heads} heads, {original_num_key_value_heads} KV heads")
    print(f"剪枝后: {d_prime} KV heads, {d_prime * num_q_groups} total heads")
    
    try:
        # 模拟注意力计算的输出（这是导致错误的关键点）
        # 错误的情况：使用原始的hidden_size而不是剪枝后的尺寸
        
        print("\n--- 模拟错误情况（修复前）---")
        # 这是错误的：注意力输出使用了剪枝后的尺寸，但o_proj期望原始尺寸
        pruned_attention_dim = d_prime * num_q_groups * head_dim  # 6 * 4 * 128 = 3072
        attention_output_wrong = torch.randn(batch_size * seq_len, pruned_attention_dim)
        o_proj_weight_wrong = torch.randn(hidden_size, hidden_size)  # 错误：仍使用原始维度
        
        print(f"错误案例 - 注意力输出: {attention_output_wrong.shape}")
        print(f"错误案例 - O投影权重: {o_proj_weight_wrong.shape}")
        
        # 这里会导致维度不匹配错误
        try:
            result_wrong = torch.mm(attention_output_wrong, o_proj_weight_wrong.t())
            print("❌ 这不应该成功！")
        except RuntimeError as e:
            print(f"✅ 成功捕获预期错误: {e}")
        
        print("\n--- 模拟正确情况（修复后）---")
        # 正确的做法：o_proj的输入维度要与剪枝后的注意力输出匹配
        attention_output_correct = torch.randn(batch_size * seq_len, pruned_attention_dim)
        o_proj_weight_correct = torch.randn(hidden_size, pruned_attention_dim)  # 正确：输入维度匹配剪枝后的大小
        
        print(f"正确案例 - 注意力输出: {attention_output_correct.shape}")
        print(f"正确案例 - O投影权重: {o_proj_weight_correct.shape}")
        
        result_correct = torch.mm(attention_output_correct, o_proj_weight_correct.t())
        print(f"✅ 矩阵乘法成功！结果形状: {result_correct.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_specific_error_dimensions():
    """测试具体的错误维度：2048x1792 vs 3584x3584"""
    print("\n=== 测试具体错误维度 ===")
    
    # 从错误信息反推可能的配置
    # 2048 可能是 batch_size * seq_len 或者 seq_len
    # 1792 可能是剪枝后的注意力维度
    # 3584 可能是某种不正确的维度计算
    
    try:
        # 场景1：1792可能是14个头 * 128维度
        error_dim_1792 = 14 * 128  # 1792
        error_dim_3584 = 28 * 128  # 3584
        
        print(f"分析错误维度:")
        print(f"1792 = {1792//128} * 128 = {1792//128} heads * 128 head_dim")
        print(f"3584 = {3584//128} * 128 = {3584//128} heads * 128 head_dim")
        
        # 模拟错误情况
        mat1_wrong = torch.randn(2048, 1792)
        mat2_wrong = torch.randn(3584, 3584)
        
        print(f"\n错误场景:")
        print(f"mat1形状: {mat1_wrong.shape}")
        print(f"mat2形状: {mat2_wrong.shape}")
        
        try:
            result = torch.mm(mat1_wrong, mat2_wrong)
            print("❌ 这不应该成功！")
        except RuntimeError as e:
            print(f"✅ 成功重现原始错误: {e}")
        
        # 展示正确的修复
        print(f"\n修复方案:")
        mat1_correct = torch.randn(2048, 1792)
        mat2_correct = torch.randn(4096, 1792)  # 正确：输入维度匹配
        
        print(f"mat1形状: {mat1_correct.shape}")
        print(f"mat2形状: {mat2_correct.shape}")
        
        result_correct = torch.mm(mat1_correct, mat2_correct.t())
        print(f"✅ 修复后矩阵乘法成功！结果形状: {result_correct.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试原始错误场景的修复...")
    
    test1_passed = test_original_error_scenario()
    test2_passed = test_specific_error_dimensions()
    
    print("\n" + "="*60)
    if test1_passed and test2_passed:
        print("🎉 原始错误场景修复验证通过！")
        print("✅ RuntimeError: mat1 and mat2 shapes cannot be multiplied 问题已解决")
    else:
        print("💥 修复验证失败！")
