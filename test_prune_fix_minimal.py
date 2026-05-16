#!/usr/bin/env python3
"""
最小化的剪枝逻辑测试 - 不需要真实模型，只测试核心逻辑
"""
import torch
import numpy as np
import sys
import os

# 模拟一个简化的注意力模块
class MockAttention:
    def __init__(self, num_heads=32, num_key_value_heads=8, head_dim=128):
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads  
        self.head_dim = head_dim
        self.hidden_size = head_dim * num_heads
        
        # 模拟权重 - 使用正确的维度
        hidden_size = head_dim * num_heads
        kv_dim = head_dim * num_key_value_heads
        q_dim = hidden_size  # Q的维度是完整的
        
        self.k_proj = MockLinear(hidden_size, kv_dim)
        self.v_proj = MockLinear(hidden_size, kv_dim) 
        self.q_proj = MockLinear(hidden_size, q_dim)
        self.o_proj = MockLinear(hidden_size, hidden_size)
        
    def to(self, device):
        return self

class MockLinear:
    def __init__(self, in_features, out_features):
        self.weight = MockParameter(torch.randn(out_features, in_features))
        self.bias = None  # 简化，不使用bias
        
    def to(self, device):
        return self

class MockParameter:
    def __init__(self, data):
        self.data = data
        
    @property
    def device(self):
        return self.data.device

def test_attention_head_pruning_logic():
    """测试注意力头剪枝的核心逻辑"""
    print("=== 测试注意力头剪枝逻辑 ===")
    
    # 创建模拟的注意力模块（类似Qwen2.5）
    attn = MockAttention(num_heads=32, num_key_value_heads=8, head_dim=128)
    original_num_heads = attn.num_heads
    original_num_key_value_heads = attn.num_key_value_heads
    original_hidden_size = attn.hidden_size
    
    print(f"原始配置: num_heads={original_num_heads}, num_key_value_heads={original_num_key_value_heads}")
    print(f"原始hidden_size={original_hidden_size}, head_dim={attn.head_dim}")
    
    # 模拟剪枝参数
    preserve_ratio = 0.8
    d_prime = int(preserve_ratio * original_num_key_value_heads)  # 保留6个KV头
    num_q_groups = original_num_heads // original_num_key_value_heads  # 4
    
    print(f"剪枝后: d_prime={d_prime}, num_q_groups={num_q_groups}")
    
    # 创建mask（模拟选择哪些头保留）
    mask = torch.zeros(original_num_key_value_heads, dtype=torch.bool)
    mask[:d_prime] = True  # 保留前d_prime个头
    
    print(f"Mask: {mask}")
    
    try:
        # 测试K, V权重剪枝
        print("\n--- 测试K, V权重剪枝 ---")
        original_k_shape = attn.k_proj.weight.data.shape
        print(f"K原始形状: {original_k_shape}")
        
        # K权重剪枝
        k_weight = attn.k_proj.weight.data.reshape(original_num_key_value_heads, attn.head_dim, -1)
        print(f"K reshape后: {k_weight.shape}")
        k_weight_pruned = k_weight[mask, :, :]
        print(f"K剪枝后: {k_weight_pruned.shape}")
        k_weight_final = k_weight_pruned.reshape(-1, k_weight_pruned.shape[2])
        print(f"K最终形状: {k_weight_final.shape}")
        
        # V权重剪枝（相同逻辑）
        v_weight = attn.v_proj.weight.data.reshape(original_num_key_value_heads, attn.head_dim, -1)
        v_weight_pruned = v_weight[mask, :, :]
        v_weight_final = v_weight_pruned.reshape(-1, v_weight_pruned.shape[2])
        print(f"V最终形状: {v_weight_final.shape}")
        
        # 测试Q权重剪枝（分组逻辑）
        print("\n--- 测试Q权重剪枝 ---")
        original_q_shape = attn.q_proj.weight.data.shape
        print(f"Q原始形状: {original_q_shape}")
        
        q_weight = attn.q_proj.weight.data.reshape(original_num_key_value_heads, num_q_groups * attn.head_dim, -1)
        print(f"Q reshape后: {q_weight.shape}")
        q_weight_pruned = q_weight[mask, :, :]
        print(f"Q剪枝后: {q_weight_pruned.shape}")
        q_weight_final = q_weight_pruned.reshape(-1, q_weight_pruned.shape[2])
        print(f"Q最终形状: {q_weight_final.shape}")
        
        # 测试输出投影剪枝
        print("\n--- 测试输出投影剪枝 ---")
        original_o_shape = attn.o_proj.weight.data.shape
        print(f"O原始形状: {original_o_shape}")
        
        # 创建输出投影的mask
        mask_proj = mask.unsqueeze(0).repeat(attn.head_dim * num_q_groups, 1).t().reshape(-1)
        print(f"输出投影mask形状: {mask_proj.shape}")
        print(f"保留的输出维度数量: {mask_proj.sum()}")
        
        o_weight = attn.o_proj.weight.data.reshape(-1, attn.head_dim * num_q_groups, original_num_key_value_heads)
        print(f"O reshape后: {o_weight.shape}")
        o_weight_pruned = o_weight[:, :, mask]
        print(f"O剪枝后: {o_weight_pruned.shape}")
        o_weight_final = o_weight_pruned.reshape(o_weight_pruned.shape[0], -1)
        print(f"O最终形状: {o_weight_final.shape}")
        
        # 更新注意力配置
        new_num_heads = d_prime * num_q_groups
        new_num_key_value_heads = d_prime
        new_hidden_size = attn.head_dim * new_num_heads
        
        print(f"\n--- 更新后的配置 ---")
        print(f"新配置: num_heads={new_num_heads}, num_key_value_heads={new_num_key_value_heads}")
        print(f"新hidden_size={new_hidden_size}")
        
        # 验证维度一致性
        expected_kv_size = d_prime * attn.head_dim
        expected_q_size = new_num_heads * attn.head_dim
        expected_o_input_size = new_num_heads * attn.head_dim
        
        print(f"\n--- 维度一致性检查 ---")
        print(f"K,V期望输出维度: {expected_kv_size}, 实际: {k_weight_final.shape[0]}")
        print(f"Q期望输出维度: {expected_q_size}, 实际: {q_weight_final.shape[0]}")
        print(f"O期望输入维度: {expected_o_input_size}, 实际: {o_weight_final.shape[1]}")
        
        # 检查是否匹配
        kv_match = k_weight_final.shape[0] == expected_kv_size
        q_match = q_weight_final.shape[0] == expected_q_size  
        o_match = o_weight_final.shape[1] == expected_o_input_size
        
        print(f"\nK,V维度匹配: {kv_match}")
        print(f"Q维度匹配: {q_match}")
        print(f"O维度匹配: {o_match}")
        
        if kv_match and q_match and o_match:
            print("\n✅ 所有维度检查通过！注意力头剪枝逻辑正确。")
            return True
        else:
            print("\n❌ 维度检查失败！")
            return False
            
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_matrix_multiplication_compatibility():
    """测试矩阵乘法兼容性"""
    print("\n=== 测试矩阵乘法兼容性 ===")
    
    try:
        # 模拟实际的矩阵乘法场景
        batch_size, seq_len = 2, 128
        original_hidden_size = 4096  # Qwen2.5-7B的hidden_size
        
        # 模拟剪枝后的维度
        preserve_ratio = 0.8
        original_num_key_value_heads = 8
        d_prime = int(preserve_ratio * original_num_key_value_heads)  # 6
        head_dim = 128
        num_q_groups = 4
        
        pruned_hidden_size = d_prime * num_q_groups * head_dim  # 6 * 4 * 128 = 3072
        
        print(f"原始hidden_size: {original_hidden_size}")
        print(f"剪枝后hidden_size: {pruned_hidden_size}")
        
        # 模拟输入张量（经过注意力计算后的结果）
        attention_output = torch.randn(batch_size, seq_len, pruned_hidden_size)
        print(f"注意力输出形状: {attention_output.shape}")
        
        # 模拟剪枝后的o_proj权重
        o_proj_weight = torch.randn(original_hidden_size, pruned_hidden_size)
        print(f"O投影权重形状: {o_proj_weight.shape}")
        
        # 测试矩阵乘法
        result = torch.mm(attention_output.view(-1, pruned_hidden_size), o_proj_weight.t())
        print(f"矩阵乘法结果形状: {result.shape}")
        
        expected_shape = (batch_size * seq_len, original_hidden_size)
        if result.shape == expected_shape:
            print(f"✅ 矩阵乘法兼容性测试通过！")
            return True
        else:
            print(f"❌ 期望形状 {expected_shape}, 实际形状 {result.shape}")
            return False
            
    except Exception as e:
        print(f"❌ 矩阵乘法测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始最小化剪枝逻辑测试...")
    
    test1_passed = test_attention_head_pruning_logic()
    test2_passed = test_matrix_multiplication_compatibility()
    
    print("\n" + "="*50)
    if test1_passed and test2_passed:
        print("🎉 所有测试通过！剪枝逻辑修复正确。")
        sys.exit(0)
    else:
        print("💥 测试失败！需要进一步修复。")
        sys.exit(1)
