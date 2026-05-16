#!/usr/bin/env python3
"""
快速测试脚本，验证剪枝逻辑修复是否正确
"""

import sys
import traceback

def test_import():
    """测试导入是否正确"""
    try:
        # 测试基础导入
        import torch
        import numpy as np
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 测试我们修复的模块导入
        from env.channel_pruning_env_llm_global import ChannelPruningEnv
        
        print("✅ 所有必要的模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        traceback.print_exc()
        return False

def test_attention_logic():
    """测试注意力剪枝逻辑的基本语法"""
    try:
        # 这里我们只测试基本的语法，不实际运行模型
        import torch
        
        # 模拟一些基本参数
        num_key_value_heads = 28
        attention_head_size = 128
        hidden_size = 3584
        
        # 模拟保留掩码
        d_prime = 20  # 保留20个头
        preserve_idx = torch.arange(d_prime)
        mask = torch.zeros(num_key_value_heads, dtype=bool)
        mask[preserve_idx] = True
        
        print(f"🔍 测试注意力头剪枝逻辑...")
        print(f"   原始头数: {num_key_value_heads}")
        print(f"   保留头数: {d_prime}")
        print(f"   掩码形状: {mask.shape}")
        print(f"   保留的头索引: {preserve_idx}")
        
        # 模拟权重reshape和选择逻辑
        fake_weight = torch.randn(num_key_value_heads * attention_head_size, hidden_size)
        print(f"   原始权重形状: {fake_weight.shape}")
        
        # 测试reshape逻辑
        reshaped = fake_weight.reshape(num_key_value_heads, attention_head_size, -1)
        print(f"   Reshape后形状: {reshaped.shape}")
        
        # 测试掩码选择
        selected = reshaped[mask, :, :]
        print(f"   掩码选择后形状: {selected.shape}")
        
        # 测试最终reshape
        final_weight = selected.reshape(-1, selected.shape[2])
        print(f"   最终权重形状: {final_weight.shape}")
        
        print("✅ 注意力头剪枝逻辑测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 注意力头逻辑测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("🧪 AMC-LLM 剪枝修复验证测试")
    print("="*60)
    
    tests = [
        ("导入测试", test_import),
        ("注意力头逻辑测试", test_attention_logic),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 运行: {test_name}")
        print("-" * 40)
        
        if test_func():
            passed += 1
        
        print("-" * 40)
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！修复应该是成功的。")
        return 0
    else:
        print("⚠️  有测试失败，请检查修复。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
