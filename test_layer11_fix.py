#!/usr/bin/env python3
"""
直接测试Layer 11 Head剪枝的维度匹配问题是否修复
不需要完整的PPO训练，只测试剪枝过程
"""

import os
import torch
import argparse
from env.channel_pruning_env_llm_global import ChannelPruningEnv

def test_layer11_pruning():
    """直接测试Layer 11 Head剪枝是否会出现维度不匹配错误"""
    
    print("=== 直接测试Layer 11 Head剪枝修复 ===")
    print(f"使用GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}")
    
    # 设置最小参数
    class Args:
        model = "Qwen25-7B"
        cache_dir = "llm_weights"
        dataset_name = "wikitext"
        preserve_ratio = 0.8
        n_samples = 4  # 最小样本数
        seed = 42
        prune = "flops"
        recon = True  # 开启重构，这是出错的地方
        recon_sample = 4
        channel_round = 8
        acc_metric = "acc5"
        use_real_val = False
        reward = "reward_ppl"
        lbound = 0.2
        rbound = 1.0
        use_new_input = False
        use_dataset_growth = False
    
    args = Args()
    
    try:
        print("=> 初始化环境...")
        env = ChannelPruningEnv(
            model=None,  # 会从args.model加载
            data=args.dataset_name,
            preserve_ratio=args.preserve_ratio,
            args=args,
            export_model=False
        )
        
        print("=> 环境初始化成功！")
        print(f"=> 模型层数: {env.num_hidden_layers}")
        print(f"=> 注意力头数: {env.num_attention_heads}")
        print(f"=> KV头数: {env.num_key_value_heads}")
        
        # 直接测试Layer 11 Head剪枝
        target_layer = 11
        is_head = True  # 测试attention head
        preserve_ratio = 0.8
        
        print(f"\n=> 直接测试Layer {target_layer} Head剪枝...")
        print(f"=> 剪枝比例: {preserve_ratio}")
        
        # 设置一些必要的状态
        env.layer_idx = target_layer
        env.head = is_head
        
        # 准备校准数据
        print("=> 准备校准数据...")
        with torch.no_grad():
            env.inps, env.outs, env.attention_mask, env.position_ids = env.prepare_calibration_input()
        
        # 设置重构数据
        if env.recon:
            idx = torch.randperm(env.n_samples)[:env.args.recon_sample] 
            env.recon_inps = env.inps[idx]
            env.recon_outs = env.outs[idx]
        
        # 执行剪枝 - 这里是关键测试点
        print(f"=> 执行Layer {target_layer} Head剪枝...")
        print(f"=> [DEBUG] 执行前的A_metric[{target_layer * 2}] shape: {env.A_metric[target_layer * 2].shape}")
        
        ratio, d_prime = env.prune(preserve_ratio, target_layer, is_head, target_layer * 2)
        
        print(f"✅ 成功！Layer {target_layer} Head剪枝完成")
        print(f"=> 实际剪枝比例: {ratio:.4f}")
        print(f"=> 保留维度: {d_prime}")
        
        # 验证剪枝后的模型状态
        layer = env.model.model.layers[target_layer]
        print(f"=> [DEBUG] 剪枝后验证:")
        print(f"   q_proj.weight.shape: {layer.self_attn.q_proj.weight.shape}")
        print(f"   k_proj.weight.shape: {layer.self_attn.k_proj.weight.shape}") 
        print(f"   v_proj.weight.shape: {layer.self_attn.v_proj.weight.shape}")
        print(f"   o_proj.weight.shape: {layer.self_attn.o_proj.weight.shape}")
        print(f"   num_heads: {layer.self_attn.num_heads}")
        print(f"   num_key_value_heads: {layer.self_attn.num_key_value_heads}")
        
        # 测试下一层FFN也不会有问题
        print(f"\n=> 继续测试Layer {target_layer} FFN剪枝...")
        env.head = False  # 切换到FFN
        ratio, d_prime = env.prune(preserve_ratio, target_layer, False, target_layer * 2 + 1)
        
        print(f"✅ 成功！Layer {target_layer} FFN剪枝也完成")
        print(f"=> 实际剪枝比例: {ratio:.4f}")
        print(f"=> 保留维度: {d_prime}")
        
        print("\n🎉 所有测试通过！维度匹配问题已修复！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 使用双GPU测试
    success = test_layer11_pruning()
    exit(0 if success else 1)
