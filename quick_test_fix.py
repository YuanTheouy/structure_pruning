#!/usr/bin/env python3
"""
快速测试修复后的代码，专门检查Layer 11 Head的维度匹配问题
"""
import torch
import os
import sys
sys.path.append('.')

from transformers import AutoTokenizer, AutoModelForCausalLM
from env.channel_pruning_env_llm_global import ChannelPruningEnv
from lib.data_utils import get_loaders

class QuickTestArgs:
    """模拟args参数"""
    def __init__(self):
        self.model = "Qwen/Qwen2.5-7B"  # 使用问题模型
        self.dataset = "c4"
        self.cache_dir = "./cache"
        self.seed = 0
        self.recon_sample = 5  # 极小的重构样本数
        self.reward = "lambda ppl: -ppl"  # 简单奖励函数
        self.prune = "flops"
        self.resume_path = None
        self.use_dataset_growth = False
        self.delayed_downstream_eval = True  # 跳过下游任务
        
def quick_test():
    print("🚀 开始快速测试 - 专门检查Layer 11 Head维度匹配问题")
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ 需要CUDA支持")
        return
    
    print(f"✅ CUDA可用，设备数量: {torch.cuda.device_count()}")
    
    try:
        # 1. 创建模拟环境
        args = QuickTestArgs()
        print(f"📝 使用模型: {args.model}")
        
        # 2. 创建环境实例 - 只测试到Layer 11
        print("🔧 创建剪枝环境...")
        env = ChannelPruningEnv(
            model=None,  # 将在环境内部加载
            data=None,   # 将在环境内部加载  
            preserve_ratio=0.8,  # 保持80%
            args=args,
            n_data_worker=1,
            batch_size=1,  # 最小批次
            export_model=False,
            use_new_input=False
        )
        
        print("✅ 环境创建成功")
        print(f"📊 模型层数: {env.num_hidden_layers}")
        print(f"🔢 注意力头数: {env.num_key_value_heads}")
        print(f"📏 隐藏维度: {env.hidden_size}")
        
        # 3. 模拟剪枝到Layer 11 Head - 这是出错的地方
        print("\n🎯 开始测试Layer 11 Head剪枝...")
        
        # 设置环境状态到Layer 11 Head
        env.layer_idx = 11
        env.head = True
        
        # 执行剪枝 - 这里应该会触发之前的错误
        try:
            preserve_ratio = 0.7  # 剪枝到70%
            ratio, d_prime = env.prune(preserve_ratio, env.layer_idx, env.head, 2*env.layer_idx + int(not env.head))
            print(f"✅ Layer 11 Head剪枝成功！")
            print(f"   - 实际比例: {ratio:.4f}")
            print(f"   - 剩余头数: {d_prime}")
            
            # 4. 测试后续层的前向传播 - 这是关键测试
            print("\n🔍 测试剪枝后的前向传播...")
            layer = env.get_layers(env.model)[11]
            
            # 使用一个小的测试输入
            test_input = torch.randn(1, 10, layer.self_attn.q_proj.in_features, 
                                   device=env.device, dtype=torch.bfloat16)
            
            with torch.no_grad():
                output = layer(test_input, 
                             attention_mask=torch.ones(1, 1, 10, 10, device=env.device),
                             position_ids=torch.arange(10, device=env.device).unsqueeze(0))
            
            print(f"✅ 前向传播成功！输出形状: {output[0].shape}")
            print("🎉 修复验证成功 - Layer 11 Head维度匹配问题已解决！")
            
        except Exception as e:
            print(f"❌ Layer 11 Head剪枝失败: {e}")
            print("🔧 错误详情:")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🏁 快速测试完成")
    return True

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n✅ 测试通过 - 可以进行完整的剪枝实验了！")
    else:
        print("\n❌ 测试失败 - 需要进一步修复")
