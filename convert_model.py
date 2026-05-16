import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import argparse
import os
import json
from copy import deepcopy

def analyze_pruned_state_dict(state_dict, original_config):
    """
    分析剪枝后的状态字典，为每一层确定实际的模型配置
    """
    print("=> 分析剪枝后的模型结构...")
    
    new_config = deepcopy(original_config)
    
    # 获取原始配置信息
    original_hidden_size = original_config.hidden_size
    original_num_attention_heads = original_config.num_attention_heads
    original_num_key_value_heads = getattr(original_config, 'num_key_value_heads', original_num_attention_heads)
    original_intermediate_size = original_config.intermediate_size
    head_dim = original_hidden_size // original_num_attention_heads
    
    print(f"   原始配置: hidden_size={original_hidden_size}, num_attention_heads={original_num_attention_heads}")
    print(f"   原始配置: num_key_value_heads={original_num_key_value_heads}, intermediate_size={original_intermediate_size}")
    
    # 分析每一层的实际结构
    layer_attention_info = {}
    layer_ffn_info = {}
    
    for key, tensor in state_dict.items():
        # 解析层号
        if "layers." in key and ".self_attn." in key:
            # 提取层号
            layer_num = int(key.split("layers.")[1].split(".")[0])
            
            if "q_proj.weight" in key:
                actual_q_dim = tensor.shape[0]
                actual_num_attention_heads = actual_q_dim // head_dim
                if layer_num not in layer_attention_info:
                    layer_attention_info[layer_num] = {}
                layer_attention_info[layer_num]["num_attention_heads"] = actual_num_attention_heads
                
            elif "k_proj.weight" in key:
                actual_kv_dim = tensor.shape[0]
                actual_num_key_value_heads = actual_kv_dim // head_dim
                if layer_num not in layer_attention_info:
                    layer_attention_info[layer_num] = {}
                layer_attention_info[layer_num]["num_key_value_heads"] = actual_num_key_value_heads
                
        elif "layers." in key and ".mlp." in key and "gate_proj.weight" in key:
            # 提取层号和FFN信息
            layer_num = int(key.split("layers.")[1].split(".")[0])
            actual_intermediate_size = tensor.shape[0]
            layer_ffn_info[layer_num] = actual_intermediate_size
    
    # 检查所有层的剪枝情况
    print("=> 每层剪枝情况分析:")
    attention_consistent = True
    ffn_consistent = True
    
    # 检查注意力头是否一致
    if layer_attention_info:
        first_layer_attn = list(layer_attention_info.values())[0]
        ref_num_attention_heads = first_layer_attn.get("num_attention_heads", original_num_attention_heads)
        ref_num_key_value_heads = first_layer_attn.get("num_key_value_heads", original_num_key_value_heads)
        
        for layer_num, info in layer_attention_info.items():
            layer_num_attention_heads = info.get("num_attention_heads", ref_num_attention_heads)
            layer_num_key_value_heads = info.get("num_key_value_heads", ref_num_key_value_heads)
            
            if layer_num_attention_heads != ref_num_attention_heads or layer_num_key_value_heads != ref_num_key_value_heads:
                attention_consistent = False
                print(f"   层 {layer_num}: attention_heads={layer_num_attention_heads}, kv_heads={layer_num_key_value_heads}")
    
    # 检查FFN是否一致
    if layer_ffn_info:
        ref_intermediate_size = list(layer_ffn_info.values())[0]
        for layer_num, intermediate_size in layer_ffn_info.items():
            if intermediate_size != ref_intermediate_size:
                ffn_consistent = False
                print(f"   层 {layer_num}: intermediate_size={intermediate_size}")
    
    if attention_consistent and ffn_consistent:
        print("   所有层剪枝一致，可以创建规整模型")
        # 更新全局配置
        if layer_attention_info:
            first_layer = list(layer_attention_info.values())[0]
            new_config.num_attention_heads = first_layer.get("num_attention_heads", original_num_attention_heads)
            new_config.num_key_value_heads = first_layer.get("num_key_value_heads", original_num_key_value_heads)
            
        if layer_ffn_info:
            new_config.intermediate_size = list(layer_ffn_info.values())[0]
            
        print(f"   新配置: num_attention_heads={new_config.num_attention_heads}, num_key_value_heads={new_config.num_key_value_heads}")
        print(f"   新配置: intermediate_size={new_config.intermediate_size}")
        return new_config, True
    else:
        print("   检测到每层剪枝比例不同，无法创建统一配置的规整模型")
        print("   将使用权重重组方法处理不规整结构")
        return original_config, False

def reorganize_irregular_weights(pruned_state_dict, original_config):
    """
    重组不规整的权重，使其适配原始模型结构
    """
    print("=> 重组不规整权重以适配原始模型结构...")
    
    new_state_dict = {}
    original_intermediate_size = original_config.intermediate_size
    original_hidden_size = original_config.hidden_size
    original_num_attention_heads = original_config.num_attention_heads
    original_num_key_value_heads = getattr(original_config, 'num_key_value_heads', original_num_attention_heads)
    head_dim = original_hidden_size // original_num_attention_heads
    
    print(f"   原始模型配置: hidden_size={original_hidden_size}, intermediate_size={original_intermediate_size}")
    print(f"   原始注意力配置: num_attention_heads={original_num_attention_heads}, num_key_value_heads={original_num_key_value_heads}")
    
    for key, tensor in pruned_state_dict.items():
        # 处理注意力层权重
        if ".self_attn.q_proj.weight" in key:
            # Q投影层: [剪枝后的q_dim, hidden_size] → [原始hidden_size, hidden_size]
            actual_q_dim = tensor.shape[0]
            hidden_size = tensor.shape[1]
            expected_q_dim = original_hidden_size  # Q的输出维度应该等于hidden_size
            
            if actual_q_dim < expected_q_dim:
                new_tensor = torch.zeros(expected_q_dim, hidden_size, dtype=tensor.dtype)
                new_tensor[:actual_q_dim, :] = tensor
                new_state_dict[key] = new_tensor
                print(f"   {key}: 扩展 {tensor.shape} → {new_tensor.shape}")
            else:
                new_state_dict[key] = tensor[:expected_q_dim, :]
                print(f"   {key}: 截断 {tensor.shape} → {new_state_dict[key].shape}")
                
        elif ".self_attn.k_proj.weight" in key or ".self_attn.v_proj.weight" in key:
            # K/V投影层: [剪枝后的kv_dim, hidden_size] → [原始kv_dim, hidden_size]
            actual_kv_dim = tensor.shape[0]
            hidden_size = tensor.shape[1]
            expected_kv_dim = original_num_key_value_heads * head_dim
            
            if actual_kv_dim < expected_kv_dim:
                new_tensor = torch.zeros(expected_kv_dim, hidden_size, dtype=tensor.dtype)
                new_tensor[:actual_kv_dim, :] = tensor
                new_state_dict[key] = new_tensor
                print(f"   {key}: 扩展 {tensor.shape} → {new_tensor.shape}")
            else:
                new_state_dict[key] = tensor[:expected_kv_dim, :]
                print(f"   {key}: 截断 {tensor.shape} → {new_state_dict[key].shape}")
                
        elif ".self_attn.o_proj.weight" in key:
            # O投影层: [hidden_size, 剪枝后的attn_dim] → [hidden_size, 原始hidden_size]
            hidden_size = tensor.shape[0]
            actual_attn_dim = tensor.shape[1]
            expected_attn_dim = original_hidden_size
            
            if actual_attn_dim < expected_attn_dim:
                new_tensor = torch.zeros(hidden_size, expected_attn_dim, dtype=tensor.dtype)
                new_tensor[:, :actual_attn_dim] = tensor
                new_state_dict[key] = new_tensor
                print(f"   {key}: 扩展 {tensor.shape} → {new_tensor.shape}")
            else:
                new_state_dict[key] = tensor[:, :expected_attn_dim]
                print(f"   {key}: 截断 {tensor.shape} → {new_state_dict[key].shape}")
                
        elif ".self_attn.q_proj.bias" in key:
            # Q投影偏置
            actual_q_dim = tensor.shape[0]
            expected_q_dim = original_hidden_size
            
            if actual_q_dim < expected_q_dim:
                new_tensor = torch.zeros(expected_q_dim, dtype=tensor.dtype)
                new_tensor[:actual_q_dim] = tensor
                new_state_dict[key] = new_tensor
                print(f"   {key}: 扩展偏置 {tensor.shape} → {new_tensor.shape}")
            else:
                new_state_dict[key] = tensor[:expected_q_dim]
                print(f"   {key}: 截断偏置 {tensor.shape} → {new_state_dict[key].shape}")
                
        elif ".self_attn.k_proj.bias" in key or ".self_attn.v_proj.bias" in key:
            # K/V投影偏置
            actual_kv_dim = tensor.shape[0]
            expected_kv_dim = original_num_key_value_heads * head_dim
            
            if actual_kv_dim < expected_kv_dim:
                new_tensor = torch.zeros(expected_kv_dim, dtype=tensor.dtype)
                new_tensor[:actual_kv_dim] = tensor
                new_state_dict[key] = new_tensor
                print(f"   {key}: 扩展偏置 {tensor.shape} → {new_tensor.shape}")
            else:
                new_state_dict[key] = tensor[:expected_kv_dim]
                print(f"   {key}: 截断偏置 {tensor.shape} → {new_state_dict[key].shape}")
                
        # 处理FFN层权重
        elif ".mlp.gate_proj.weight" in key or ".mlp.up_proj.weight" in key:
            # FFN输入层权重: [剪枝后的intermediate_size, hidden_size] → [原始intermediate_size, hidden_size]
            actual_intermediate_size = tensor.shape[0]
            hidden_size = tensor.shape[1]
            
            if actual_intermediate_size < original_intermediate_size:
                # 需要扩展：用零填充到原始大小
                new_tensor = torch.zeros(original_intermediate_size, hidden_size, dtype=tensor.dtype)
                new_tensor[:actual_intermediate_size, :] = tensor
                new_state_dict[key] = new_tensor
                print(f"   {key}: 扩展 {tensor.shape} → {new_tensor.shape}")
            else:
                # 大小匹配或需要截断
                new_state_dict[key] = tensor[:original_intermediate_size, :]
                print(f"   {key}: 截断 {tensor.shape} → {new_state_dict[key].shape}")
                
        elif ".mlp.down_proj.weight" in key:
            # FFN输出层权重: [hidden_size, 剪枝后的intermediate_size] → [hidden_size, 原始intermediate_size]
            hidden_size = tensor.shape[0]
            actual_intermediate_size = tensor.shape[1]
            
            if actual_intermediate_size < original_intermediate_size:
                # 需要扩展：用零填充到原始大小
                new_tensor = torch.zeros(hidden_size, original_intermediate_size, dtype=tensor.dtype)
                new_tensor[:, :actual_intermediate_size] = tensor
                new_state_dict[key] = new_tensor
                print(f"   {key}: 扩展 {tensor.shape} → {new_tensor.shape}")
            else:
                # 大小匹配或需要截断
                new_state_dict[key] = tensor[:, :original_intermediate_size]
                print(f"   {key}: 截断 {tensor.shape} → {new_state_dict[key].shape}")
                
        elif ".mlp.gate_proj.bias" in key or ".mlp.up_proj.bias" in key:
            # FFN输入层偏置
            actual_intermediate_size = tensor.shape[0]
            
            if actual_intermediate_size < original_intermediate_size:
                new_tensor = torch.zeros(original_intermediate_size, dtype=tensor.dtype)
                new_tensor[:actual_intermediate_size] = tensor
                new_state_dict[key] = new_tensor
                print(f"   {key}: 扩展偏置 {tensor.shape} → {new_tensor.shape}")
            else:
                new_state_dict[key] = tensor[:original_intermediate_size]
                print(f"   {key}: 截断偏置 {tensor.shape} → {new_state_dict[key].shape}")
                
        else:
            # 其他权重保持不变
            new_state_dict[key] = tensor
    
    return new_state_dict

def convert_pruned_model(base_model_path, checkpoint_path, output_path):
    """
    加载剪枝模型并将其保存为规整的Hugging Face格式。
    """
    print(f"--- 步骤 1: 加载原始模型配置: {base_model_path} ---")
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 先加载原始配置
    original_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    
    print(f"--- 步骤 2: 分析剪枝检查点: {checkpoint_path} ---")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    # 加载剪枝后的状态字典
    pruned_state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # 分析剪枝后的结构
    new_config, is_regular = analyze_pruned_state_dict(pruned_state_dict, original_config)
    
    if is_regular:
        print(f"--- 步骤 3: 使用新配置创建规整模型 ---")
        # 使用新配置创建一个规整的模型
        model = AutoModelForCausalLM.from_config(
            new_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        final_state_dict = pruned_state_dict
    else:
        print(f"--- 步骤 3: 使用原始配置创建模型并重组权重 ---")
        # 使用原始配置创建模型
        model = AutoModelForCausalLM.from_config(
            original_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        # 重组权重以适配原始结构
        final_state_dict = reorganize_irregular_weights(pruned_state_dict, original_config)
    
    print(f"--- 步骤 4: 加载权重到新模型 ---")
    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(final_state_dict, strict=False)
    
    if missing_keys:
        print(f"   警告: 缺少以下键: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
    if unexpected_keys:
        print(f"   警告: 意外的键: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
    
    print("   权重已成功加载到规整模型。")
    
    print(f"--- 步骤 5: 保存为标准Hugging Face格式: {output_path} ---")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 保存模型和配置
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # 保存详细的转换信息
    config_info = {
        "original_model": base_model_path,
        "pruned_checkpoint": checkpoint_path,
        "conversion_method": "regular_config" if is_regular else "weight_reorganization",
        "conversion_info": {
            "original_num_attention_heads": original_config.num_attention_heads,
            "final_num_attention_heads": model.config.num_attention_heads,
            "original_num_key_value_heads": getattr(original_config, 'num_key_value_heads', original_config.num_attention_heads),
            "final_num_key_value_heads": getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads),
            "original_intermediate_size": original_config.intermediate_size,
            "final_intermediate_size": model.config.intermediate_size,
            "is_regular_pruning": is_regular
        }
    }
    
    with open(os.path.join(output_path, "pruning_info.json"), "w") as f:
        json.dump(config_info, f, indent=2)
    
    print("--- 转换完成！---")
    print(f"   模型已保存为标准Hugging Face格式，可直接用于lm-evaluation-harness")
    print(f"   转换方法: {'规整配置' if is_regular else '权重重组'}")
    print(f"   配置信息已保存到: {os.path.join(output_path, 'pruning_info.json')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="转换剪枝模型为Hugging Face格式")
    parser.add_argument('--base_model_path', type=str, required=True, help='原始Hugging Face模型路径')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='剪枝后的.pth.tar文件路径')
    parser.add_argument('--output_path', type=str, required=True, help='转换后模型的保存路径')
    args = parser.parse_args()
    
    convert_pruned_model(args.base_model_path, args.checkpoint_path, args.output_path)
