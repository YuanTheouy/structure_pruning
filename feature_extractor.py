#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FeatureExtractor - LLM 模块功能画像特征提取器

该模块为强化学习剪枝项目提供核心功能：在训练开始前，通过一次离线计算，
提取出 LLM 中每一个可剪枝模块的功能画像特征，输出二维张量作为智能体的静态输入状态。

作者: AI Assistant
创建时间: 2025年8月4日
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm
import numpy as np
import warnings


class FeatureExtractor:
    """
    LLM 模块功能画像特征提取器
    
    该类用于从大语言模型中提取可剪枝模块（MHA和FFN）的功能特征，
    包括激活模式、注意力分布、稀疏性等多维度特征，为后续的强化学习剪枝提供输入状态。
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 dataloader, 
                 prunable_module_names: List[str],
                 max_samples: Optional[int] = 8):  # 进一步减少默认样本数以节省显存
        """
        初始化特征提取器
        
        Args:
            model (nn.Module): 待分析的 PyTorch 模型
            dataloader: 校准数据集的数据加载器
            prunable_module_names (List[str]): 可剪枝模块名称列表
            max_samples (Optional[int]): 最大样本数量，默认为8（极致内存优化）
        """
        self.model = model
        self.dataloader = dataloader
        self.prunable_module_names = prunable_module_names
        
        # 自动检测设备
        self.device = next(model.parameters()).device
        
        # 内存优化：使用累积统计而不是存储原始数据
        # 进一步减少内存占用
        self.running_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self.feature_vectors: Dict[str, torch.Tensor] = {}
        self.handles: List = []
        
        # 特征计算相关参数
        self.sparsity_threshold = 1e-5
        self.max_samples = max_samples  # 使用传入的参数
        self.current_sample_count = 0
        
        # 初始化运行时统计
        self._init_running_stats()
    
    def extract(self) -> torch.Tensor:
        """
        公开的主方法，按顺序执行所有特征提取步骤
        
        Returns:
            torch.Tensor: 形状为 [num_modules, num_features] 的最终状态张量
        """
        print("开始特征提取流程...")
        print(f"=> 内存优化模式：最大样本数 {self.max_samples}")
        
        # 执行特征提取的四个主要步骤
        self._register_hooks()
        self._inference_pass()
        self._calculate_features()
        final_tensor = self._aggregate_and_normalize()
        
        print(f"特征提取完成！最终张量形状: {final_tensor.shape}")
        return final_tensor
    
    def _init_running_stats(self) -> None:
        """
        初始化运行时统计数据结构（内存优化）
        """
        for name in self.prunable_module_names:
            self.running_stats[name] = {
                'count': torch.tensor(0.0),
                'sum': torch.tensor(0.0),
                'sum_squared': torch.tensor(0.0),
                'min_val': torch.tensor(float('inf')),
                'max_val': torch.tensor(float('-inf')),
                'sparsity_count': torch.tensor(0.0),
                'l2_norm_sum': torch.tensor(0.0),
                'positive_count': torch.tensor(0.0),
                'has_attention': False,
                'attn_entropy_sum': torch.tensor(0.0),
                'attn_locality_sum': torch.tensor(0.0),
                'attn_max_sum': torch.tensor(0.0),
                'attn_count': torch.tensor(0.0)
            }
    
    def _register_hooks(self) -> None:
        """
        为所有可剪枝模块注册前向钩子（内存优化版本）
        """
        print(f"正在为 {len(self.prunable_module_names)} 个模块注册钩子...")
        
        for name in self.prunable_module_names:
            # 根据模块名称获取实际的模块对象
            module = self._get_module_by_name(name)
            if module is None:
                warnings.warn(f"无法找到模块: {name}")
                continue
            
            # 定义内存优化的钩子函数
            def create_hook(module_name: str):
                def hook(module: nn.Module, input: Tuple, output: Any) -> None:
                    # 限制样本数量以控制内存
                    if self.current_sample_count >= self.max_samples:
                        return
                    
                    stats = self.running_stats[module_name]
                    
                    # 处理不同类型的输出
                    if isinstance(output, tuple):
                        # MHA 模块通常返回 (hidden_states, attn_weights)
                        hidden_states = output[0].detach()
                        self._update_activation_stats(stats, hidden_states)
                        
                        if len(output) > 1 and output[1] is not None:
                            stats['has_attention'] = True
                            attn_weights = output[1].detach()
                            self._update_attention_stats(stats, attn_weights)
                    else:
                        # FFN 模块通常只返回张量
                        hidden_states = output.detach()
                        self._update_activation_stats(stats, hidden_states)
                
                return hook
            
            # 注册钩子并保存句柄
            handle = module.register_forward_hook(create_hook(name))
            self.handles.append(handle)
        
        print(f"成功注册了 {len(self.handles)} 个钩子")
    
    def _update_activation_stats(self, stats: Dict, activations: torch.Tensor) -> None:
        """
        更新激活统计（内存优化）
        """
        flat_activations = activations.view(-1).float()
        n = flat_activations.numel()
        
        if n == 0:
            return
        
        # 更新基本统计
        stats['count'] += n
        stats['sum'] += flat_activations.sum().cpu()
        stats['sum_squared'] += (flat_activations ** 2).sum().cpu()
        stats['min_val'] = torch.min(stats['min_val'], flat_activations.min().cpu())
        stats['max_val'] = torch.max(stats['max_val'], flat_activations.max().cpu())
        
        # 更新稀疏性统计
        sparse_count = (torch.abs(flat_activations) < self.sparsity_threshold).sum().cpu()
        stats['sparsity_count'] += sparse_count
        
        # 更新L2范数
        l2_norm = torch.norm(flat_activations, p=2).cpu()
        stats['l2_norm_sum'] += l2_norm
        
        # 更新正激活统计
        positive_count = (flat_activations > 0).sum().cpu()
        stats['positive_count'] += positive_count
    
    def _update_attention_stats(self, stats: Dict, attention_weights: torch.Tensor) -> None:
        """
        更新注意力统计（内存优化）
        """
        if attention_weights.dim() != 4:
            return
        
        # 计算注意力熵
        attention_weights = torch.clamp(attention_weights, min=1e-12)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
        log_weights = torch.log(attention_weights + 1e-12)
        entropy = -(attention_weights * log_weights).sum(dim=-1)
        valid_entropy = entropy[~torch.isnan(entropy) & ~torch.isinf(entropy)]
        
        if len(valid_entropy) > 0:
            stats['attn_entropy_sum'] += valid_entropy.mean().cpu()
            stats['attn_count'] += 1
        
        # 计算局部性
        seq_len = attention_weights.size(-1)
        if seq_len > 1:
            diagonal_mask = torch.zeros(seq_len, seq_len, device=attention_weights.device)
            for i in range(seq_len):
                for j in range(max(0, i-2), min(seq_len, i+3)):
                    diagonal_mask[i, j] = 1.0
            
            diagonal_attention = (attention_weights * diagonal_mask).sum(dim=(-2, -1))
            total_attention = attention_weights.sum(dim=(-2, -1))
            locality_score = (diagonal_attention / (total_attention + 1e-12)).mean().cpu()
            stats['attn_locality_sum'] += locality_score
        
        # 计算最大注意力
        max_attention = attention_weights.max(dim=-1)[0].mean().cpu()
        stats['attn_max_sum'] += max_attention
    
    def _get_module_by_name(self, name: str) -> Optional[nn.Module]:
        """
        根据模块名称获取模块对象
        
        Args:
            name (str): 模块的完整名称路径
            
        Returns:
            Optional[nn.Module]: 找到的模块对象，如果未找到则返回 None
        """
        try:
            parts = name.split('.')
            module = self.model
            for part in parts:
                module = getattr(module, part)
            return module
        except AttributeError:
            return None
    
    def _inference_pass(self) -> None:
        """
        执行推理过程以收集激活数据
        """
        print("开始前向推理以收集激活数据...")
        
        # 从模型配置中获取词汇表大小
        vocab_size = getattr(self.model.config, 'vocab_size', 50272)  # OPT-1.3B默认词汇表大小
        print(f"=> 模型词汇表大小: {vocab_size}")
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(self.dataloader, desc="推理进度")):
                if self.current_sample_count >= self.max_samples:
                    break
                
                # 处理数据加载器的输出格式: (input_ids, target)
                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                    input_ids, target = batch_data[0], batch_data[1]
                    # 只使用input_ids进行推理
                    batch = input_ids
                elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 1:
                    batch = batch_data[0]
                else:
                    batch = batch_data
                
                # 验证token ID范围
                if isinstance(batch, torch.Tensor):
                    max_token_id = batch.max().item()
                    min_token_id = batch.min().item()
                    
                    if max_token_id >= vocab_size or min_token_id < 0:
                        print(f"=> 警告: 批次 {batch_idx} 包含无效token ID")
                        print(f"   Token ID范围: [{min_token_id}, {max_token_id}], 词汇表大小: {vocab_size}")
                        
                        # 清理无效的token ID
                        batch = torch.clamp(batch, 0, vocab_size - 1)
                
                # 将数据移动到正确的设备
                if isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device)
                
                try:
                    # 执行前向传播，确保输出注意力权重
                    kwargs = {
                        'input_ids': batch,
                        'output_attentions': True
                    }
                    
                    # 对于OPT模型，可能需要attention_mask
                    if hasattr(self.model.config, 'model_type') and 'opt' in self.model.config.model_type.lower():
                        # 创建attention mask (1表示非padding位置)
                        attention_mask = torch.ones_like(batch)
                        kwargs['attention_mask'] = attention_mask
                    
                    _ = self.model(**kwargs)
                    
                    self.current_sample_count += 1
                    
                    # 频繁清理显存以减少峰值占用
                    if batch_idx % 2 == 0:  # 每2个batch清理一次
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    warnings.warn(f"批次 {batch_idx} 推理失败: {e}")
                    continue
                
                # 进一步减少内存占用
                del batch
                if 'attention_mask' in locals():
                    del attention_mask
                
                # 每处理4个样本后强制清理一次
                if self.current_sample_count % 4 == 0:
                    torch.cuda.empty_cache()
        
        print(f"推理完成，共收集 {self.current_sample_count} 个样本的数据")
    
    def _calculate_features(self) -> None:
        """
        根据累积的统计数据计算每个模块的特征向量（内存优化）
        """
        print("正在计算模块特征...")
        
        for name in tqdm(self.prunable_module_names, desc="特征计算"):
            stats = self.running_stats[name]
            
            # 检查是否有数据
            if stats['count'] == 0:
                warnings.warn(f"模块 {name} 没有收集到数据")
                self.feature_vectors[name] = torch.zeros(8, dtype=torch.float32)
                continue
            
            features = []
            
            # 计算基本激活特征
            count = stats['count'].item()
            mean_val = (stats['sum'] / count).item()
            var_val = (stats['sum_squared'] / count - mean_val ** 2).item()
            std_val = max(0.0, var_val) ** 0.5
            
            # 1. L2 范数
            l2_norm = (stats['l2_norm_sum'] / self.current_sample_count).item()
            features.append(l2_norm)
            
            # 2. 稀疏度
            sparsity = (stats['sparsity_count'] / count).item()
            features.append(sparsity)
            
            # 3. 激活的均值
            features.append(mean_val)
            
            # 4. 激活的标准差
            features.append(std_val)
            
            # 计算特定模块特征
            if stats['has_attention'] and stats['attn_count'] > 0:
                # 注意力模块特征
                attn_count = stats['attn_count'].item()
                
                # 5. 注意力熵
                entropy = (stats['attn_entropy_sum'] / attn_count).item()
                features.append(entropy)
                
                # 6. 局部性得分
                locality = (stats['attn_locality_sum'] / attn_count).item()
                features.append(locality)
                
                # 7. 注意力集中度
                max_attn = (stats['attn_max_sum'] / attn_count).item()
                features.append(max_attn)
                
                # 8. 分隔符关注得分（占位）
                features.append(0.0)
            else:
                # FFN 模块特征
                min_val = stats['min_val'].item()
                max_val = stats['max_val'].item()
                
                # 5. 动态范围
                dynamic_range = max_val - min_val
                features.append(dynamic_range)
                
                # 6. 正激活比例
                positive_ratio = (stats['positive_count'] / count).item()
                features.append(positive_ratio)
                
                # 7. GLU 门控均值（占位）
                features.append(0.0)
                
                # 8. 峰度（简化计算）
                # 使用方差作为峰度的近似
                kurtosis = var_val / (std_val ** 4 + 1e-12) - 3.0 if std_val > 1e-6 else 0.0
                features.append(kurtosis)
            
            # 清理NaN值
            features = [0.0 if np.isnan(f) or np.isinf(f) else f for f in features]
            
            # 存储特征向量
            self.feature_vectors[name] = torch.tensor(features, dtype=torch.float32)
        
        # 清理钩子
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        
        print(f"特征计算完成，共计算了 {len(self.feature_vectors)} 个模块的特征")
    
    def _aggregate_and_normalize(self) -> torch.Tensor:
        """
        聚合所有模块的特征，进行归一化，并整合成最终的张量
        
        Returns:
            torch.Tensor: 标准化后的状态张量 [num_modules, num_features]
        """
        print("正在聚合和标准化特征...")
        
        # 按照模块名称顺序收集特征向量
        feature_list = []
        for name in self.prunable_module_names:
            if name in self.feature_vectors:
                feature_vec = self.feature_vectors[name]
                # 检查并清理NaN值
                if torch.isnan(feature_vec).any():
                    print(f"  ⚠️  模块 {name} 的特征向量包含NaN，替换为0")
                    feature_vec = torch.nan_to_num(feature_vec, nan=0.0)
                feature_list.append(feature_vec)
            else:
                # 如果某个模块没有特征，使用零向量
                warnings.warn(f"模块 {name} 没有特征向量，使用零向量")
                feature_list.append(torch.zeros(8, dtype=torch.float32))
        
        if not feature_list:
            raise ValueError("没有提取到任何特征向量")
        
        # 堆叠成二维张量
        state_tensor = torch.stack(feature_list, dim=0)
        
        # 清理NaN和Inf值
        state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 标准化处理（在特征维度上）
        feature_means = state_tensor.mean(dim=0, keepdim=True)
        feature_stds = state_tensor.std(dim=0, keepdim=True)
        
        # 避免除零
        feature_stds = torch.where(feature_stds < 1e-8, 
                                  torch.ones_like(feature_stds), 
                                  feature_stds)
        
        # 标准化
        normalized_tensor = (state_tensor - feature_means) / feature_stds
        
        # 最终清理
        normalized_tensor = torch.nan_to_num(normalized_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        
        print(f"特征聚合完成，最终张量形状: {normalized_tensor.shape}")
        print(f"特征统计: 均值={normalized_tensor.mean().item():.4f}, "
              f"标准差={normalized_tensor.std().item():.4f}")
        
        return normalized_tensor


if __name__ == '__main__':
    """
    使用示例和测试代码
    """
    print("FeatureExtractor 演示代码")
    print("=" * 50)
    
    # 这是一个演示如何使用该类的完整示例
    # 在实际使用中，请取消注释并根据具体需求调整
    
    # 1. 加载预训练模型和分词器
    print("步骤 1: 加载模型")
    print("示例代码:")
    print("""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "facebook/opt-1.3b"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    """)
    
    # 2. 准备校准数据集
    print("\n步骤 2: 准备校准数据集")
    print("示例代码:")
    print("""
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    
    # 加载数据集
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # 数据预处理
    def preprocess_function(examples):
        return tokenizer(examples["text"], 
                        truncation=True, 
                        padding=True, 
                        max_length=512, 
                        return_tensors="pt")
    
    # 创建数据加载器
    processed_dataset = dataset.map(preprocess_function, batched=True)
    dataloader = DataLoader(processed_dataset, batch_size=4, shuffle=False)
    """)
    
    # 3. 定义可剪枝模块名称
    print("\n步骤 3: 定义可剪枝模块")
    print("示例代码:")
    print("""
    # 获取 OPT-1.3B 模型的可剪枝模块名称
    prunable_module_names = []
    
    # 假设模型有 24 层
    num_layers = 24  # model.config.num_hidden_layers
    
    for i in range(num_layers):
        # 多头注意力模块
        prunable_module_names.append(f"model.decoder.layers.{i}.self_attn")
        # FFN 模块
        prunable_module_names.append(f"model.decoder.layers.{i}.fc1")
        prunable_module_names.append(f"model.decoder.layers.{i}.fc2")
    
    print(f"共识别出 {len(prunable_module_names)} 个可剪枝模块")
    """)
    
    # 4. 运行特征提取
    print("\n步骤 4: 运行特征提取")
    print("示例代码:")
    print("""
    # 实例化特征提取器
    print("开始特征提取...")
    extractor = FeatureExtractor(model, dataloader, prunable_module_names)
    
    # 执行特征提取
    state_tensor = extractor.extract()
    
    print(f"特征提取完成！")
    print(f"最终状态张量形状: {state_tensor.shape}")
    print(f"张量统计信息:")
    print(f"  均值: {state_tensor.mean().item():.6f}")
    print(f"  标准差: {state_tensor.std().item():.6f}")
    print(f"  最小值: {state_tensor.min().item():.6f}")
    print(f"  最大值: {state_tensor.max().item():.6f}")
    """)
    
    # 5. 保存结果
    print("\n步骤 5: 保存特征张量")
    print("示例代码:")
    print("""
    # 保存特征张量
    output_path = 'functional_state.pt'
    torch.save({
        'state_tensor': state_tensor,
        'module_names': prunable_module_names,
        'extraction_metadata': {
            'num_modules': len(prunable_module_names),
            'num_features': state_tensor.shape[1],
            'extraction_time': datetime.now().isoformat()
        }
    }, output_path)
    
    print(f"特征张量已保存到: {output_path}")
    """)
    
    # 使用建议
    print("\n" + "=" * 50)
    print("使用建议:")
    print("1. 根据具体模型调整 prunable_module_names")
    print("2. 根据硬件情况调整 max_samples 参数")
    print("3. 可以通过修改 sparsity_threshold 调整稀疏度计算阈值")
    print("4. 建议在使用前先用小数据集测试")
    print("5. 确保模型和数据在同一设备上")
    
    print("\n特征提取器类定义完成！")
