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
import os
import hashlib
import json
from datetime import datetime


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
                 max_samples: Optional[int] = 8,
                 cache_dir: str = "./feature_cache"):  # 添加缓存目录参数
        """
        初始化特征提取器
        
        Args:
            model (nn.Module): 待分析的 PyTorch 模型
            dataloader: 校准数据集的数据加载器
            prunable_module_names (List[str]): 可剪枝模块名称列表
            max_samples (Optional[int]): 最大样本数量，默认为8（极致内存优化）
            cache_dir (str): 特征缓存目录
        """
        print(f"=> 初始化FeatureExtractor，最大样本数: {max_samples}")
        
        # 显示初始化前的内存状态
        if torch.cuda.is_available():
            init_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"=> 初始化前GPU内存: {init_memory:.2f} GB")
        
        self.model = model
        self.dataloader = dataloader
        self.prunable_module_names = prunable_module_names
        self.cache_dir = cache_dir
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 自动检测设备
        self.device = next(model.parameters()).device
        print(f"=> 模型设备: {self.device}")
        
        # 内存优化：使用累积统计而不是存储原始数据
        # 进一步减少内存占用
        self.running_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self.feature_vectors: Dict[str, torch.Tensor] = {}
        self.handles: List = []
        
        # 特征计算相关参数
        self.sparsity_threshold = 1e-5
        self.max_samples = max_samples  # 使用传入的参数
        self.current_sample_count = 0
        
        # 生成缓存键和文件路径
        self.cache_key = self._generate_cache_key()
        self.cache_file = os.path.join(cache_dir, f"features_{self.cache_key}.pt")
        print(f"=> 缓存文件: {self.cache_file}")
        
        # 初始化运行时统计
        self._init_running_stats()
        
        # 显示初始化后的内存状态
        if torch.cuda.is_available():
            post_init_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"=> 初始化后GPU内存: {post_init_memory:.2f} GB")
            print(f"=> 初始化内存增量: {post_init_memory - init_memory:.2f} GB")
    
    def extract(self) -> torch.Tensor:
        """
        公开的主方法，使用分层特征提取避免内存爆炸，支持缓存
        
        Returns:
            torch.Tensor: 形状为 [num_modules, num_features] 的最终状态张量
        """
        print("开始分层特征提取流程...")
        print(f"=> 内存优化模式：最大样本数 {self.max_samples}")
        print(f"=> 分层处理模式：逐层提取特征以避免内存爆炸")
        
        # 首先尝试从缓存加载
        cached_features = self._load_features_from_cache()
        if cached_features is not None:
            print("=> 成功从缓存加载特征，跳过计算！")
            return cached_features
        
        # 缓存不存在或无效，进行特征提取
        print("=> 开始重新计算特征...")
        final_tensor = self._layered_extract()
        
        # 保存到缓存
        self._save_features_to_cache(final_tensor)
        
        print(f"分层特征提取完成！最终张量形状: {final_tensor.shape}")
        return final_tensor
    
    def _layered_extract(self) -> torch.Tensor:
        """
        分层特征提取，每次只处理一层的模块，大幅降低内存使用
        
        Returns:
            torch.Tensor: 形状为 [num_modules, num_features] 的最终状态张量
        """
        print("开始分层特征提取...")
        
        # 显示初始内存状态
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"=> 初始GPU内存使用: {initial_memory:.2f} GB")
            # 强制清理一次缓存
            torch.cuda.empty_cache()
            print(f"=> 清理缓存后GPU内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # 按层分组模块
        layer_groups = self._group_modules_by_layer()
        print(f"=> 共分为 {len(layer_groups)} 个层组进行处理")
        print(f"=> 每层模块数: {[len(group) for group in layer_groups]}")
        
        all_features = []
        
        for layer_idx, layer_modules in enumerate(layer_groups):
            print(f"=> 处理第 {layer_idx + 1}/{len(layer_groups)} 层，包含 {len(layer_modules)} 个模块")
            print(f"   模块列表: {layer_modules}")
            
            # 显示当前内存状态
            if torch.cuda.is_available():
                pre_layer_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"   层处理前GPU内存: {pre_layer_memory:.2f} GB")
            
            # 为当前层模块初始化统计数据
            current_stats = {}
            for module_name in layer_modules:
                current_stats[module_name] = {
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
            
            # 注册当前层的钩子
            current_handles = []
            for module_name in layer_modules:
                module = self._get_module_by_name(module_name)
                if module is None:
                    warnings.warn(f"无法找到模块: {module_name}")
                    continue
                
                def create_hook(name: str):
                    def hook(module: nn.Module, input: Tuple, output: Any) -> None:
                        stats = current_stats[name]
                        
                        # 处理不同类型的输出
                        if isinstance(output, tuple):
                            hidden_states = output[0].detach()
                            self._update_activation_stats(stats, hidden_states)
                            
                            if len(output) > 1 and output[1] is not None:
                                stats['has_attention'] = True
                                attn_weights = output[1].detach()
                                self._update_attention_stats(stats, attn_weights)
                        else:
                            hidden_states = output.detach()
                            self._update_activation_stats(stats, hidden_states)
                    return hook
                
                handle = module.register_forward_hook(create_hook(module_name))
                current_handles.append(handle)
            
            print(f"   注册了 {len(current_handles)} 个钩子")
            
            # 显示钩子注册后的内存
            if torch.cuda.is_available():
                hook_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"   钩子注册后GPU内存: {hook_memory:.2f} GB")
            
            # 执行推理（只为当前层）
            self._inference_pass_for_layer(layer_idx + 1, len(layer_groups))
            
            # 显示推理后的内存
            if torch.cuda.is_available():
                inference_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"   推理完成后GPU内存: {inference_memory:.2f} GB")
            
            # 立即移除当前层钩子
            for handle in current_handles:
                handle.remove()
            current_handles.clear()
            
            # 显示钩子移除后的内存
            if torch.cuda.is_available():
                unhook_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"   钩子移除后GPU内存: {unhook_memory:.2f} GB")
            
            # 计算当前层的特征
            for module_name in layer_modules:
                if module_name in current_stats:
                    features = self._calculate_module_features(module_name, current_stats[module_name])
                    all_features.append(features)
                else:
                    # 如果模块没有统计数据，使用零向量
                    all_features.append(torch.zeros(8, dtype=torch.float32))
            
            # 清理当前层数据
            current_stats.clear()
            
            # 极限内存清理
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
            
            # 显示清理后的内存
            if torch.cuda.is_available():
                final_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"   内存清理后GPU内存: {final_memory:.2f} GB")
                print(f"   本层内存增量: {final_memory - pre_layer_memory:.2f} GB")
            
            print(f"=> 第 {layer_idx + 1} 层处理完成，累计特征数: {len(all_features)}")
        
        # 聚合所有特征
        if not all_features:
            raise ValueError("没有提取到任何特征向量")
        
        # 堆叠成二维张量
        state_tensor = torch.stack(all_features, dim=0)
        
        # 标准化处理
        normalized_tensor = self._normalize_features(state_tensor)
        
        # 最终内存状态
        if torch.cuda.is_available():
            final_total_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"=> 特征提取完成，最终GPU内存: {final_total_memory:.2f} GB")
        
        return normalized_tensor
    
    def _group_modules_by_layer(self) -> List[List[str]]:
        """
        将模块按层分组，每组包含同一层的所有模块
        
        Returns:
            List[List[str]]: 每个子列表包含一层的模块名称
        """
        layer_groups = {}
        
        for module_name in self.prunable_module_names:
            # 提取层编号
            layer_num = self._extract_layer_number(module_name)
            if layer_num not in layer_groups:
                layer_groups[layer_num] = []
            layer_groups[layer_num].append(module_name)
        
        # 按层编号排序并返回
        sorted_layers = sorted(layer_groups.keys())
        return [layer_groups[layer_num] for layer_num in sorted_layers]
    
    def _extract_layer_number(self, module_name: str) -> int:
        """
        从模块名称中提取层编号
        
        Args:
            module_name (str): 模块名称，如 "model.decoder.layers.0.self_attn"
            
        Returns:
            int: 层编号
        """
        import re
        # 匹配 "layers.数字" 模式
        match = re.search(r'layers\.(\d+)', module_name)
        if match:
            return int(match.group(1))
        else:
            # 如果没有找到层编号，返回 -1（将被分到最前面）
            return -1
    
    def _inference_pass_for_layer(self, current_layer: int, total_layers: int) -> None:
        """
        为特定层执行推理过程（极限内存优化版本）
        
        Args:
            current_layer (int): 当前层编号（用于显示）
            total_layers (int): 总层数（用于显示）
        """
        print(f"   执行第 {current_layer}/{total_layers} 层的推理...")
        
        # 从模型配置中获取词汇表大小
        vocab_size = getattr(self.model.config, 'vocab_size', 50272)
        
        self.model.eval()
        sample_count = 0
        
        # 极限内存优化：逐个样本处理
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.dataloader):
                if sample_count >= self.max_samples:
                    break
                
                # 处理数据加载器的输出格式
                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                    input_ids, target = batch_data[0], batch_data[1]
                    batch = input_ids
                elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 1:
                    batch = batch_data[0]
                else:
                    batch = batch_data
                
                # 验证和清理token ID
                if isinstance(batch, torch.Tensor):
                    max_token_id = batch.max().item()
                    min_token_id = batch.min().item()
                    
                    if max_token_id >= vocab_size or min_token_id < 0:
                        batch = torch.clamp(batch, 0, vocab_size - 1)
                
                # 移动到设备
                if isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device)
                
                # 逐个样本处理以减少内存峰值
                if batch.dim() > 1:
                    # 如果是批次数据，逐个处理
                    for single_sample_idx in range(batch.size(0)):
                        if sample_count >= self.max_samples:
                            break
                            
                        single_input = batch[single_sample_idx:single_sample_idx+1]  # 保持批次维度
                        
                        try:
                            # 执行前向传播 - 禁用attention输出以减少内存
                            kwargs = {
                                'input_ids': single_input,
                                'output_attentions': False,  # 关键改动：禁用attention输出
                                'use_cache': False  # 禁用缓存
                            }
                            
                            # 对于OPT模型添加attention_mask
                            if hasattr(self.model.config, 'model_type') and 'opt' in self.model.config.model_type.lower():
                                attention_mask = torch.ones_like(single_input)
                                kwargs['attention_mask'] = attention_mask
                            
                            # 确保输出被丢弃，不保留引用
                            with torch.no_grad():
                                _ = self.model(**kwargs)
                            
                            sample_count += 1
                            
                            # 立即清理单样本数据
                            del single_input
                            if 'attention_mask' in locals():
                                del attention_mask
                            
                            # 每个样本后都清理显存
                            torch.cuda.empty_cache()
                            
                        except Exception as e:
                            warnings.warn(f"层 {current_layer} 样本 {single_sample_idx} 推理失败: {e}")
                            continue
                else:
                    # 单个样本处理
                    if batch.dim() == 1:
                        batch = batch.unsqueeze(0)
                    
                    try:
                        kwargs = {
                            'input_ids': batch,
                            'output_attentions': False,  # 关键改动：禁用attention输出
                            'use_cache': False  # 禁用缓存
                        }
                        
                        if hasattr(self.model.config, 'model_type') and 'opt' in self.model.config.model_type.lower():
                            attention_mask = torch.ones_like(batch)
                            kwargs['attention_mask'] = attention_mask
                        
                        with torch.no_grad():
                            _ = self.model(**kwargs)
                        sample_count += 1
                        
                        # 清理数据
                        if 'attention_mask' in locals():
                            del attention_mask
                        
                    except Exception as e:
                        warnings.warn(f"层 {current_layer} 批次 {batch_idx} 推理失败: {e}")
                        continue
                
                # 清理批次数据
                del batch
                
                # 强制清理显存
                torch.cuda.empty_cache()
        
        print(f"   第 {current_layer} 层推理完成，处理了 {sample_count} 个样本")
    
    def _calculate_module_features(self, module_name: str, stats: dict) -> torch.Tensor:
        """
        根据统计数据计算单个模块的特征向量
        
        Args:
            module_name (str): 模块名称
            stats (dict): 模块的统计数据
            
        Returns:
            torch.Tensor: 该模块的8维特征向量
        """
        # 检查是否有数据
        if stats['count'] == 0:
            warnings.warn(f"模块 {module_name} 没有收集到数据")
            return torch.zeros(8, dtype=torch.float32)
        
        features = []
        
        # 计算基本激活特征
        count = stats['count'].item()
        mean_val = (stats['sum'] / count).item()
        var_val = (stats['sum_squared'] / count - mean_val ** 2).item()
        std_val = max(0.0, var_val) ** 0.5
        
        # 1. L2 范数
        l2_norm = (stats['l2_norm_sum'] / max(1, self.current_sample_count)).item()
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
            kurtosis = var_val / (std_val ** 4 + 1e-12) - 3.0 if std_val > 1e-6 else 0.0
            features.append(kurtosis)
        
        # 清理NaN值
        features = [0.0 if np.isnan(f) or np.isinf(f) else f for f in features]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _normalize_features(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        标准化特征张量
        
        Args:
            state_tensor (torch.Tensor): 原始特征张量
            
        Returns:
            torch.Tensor: 标准化后的特征张量
        """
        print("正在标准化特征张量...")
        
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
        
        print(f"特征标准化完成，张量形状: {normalized_tensor.shape}")
        print(f"特征统计: 均值={normalized_tensor.mean().item():.4f}, "
              f"标准差={normalized_tensor.std().item():.4f}")
        
        return normalized_tensor
    
    def _generate_cache_key(self) -> str:
        """
        根据模型、数据集、样本数量等生成唯一的缓存键
        
        Returns:
            str: 缓存键字符串
        """
        # 收集关键信息
        key_info = {
            'model_name': getattr(self.model.config, '_name_or_path', 'unknown_model'),
            'model_type': getattr(self.model.config, 'model_type', 'unknown_type'),
            'num_layers': getattr(self.model.config, 'num_hidden_layers', 0),
            'hidden_size': getattr(self.model.config, 'hidden_size', 0),
            'vocab_size': getattr(self.model.config, 'vocab_size', 0),
            'max_samples': self.max_samples,
            'sparsity_threshold': self.sparsity_threshold,
            'prunable_modules': sorted(self.prunable_module_names),  # 排序确保一致性
            'num_modules': len(self.prunable_module_names)
        }
        
        # 转换为JSON字符串并生成哈希
        key_str = json.dumps(key_info, sort_keys=True)
        hash_obj = hashlib.md5(key_str.encode('utf-8'))
        cache_key = hash_obj.hexdigest()[:16]  # 使用前16位作为缓存键
        
        print(f"=> 缓存键生成信息:")
        print(f"   模型: {key_info['model_name']}")
        print(f"   类型: {key_info['model_type']}")
        print(f"   层数: {key_info['num_layers']}")
        print(f"   样本数: {key_info['max_samples']}")
        print(f"   模块数: {key_info['num_modules']}")
        print(f"   缓存键: {cache_key}")
        
        return cache_key
    
    def _save_features_to_cache(self, features_tensor: torch.Tensor) -> None:
        """
        保存特征张量到缓存文件
        
        Args:
            features_tensor (torch.Tensor): 要保存的特征张量
        """
        cache_data = {
            'features': features_tensor,
            'metadata': {
                'model_name': getattr(self.model.config, '_name_or_path', 'unknown_model'),
                'model_type': getattr(self.model.config, 'model_type', 'unknown_type'),
                'num_layers': getattr(self.model.config, 'num_hidden_layers', 0),
                'hidden_size': getattr(self.model.config, 'hidden_size', 0),
                'vocab_size': getattr(self.model.config, 'vocab_size', 0),
                'max_samples': self.max_samples,
                'sparsity_threshold': self.sparsity_threshold,
                'prunable_modules': self.prunable_module_names,
                'num_modules': len(self.prunable_module_names),
                'feature_shape': list(features_tensor.shape),
                'extraction_time': datetime.now().isoformat(),
                'cache_key': self.cache_key
            }
        }
        
        try:
            torch.save(cache_data, self.cache_file)
            print(f"=> 特征已保存到缓存: {self.cache_file}")
            print(f"   张量形状: {features_tensor.shape}")
            print(f"   文件大小: {os.path.getsize(self.cache_file) / 1024:.2f} KB")
        except Exception as e:
            warnings.warn(f"保存缓存失败: {e}")
    
    def _load_features_from_cache(self) -> Optional[torch.Tensor]:
        """
        从缓存文件加载特征张量
        
        Returns:
            Optional[torch.Tensor]: 加载的特征张量，如果失败则返回None
        """
        if not os.path.exists(self.cache_file):
            print("=> 缓存文件不存在，需要重新计算特征")
            return None
        
        try:
            print(f"=> 发现缓存文件: {self.cache_file}")
            cache_data = torch.load(self.cache_file, map_location='cpu')
            
            # 验证缓存有效性
            if not self._validate_cache(cache_data):
                print("=> 缓存验证失败，需要重新计算特征")
                return None
            
            features = cache_data['features']
            metadata = cache_data['metadata']
            
            print("=> 缓存验证成功，加载特征:")
            print(f"   提取时间: {metadata['extraction_time']}")
            print(f"   张量形状: {features.shape}")
            print(f"   样本数: {metadata['max_samples']}")
            print(f"   模块数: {metadata['num_modules']}")
            
            return features
            
        except Exception as e:
            warnings.warn(f"加载缓存失败: {e}")
            return None
    
    def _validate_cache(self, cache_data: dict) -> bool:
        """
        验证缓存数据的有效性
        
        Args:
            cache_data (dict): 缓存数据
            
        Returns:
            bool: 缓存是否有效
        """
        try:
            metadata = cache_data['metadata']
            
            # 验证关键参数是否匹配
            checks = [
                metadata['max_samples'] == self.max_samples,
                metadata['sparsity_threshold'] == self.sparsity_threshold,
                metadata['num_modules'] == len(self.prunable_module_names),
                metadata['prunable_modules'] == self.prunable_module_names,
                metadata['cache_key'] == self.cache_key
            ]
            
            if not all(checks):
                print("=> 缓存参数不匹配:")
                print(f"   样本数: {metadata['max_samples']} vs {self.max_samples}")
                print(f"   模块数: {metadata['num_modules']} vs {len(self.prunable_module_names)}")
                print(f"   缓存键: {metadata['cache_key']} vs {self.cache_key}")
                return False
            
            # 验证特征张量形状
            features = cache_data['features']
            expected_shape = (len(self.prunable_module_names), 8)
            if features.shape != expected_shape:
                print(f"=> 特征张量形状不匹配: {features.shape} vs {expected_shape}")
                return False
            
            return True
            
        except KeyError as e:
            print(f"=> 缓存格式错误，缺少字段: {e}")
            return False
    
    @staticmethod
    def list_cache_files(cache_dir: str = "./feature_cache") -> List[Dict]:
        """
        列出缓存目录中的所有特征缓存文件
        
        Args:
            cache_dir (str): 缓存目录路径
            
        Returns:
            List[Dict]: 缓存文件信息列表
        """
        if not os.path.exists(cache_dir):
            print(f"缓存目录不存在: {cache_dir}")
            return []
        
        cache_files = []
        for filename in os.listdir(cache_dir):
            if filename.startswith("features_") and filename.endswith(".pt"):
                filepath = os.path.join(cache_dir, filename)
                try:
                    cache_data = torch.load(filepath, map_location='cpu')
                    metadata = cache_data['metadata']
                    
                    file_info = {
                        'filename': filename,
                        'filepath': filepath,
                        'size_kb': os.path.getsize(filepath) / 1024,
                        'model_name': metadata.get('model_name', 'unknown'),
                        'max_samples': metadata.get('max_samples', 0),
                        'num_modules': metadata.get('num_modules', 0),
                        'extraction_time': metadata.get('extraction_time', 'unknown'),
                        'cache_key': metadata.get('cache_key', 'unknown')
                    }
                    cache_files.append(file_info)
                    
                except Exception as e:
                    print(f"读取缓存文件失败 {filename}: {e}")
        
        return cache_files
    
    @staticmethod
    def print_cache_summary(cache_dir: str = "./feature_cache") -> None:
        """
        打印缓存文件摘要信息
        
        Args:
            cache_dir (str): 缓存目录路径
        """
        cache_files = FeatureExtractor.list_cache_files(cache_dir)
        
        if not cache_files:
            print("没有找到缓存文件")
            return
        
        print(f"\n缓存目录: {cache_dir}")
        print(f"缓存文件数量: {len(cache_files)}")
        print("-" * 80)
        print(f"{'文件名':<25} {'模型':<20} {'样本数':<8} {'模块数':<8} {'大小(KB)':<10} {'提取时间':<20}")
        print("-" * 80)
        
        total_size = 0
        for info in cache_files:
            print(f"{info['filename']:<25} {info['model_name']:<20} {info['max_samples']:<8} "
                  f"{info['num_modules']:<8} {info['size_kb']:<10.1f} {info['extraction_time'][:19]:<20}")
            total_size += info['size_kb']
        
        print("-" * 80)
        print(f"总缓存大小: {total_size:.1f} KB ({total_size/1024:.1f} MB)")
    
    @staticmethod
    def clear_cache(cache_dir: str = "./feature_cache", confirm: bool = True) -> None:
        """
        清理缓存目录
        
        Args:
            cache_dir (str): 缓存目录路径
            confirm (bool): 是否需要确认
        """
        cache_files = FeatureExtractor.list_cache_files(cache_dir)
        
        if not cache_files:
            print("没有找到缓存文件")
            return
        
        if confirm:
            FeatureExtractor.print_cache_summary(cache_dir)
            response = input(f"\n确定要删除所有 {len(cache_files)} 个缓存文件吗？ (y/N): ")
            if response.lower() != 'y':
                print("取消清理")
                return
        
        deleted_count = 0
        for info in cache_files:
            try:
                os.remove(info['filepath'])
                deleted_count += 1
            except Exception as e:
                print(f"删除文件失败 {info['filename']}: {e}")
        
        print(f"成功删除 {deleted_count} 个缓存文件")
    
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
    使用示例和缓存管理工具
    """
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        cache_dir = sys.argv[2] if len(sys.argv) > 2 else "./feature_cache"
        
        if command == "list":
            print("特征缓存管理工具 - 列出缓存文件")
            FeatureExtractor.print_cache_summary(cache_dir)
            
        elif command == "clear":
            print("特征缓存管理工具 - 清理缓存")
            FeatureExtractor.clear_cache(cache_dir)
            
        elif command == "help":
            print("特征缓存管理工具")
            print("用法:")
            print("  python feature_extractor.py list [cache_dir]    # 列出缓存文件")
            print("  python feature_extractor.py clear [cache_dir]   # 清理缓存文件")
            print("  python feature_extractor.py help                # 显示帮助信息")
            print("  python feature_extractor.py                     # 显示使用示例")
            
        else:
            print(f"未知命令: {command}")
            print("使用 'python feature_extractor.py help' 查看帮助")
        
        sys.exit(0)
    
    print("FeatureExtractor 演示代码和缓存功能")
    print("=" * 50)
    
    # 缓存功能说明
    print("缓存功能说明:")
    print("- 特征提取完成后会自动保存到缓存文件")
    print("- 缓存键基于模型、样本数、模块列表等参数生成")
    print("- 下次相同配置时会自动从缓存加载，无需重新计算")
    print("- 缓存文件默认保存在 './feature_cache/' 目录")
    print()
    
    print("缓存管理命令:")
    print("  python feature_extractor.py list     # 列出所有缓存文件")
    print("  python feature_extractor.py clear    # 清理所有缓存文件")
    print()
    
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
    
    # 4. 运行特征提取（含缓存）
    print("\n步骤 4: 运行特征提取（支持缓存）")
    print("示例代码:")
    print("""
    # 实例化特征提取器（支持缓存）
    print("开始特征提取...")
    extractor = FeatureExtractor(
        model=model, 
        dataloader=dataloader, 
        prunable_module_names=prunable_module_names,
        max_samples=32,
        cache_dir="./my_feature_cache"  # 指定缓存目录
    )
    
    # 执行特征提取（会自动检查缓存）
    state_tensor = extractor.extract()
    
    print(f"特征提取完成！")
    print(f"最终状态张量形状: {state_tensor.shape}")
    print(f"张量统计信息:")
    print(f"  均值: {state_tensor.mean().item():.6f}")
    print(f"  标准差: {state_tensor.std().item():.6f}")
    print(f"  最小值: {state_tensor.min().item():.6f}")
    print(f"  最大值: {state_tensor.max().item():.6f}")
    """)
    
    # 5. 缓存管理
    print("\n步骤 5: 缓存管理")
    print("示例代码:")
    print("""
    # 查看缓存文件
    FeatureExtractor.print_cache_summary("./my_feature_cache")
    
    # 列出缓存文件详情
    cache_files = FeatureExtractor.list_cache_files("./my_feature_cache")
    for file_info in cache_files:
        print(f"文件: {file_info['filename']}")
        print(f"模型: {file_info['model_name']}")
        print(f"样本数: {file_info['max_samples']}")
        print(f"大小: {file_info['size_kb']:.1f} KB")
        print()
    
    # 清理缓存（可选）
    # FeatureExtractor.clear_cache("./my_feature_cache")
    """)
    
    # 使用建议
    print("\n" + "=" * 50)
    print("使用建议:")
    print("1. 根据具体模型调整 prunable_module_names")
    print("2. 根据硬件情况调整 max_samples 参数")
    print("3. 指定合适的缓存目录，避免磁盘空间不足")
    print("4. 定期清理不需要的缓存文件")
    print("5. 相同配置的训练会自动使用缓存，大幅节省时间")
    
    print("\n特征提取器类定义完成！支持智能缓存功能！")
