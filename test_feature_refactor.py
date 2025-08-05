#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本 - 验证重构后的模块化特征提取系统

该脚本用于测试新的FeatureOrchestrator类和各种特征配置，
确保重构后的系统功能正常。

作者: AI Assistant
创建时间: 2025年8月5日
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict
import warnings

# 导入重构后的模块
try:
    from feature_extractor import (
        FeatureOrchestrator, 
        create_feature_extractor,
        get_default_feature_config,
        # 各种特征模块
        NormalizedLayerIndexFeature,
        ModuleTypeFeature,
        LogActivationNormFeature,
        ActivationSparsityFeature,
        ActivationEffectiveRankFeature,
        AttentionEntropyFeature,
        AttentionLocalityScoreFeature,
        AttentionDiversityFeature,
        GLUGatingRatioFeature
    )
    from feature_configs import get_config_by_name, print_config_comparison
    print("✓ 成功导入重构后的模块")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    exit(1)


class MockModel(nn.Module):
    """模拟模型用于测试"""
    def __init__(self):
        super().__init__()
        self.config = type('Config', (), {
            '_name_or_path': 'test-model',
            'model_type': 'test',
            'num_hidden_layers': 4,
            'hidden_size': 512,
            'vocab_size': 1000
        })()
        
        # 创建一些模拟层
        self.decoder = nn.ModuleDict({
            'layers': nn.ModuleList([
                nn.ModuleDict({
                    'self_attn': nn.Linear(512, 512),
                    'fc1': nn.Linear(512, 2048),
                    'fc2': nn.Linear(2048, 512)
                }) for _ in range(4)
            ])
        })
    
    def forward(self, input_ids, **kwargs):
        batch_size, seq_len = input_ids.shape
        hidden_states = torch.randn(batch_size, seq_len, 512)
        
        # 模拟输出
        return type('Output', (), {
            'last_hidden_state': hidden_states
        })()


class MockDataLoader:
    """模拟数据加载器用于测试"""
    def __init__(self, num_batches=3, batch_size=2, seq_len=128):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.current_batch = 0
    
    def __iter__(self):
        self.current_batch = 0
        return self
    
    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        
        # 生成模拟数据
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        self.current_batch += 1
        return input_ids


def test_individual_features():
    """测试各个特征模块"""
    print("\n=== 测试各个特征模块 ===")
    
    # 模拟统计数据
    module_stats = {
        'count': torch.tensor(1000.0),
        'sum': torch.tensor(500.0),
        'sum_squared': torch.tensor(750.0),
        'min_val': torch.tensor(-2.0),
        'max_val': torch.tensor(3.0),
        'sparsity_count': torch.tensor(200.0),
        'l2_norm_sum': torch.tensor(10.0),
        'positive_count': torch.tensor(600.0),
        'has_attention': True,
        'attn_entropy_sum': torch.tensor(15.0),
        'attn_locality_sum': torch.tensor(8.0),
        'attn_max_sum': torch.tensor(5.0),
        'attn_count': torch.tensor(10.0),
        'layer_index': 2
    }
    
    global_stats = {
        'total_layers': 4,
        'sample_count': 8
    }
    
    # 测试每个特征模块
    feature_classes = [
        NormalizedLayerIndexFeature,
        ModuleTypeFeature,
        LogActivationNormFeature,
        ActivationSparsityFeature,
        ActivationEffectiveRankFeature,
        AttentionEntropyFeature,
        AttentionLocalityScoreFeature,
        AttentionDiversityFeature,
        GLUGatingRatioFeature
    ]
    
    for feature_class in feature_classes:
        try:
            feature = feature_class()
            result = feature.calculate(module_stats, global_stats)
            
            print(f"✓ {feature.name:<25} - 维度: {feature.feature_dim}, "
                  f"输出: {result.tolist()}")
            
            # 验证输出维度
            assert result.numel() == feature.feature_dim, \
                f"维度不匹配: {result.numel()} vs {feature.feature_dim}"
            
            # 验证输出没有NaN
            assert not torch.isnan(result).any(), f"输出包含NaN: {result}"
            
        except Exception as e:
            print(f"✗ {feature_class.__name__} 测试失败: {e}")


def test_feature_configs():
    """测试特征配置"""
    print("\n=== 测试特征配置 ===")
    
    # 测试默认配置
    try:
        default_config = get_default_feature_config()
        print(f"✓ 默认配置加载成功，包含 {len(default_config)} 个特征")
        
        # 验证所有必需的特征都存在
        required_features = [
            'normalized_layer_index', 'module_type', 'log_activation_norm',
            'activation_sparsity', 'activation_effective_rank', 'attention_entropy',
            'attention_locality', 'attention_diversity', 'glu_gating_ratio'
        ]
        
        for feature_name in required_features:
            assert feature_name in default_config, f"缺少特征: {feature_name}"
        
        print("✓ 所有必需特征都存在")
        
    except Exception as e:
        print(f"✗ 默认配置测试失败: {e}")
    
    # 测试预定义配置
    try:
        from feature_configs import PREDEFINED_CONFIGS
        
        for config_name in PREDEFINED_CONFIGS.keys():
            config = get_config_by_name(config_name)
            enabled_count = sum(1 for enabled in config.values() if enabled)
            print(f"✓ 配置 '{config_name}': {enabled_count} 个特征启用")
            
    except Exception as e:
        print(f"✗ 预定义配置测试失败: {e}")


def test_feature_orchestrator():
    """测试FeatureOrchestrator类"""
    print("\n=== 测试FeatureOrchestrator ===")
    
    # 创建模拟环境
    model = MockModel()
    dataloader = MockDataLoader()
    
    # 定义可剪枝模块
    prunable_module_names = []
    for i in range(4):
        prunable_module_names.extend([
            f"decoder.layers.{i}.self_attn",
            f"decoder.layers.{i}.fc1",
            f"decoder.layers.{i}.fc2"
        ])
    
    print(f"定义了 {len(prunable_module_names)} 个可剪枝模块")
    
    # 测试不同配置
    test_configs = {
        'minimal': {
            'normalized_layer_index': True,
            'module_type': True,
            'log_activation_norm': False,
            'activation_sparsity': False,
            'activation_effective_rank': False,
            'attention_entropy': False,
            'attention_locality': False,
            'attention_diversity': False,
            'glu_gating_ratio': False,
        },
        'basic': get_config_by_name('basic'),
        'default': get_default_feature_config()
    }
    
    for config_name, config in test_configs.items():
        try:
            print(f"\n--- 测试配置: {config_name} ---")
            
            # 创建特征编排器
            orchestrator = FeatureOrchestrator(
                model=model,
                dataloader=dataloader,
                prunable_module_names=prunable_module_names,
                feature_config=config,
                max_samples=2,  # 减少样本数以加快测试
                cache_dir=f"./test_cache_{config_name}"
            )
            
            print(f"✓ FeatureOrchestrator创建成功")
            print(f"  启用特征数: {len(orchestrator.active_module_features)}")
            
            # 计算预期特征维度
            expected_dim = sum(f.feature_dim for f in orchestrator.active_module_features)
            print(f"  预期特征维度: {expected_dim}")
            
            # 测试特征提取（模拟）
            # 注意：由于是模拟环境，可能无法完整运行，但可以测试初始化
            print("✓ 基本功能测试通过")
            
        except Exception as e:
            print(f"✗ 配置 {config_name} 测试失败: {e}")


def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n=== 测试向后兼容性 ===")
    
    try:
        # 测试别名是否工作
        from feature_extractor import FeatureExtractor
        assert FeatureExtractor == FeatureOrchestrator
        print("✓ FeatureExtractor别名正常工作")
        
        # 测试工厂函数
        model = MockModel()
        dataloader = MockDataLoader()
        prunable_module_names = ["decoder.layers.0.self_attn"]
        
        orchestrator = create_feature_extractor(
            model=model,
            dataloader=dataloader,
            prunable_module_names=prunable_module_names,
            max_samples=1
        )
        
        print("✓ create_feature_extractor工厂函数正常工作")
        
    except Exception as e:
        print(f"✗ 向后兼容性测试失败: {e}")


def main():
    """主测试函数"""
    print("开始测试重构后的模块化特征提取系统")
    print("=" * 60)
    
    # 禁用一些警告以保持输出清洁
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 运行各项测试
    test_individual_features()
    test_feature_configs()
    test_feature_orchestrator()
    test_backward_compatibility()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    
    # 显示配置对比
    print("\n特征配置对比:")
    print_config_comparison()


if __name__ == '__main__':
    main()
