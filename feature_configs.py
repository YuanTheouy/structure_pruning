#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征配置示例 - 展示如何配置和使用模块化特征提取系统

该文件提供了各种特征配置示例，演示如何根据不同的实验需求
配置特征模块组合。

作者: AI Assistant
创建时间: 2025年8月5日
"""

from feature_extractor import get_default_feature_config, create_feature_extractor


def get_basic_feature_config():
    """获取基础特征配置 - 只包含最基本的特征"""
    return {
        'normalized_layer_index': True,      # 层索引
        'module_type': True,                 # 模块类型
        'log_activation_norm': True,         # 激活范数
        'activation_sparsity': True,         # 稀疏度
        'activation_effective_rank': False,  # 禁用复杂特征
        'attention_entropy': False,
        'attention_locality': False,
        'attention_diversity': False,
        'glu_gating_ratio': False,
    }


def get_attention_focused_config():
    """获取注意力导向配置 - 重点关注注意力相关特征"""
    return {
        'normalized_layer_index': True,      # 保留基础特征
        'module_type': True,
        'log_activation_norm': False,        # 简化激活特征
        'activation_sparsity': False,
        'activation_effective_rank': False,
        'attention_entropy': True,           # 重点：注意力特征
        'attention_locality': True,
        'attention_diversity': True,
        'glu_gating_ratio': False,
    }


def get_comprehensive_config():
    """获取全面特征配置 - 启用所有特征"""
    return {
        'normalized_layer_index': True,
        'module_type': True,
        'log_activation_norm': True,
        'activation_sparsity': True,
        'activation_effective_rank': True,
        'attention_entropy': True,
        'attention_locality': True,
        'attention_diversity': True,
        'glu_gating_ratio': True,
    }


def get_minimal_config():
    """获取最小特征配置 - 只使用必要特征"""
    return {
        'normalized_layer_index': True,      # 只保留最基本的特征
        'module_type': True,
        'log_activation_norm': False,
        'activation_sparsity': False,
        'activation_effective_rank': False,
        'attention_entropy': False,
        'attention_locality': False,
        'attention_diversity': False,
        'glu_gating_ratio': False,
    }


def get_activation_focused_config():
    """获取激活导向配置 - 重点关注激活相关特征"""
    return {
        'normalized_layer_index': True,
        'module_type': True,
        'log_activation_norm': True,         # 重点：激活特征
        'activation_sparsity': True,
        'activation_effective_rank': True,
        'attention_entropy': False,          # 简化注意力特征
        'attention_locality': False,
        'attention_diversity': False,
        'glu_gating_ratio': True,            # FFN门控特征
    }


# 预定义配置字典
PREDEFINED_CONFIGS = {
    'default': get_default_feature_config,
    'basic': get_basic_feature_config,
    'attention': get_attention_focused_config,
    'comprehensive': get_comprehensive_config,
    'minimal': get_minimal_config,
    'activation': get_activation_focused_config,
}


def get_config_by_name(config_name: str):
    """
    根据名称获取预定义配置
    
    Args:
        config_name (str): 配置名称
        
    Returns:
        dict: 特征配置字典
        
    Raises:
        ValueError: 如果配置名称不存在
    """
    if config_name not in PREDEFINED_CONFIGS:
        available_configs = list(PREDEFINED_CONFIGS.keys())
        raise ValueError(f"未知配置名称: {config_name}. 可用配置: {available_configs}")
    
    return PREDEFINED_CONFIGS[config_name]()


def print_config_comparison():
    """打印各种配置的对比"""
    print("特征配置对比:")
    print("=" * 80)
    
    feature_names = list(get_default_feature_config().keys())
    config_names = list(PREDEFINED_CONFIGS.keys())
    
    # 打印表头
    print(f"{'特征名称':<25}", end="")
    for config_name in config_names:
        print(f"{config_name:<12}", end="")
    print()
    print("-" * 80)
    
    # 打印每个特征在各配置中的状态
    for feature_name in feature_names:
        print(f"{feature_name:<25}", end="")
        for config_name in config_names:
            config = PREDEFINED_CONFIGS[config_name]()
            status = "✓" if config.get(feature_name, False) else "✗"
            print(f"{status:<12}", end="")
        print()
    
    print("-" * 80)
    
    # 打印特征维度统计
    print(f"{'总特征维度':<25}", end="")
    for config_name in config_names:
        config = PREDEFINED_CONFIGS[config_name]()
        total_dim = sum(1 for enabled in config.values() if enabled)
        print(f"{total_dim:<12}", end="")
    print()
    print("=" * 80)


def demonstrate_usage():
    """演示如何使用配置"""
    print("\n使用示例:")
    print("1. 使用预定义配置:")
    print("""
    # 获取注意力导向配置
    config = get_config_by_name('attention')
    
    # 创建特征提取器
    orchestrator = create_feature_extractor(
        model=model,
        dataloader=dataloader,
        prunable_module_names=prunable_module_names,
        feature_config=config,
        max_samples=32
    )
    
    # 提取特征
    features = orchestrator.extract()
    """)
    
    print("2. 自定义配置:")
    print("""
    # 创建自定义配置
    custom_config = {
        'normalized_layer_index': True,
        'module_type': True,
        'log_activation_norm': True,
        'activation_sparsity': False,      # 禁用特定特征
        'activation_effective_rank': False,
        'attention_entropy': True,         # 只启用部分注意力特征
        'attention_locality': False,
        'attention_diversity': False,
        'glu_gating_ratio': True,
    }
    
    # 使用自定义配置
    orchestrator = create_feature_extractor(
        model=model,
        dataloader=dataloader,
        prunable_module_names=prunable_module_names,
        feature_config=custom_config
    )
    """)
    
    print("3. 消融实验:")
    print("""
    # 比较不同配置的效果
    configs_to_test = ['basic', 'attention', 'comprehensive']
    
    results = {}
    for config_name in configs_to_test:
        print(f"测试配置: {config_name}")
        
        config = get_config_by_name(config_name)
        orchestrator = create_feature_extractor(
            model=model,
            dataloader=dataloader,
            prunable_module_names=prunable_module_names,
            feature_config=config,
            cache_dir=f"./cache_{config_name}"  # 不同配置使用不同缓存
        )
        
        features = orchestrator.extract()
        results[config_name] = features
        
        print(f"特征形状: {features.shape}")
        print(f"特征统计: 均值={features.mean():.4f}, 标准差={features.std():.4f}")
        print()
    """)


if __name__ == '__main__':
    print("特征配置管理工具")
    print("=" * 40)
    
    # 显示配置对比
    print_config_comparison()
    
    # 演示使用方法
    demonstrate_usage()
    
    print("\n可用的预定义配置:")
    for config_name, config_func in PREDEFINED_CONFIGS.items():
        config = config_func()
        enabled_count = sum(1 for enabled in config.values() if enabled)
        print(f"  {config_name}: {enabled_count} 个特征启用")
    
    print("\n使用方法:")
    print("  from feature_configs import get_config_by_name")
    print("  config = get_config_by_name('attention')")
    print("  # 然后使用config创建FeatureOrchestrator")
