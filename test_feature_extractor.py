#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FeatureExtractor 测试脚本

快速验证 FeatureExtractor 类的正确性，包括：
1. 模型加载测试
2. 模块名称自动检测
3. 小数据集特征提取测试
4. 输出验证和性能测试

作者: AI Assistant
创建时间: 2025年8月4日
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import traceback
from typing import List, Dict

# 添加当前目录到路径以便导入 FeatureExtractor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import FeatureExtractor


class FeatureExtractorTester:
    """
    FeatureExtractor 测试器
    """
    
    def __init__(self):
        # 使用与 searchPPO13.sh 相同的配置
        self.model_path = "/home/theo/data/yx_repository/01_Models/opt-1.3b"
        self.model_name = "opt-1.3b"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 测试配置
        self.test_batch_size = 2
        self.test_seq_length = 128
        self.test_samples = 5  # 小数据集测试
        self.max_test_layers = 3  # 只测试前3层以加快速度
        
        print(f"🚀 FeatureExtractor 测试器初始化")
        print(f"📍 模型路径: {self.model_path}")
        print(f"🎯 设备: {self.device}")
        print(f"📊 测试配置: {self.test_samples} 样本, {self.max_test_layers} 层")
        print("=" * 60)
    
    def test_model_loading(self) -> bool:
        """
        测试1: 模型和分词器加载
        """
        print("🔍 测试1: 模型和分词器加载...")
        
        try:
            start_time = time.time()
            
            # 加载分词器
            print("  📥 加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            print("  📥 加载模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device.type != "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            load_time = time.time() - start_time
            
            # 获取模型信息
            num_layers = self.model.config.num_hidden_layers
            hidden_size = self.model.config.hidden_size
            num_attention_heads = self.model.config.num_attention_heads
            
            print(f"  ✅ 模型加载成功! 耗时: {load_time:.2f}秒")
            print(f"     - 层数: {num_layers}")
            print(f"     - 隐藏维度: {hidden_size}")
            print(f"     - 注意力头数: {num_attention_heads}")
            print(f"     - 模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ 模型加载失败: {e}")
            traceback.print_exc()
            return False
    
    def test_module_detection(self) -> bool:
        """
        测试2: 可剪枝模块名称自动检测
        """
        print("\n🔍 测试2: 可剪枝模块检测...")
        
        try:
            self.prunable_module_names = []
            
            # 检测 OPT 模型的层结构
            num_layers = min(self.model.config.num_hidden_layers, self.max_test_layers)
            
            print(f"  🔎 检测前 {num_layers} 层的可剪枝模块...")
            
            for i in range(num_layers):
                # 检查各种可能的模块名称
                possible_names = [
                    f"model.decoder.layers.{i}.self_attn",
                    f"model.decoder.layers.{i}.fc1",
                    f"model.decoder.layers.{i}.fc2",
                    f"decoder.layers.{i}.self_attn",
                    f"decoder.layers.{i}.fc1",
                    f"decoder.layers.{i}.fc2",
                ]
                
                for name in possible_names:
                    if self._module_exists(name):
                        self.prunable_module_names.append(name)
                        print(f"    ✓ 找到模块: {name}")
            
            if not self.prunable_module_names:
                print("  ⚠️  未找到预期的模块名称，尝试通用检测...")
                self._detect_modules_generic()
            
            print(f"  ✅ 检测完成! 共找到 {len(self.prunable_module_names)} 个可剪枝模块")
            
            # 显示模块类型统计
            attn_modules = [name for name in self.prunable_module_names if "attn" in name]
            ffn_modules = [name for name in self.prunable_module_names if "fc" in name or "mlp" in name]
            
            print(f"     - 注意力模块: {len(attn_modules)} 个")
            print(f"     - FFN模块: {len(ffn_modules)} 个")
            
            return len(self.prunable_module_names) > 0
            
        except Exception as e:
            print(f"  ❌ 模块检测失败: {e}")
            traceback.print_exc()
            return False
    
    def _module_exists(self, name: str) -> bool:
        """检查模块是否存在"""
        try:
            parts = name.split('.')
            module = self.model
            for part in parts:
                module = getattr(module, part)
            return True
        except AttributeError:
            return False
    
    def _detect_modules_generic(self):
        """通用模块检测方法"""
        print("  🔍 执行通用模块检测...")
        
        def find_modules(module, prefix=""):
            for name, submodule in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # 检查是否是注意力或FFN模块
                if any(keyword in name.lower() for keyword in ["attn", "attention"]):
                    self.prunable_module_names.append(full_name)
                    print(f"    ✓ 检测到注意力模块: {full_name}")
                elif any(keyword in name.lower() for keyword in ["fc", "linear", "mlp", "feed_forward"]):
                    self.prunable_module_names.append(full_name)
                    print(f"    ✓ 检测到FFN模块: {full_name}")
                else:
                    # 递归检查子模块，但限制深度
                    if len(full_name.split('.')) < 5:
                        find_modules(submodule, full_name)
        
        find_modules(self.model)
        
        # 限制测试模块数量
        if len(self.prunable_module_names) > 10:
            self.prunable_module_names = self.prunable_module_names[:10]
            print(f"    🔧 限制测试模块数量为前10个")
    
    def test_data_preparation(self) -> bool:
        """
        测试3: 测试数据准备
        """
        print("\n🔍 测试3: 测试数据准备...")
        
        try:
            # 创建简单的测试文本
            test_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "Large language models have revolutionized natural language processing.",
                "Neural networks can learn complex patterns from data.",
                "Transformer architecture has become the foundation of modern NLP."
            ]
            
            print(f"  📝 准备 {len(test_texts)} 个测试样本...")
            
            # 分词和编码
            encoded_inputs = self.tokenizer(
                test_texts,
                padding=True,
                truncation=True,
                max_length=self.test_seq_length,
                return_tensors="pt"
            )
            
            # 创建自定义数据集类来返回字典格式
            class CustomDataset(torch.utils.data.Dataset):
                def __init__(self, input_ids, attention_mask):
                    self.input_ids = input_ids
                    self.attention_mask = attention_mask
                
                def __len__(self):
                    return len(self.input_ids)
                
                def __getitem__(self, idx):
                    return {
                        'input_ids': self.input_ids[idx],
                        'attention_mask': self.attention_mask[idx]
                    }
            
            dataset = CustomDataset(
                encoded_inputs['input_ids'],
                encoded_inputs['attention_mask']
            )
            
            self.test_dataloader = DataLoader(
                dataset,
                batch_size=self.test_batch_size,
                shuffle=False
            )
            
            print(f"  ✅ 数据准备完成!")
            print(f"     - 样本数: {len(test_texts)}")
            print(f"     - 批次大小: {self.test_batch_size}")
            print(f"     - 序列长度: {self.test_seq_length}")
            print(f"     - 数据形状: {encoded_inputs['input_ids'].shape}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ 数据准备失败: {e}")
            traceback.print_exc()
            return False
    
    def test_feature_extraction(self) -> bool:
        """
        测试4: 特征提取核心功能
        """
        print("\n🔍 测试4: 特征提取...")
        
        try:
            start_time = time.time()
            
            # 创建特征提取器
            print("  🎯 初始化 FeatureExtractor...")
            extractor = FeatureExtractor(
                model=self.model,
                dataloader=self.test_dataloader,
                prunable_module_names=self.prunable_module_names
            )
            
            # 设置更小的样本限制以加快测试
            extractor.max_samples = self.test_samples
            
            print("  🔄 开始特征提取...")
            
            # 执行特征提取
            state_tensor = extractor.extract()
            
            extraction_time = time.time() - start_time
            
            # 验证输出
            expected_shape = (len(self.prunable_module_names), 8)  # 8个特征
            
            print(f"  ✅ 特征提取完成! 耗时: {extraction_time:.2f}秒")
            print(f"     - 输出形状: {state_tensor.shape}")
            print(f"     - 期望形状: {expected_shape}")
            print(f"     - 数据类型: {state_tensor.dtype}")
            print(f"     - 设备: {state_tensor.device}")
            
            # 形状验证
            if state_tensor.shape != expected_shape:
                print(f"  ⚠️  形状不匹配！期望 {expected_shape}，得到 {state_tensor.shape}")
                return False
            
            # 数值验证
            self._validate_tensor_values(state_tensor)
            
            # 保存测试结果
            self._save_test_results(state_tensor, extraction_time)
            
            return True
            
        except Exception as e:
            print(f"  ❌ 特征提取失败: {e}")
            traceback.print_exc()
            return False
    
    def _validate_tensor_values(self, tensor: torch.Tensor):
        """验证张量数值的合理性"""
        print("  🔍 验证张量数值...")
        
        # 基本统计
        mean_val = tensor.mean().item()
        std_val = tensor.std().item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        print(f"     - 均值: {mean_val:.6f}")
        print(f"     - 标准差: {std_val:.6f}")
        print(f"     - 最小值: {min_val:.6f}")
        print(f"     - 最大值: {max_val:.6f}")
        
        # 检查异常值
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        
        if nan_count > 0:
            print(f"  ⚠️  发现 {nan_count} 个 NaN 值")
        if inf_count > 0:
            print(f"  ⚠️  发现 {inf_count} 个 Inf 值")
        
        # 检查标准化效果（应该接近标准正态分布）
        if abs(mean_val) < 0.1 and 0.8 < std_val < 1.2:
            print("  ✅ 标准化效果良好")
        else:
            print("  ⚠️  标准化效果可能需要检查")
    
    def _save_test_results(self, tensor: torch.Tensor, extraction_time: float):
        """保存测试结果"""
        try:
            test_results = {
                'state_tensor': tensor,
                'module_names': self.prunable_module_names,
                'test_metadata': {
                    'extraction_time': extraction_time,
                    'num_modules': len(self.prunable_module_names),
                    'num_features': tensor.shape[1],
                    'test_samples': self.test_samples,
                    'model_path': self.model_path,
                    'device': str(self.device),
                    'timestamp': time.strftime('%Y%m%d_%H%M%S')
                }
            }
            
            output_path = 'test_feature_extraction_results.pt'
            torch.save(test_results, output_path)
            print(f"  💾 测试结果已保存到: {output_path}")
            
        except Exception as e:
            print(f"  ⚠️  保存测试结果失败: {e}")
    
    def test_performance_benchmark(self) -> bool:
        """
        测试5: 性能基准测试
        """
        print("\n🔍 测试5: 性能基准测试...")
        
        try:
            print("  ⏱️  测试不同样本数量的性能...")
            
            sample_sizes = [1, 3, 5, 10]
            performance_results = {}
            
            for sample_size in sample_sizes:
                if sample_size > len(list(self.test_dataloader.dataset)):
                    continue
                
                print(f"    📊 测试 {sample_size} 个样本...")
                
                start_time = time.time()
                
                # 创建新的特征提取器
                extractor = FeatureExtractor(
                    model=self.model,
                    dataloader=self.test_dataloader,
                    prunable_module_names=self.prunable_module_names[:3]  # 只测试前3个模块
                )
                extractor.max_samples = sample_size
                
                # 执行特征提取
                state_tensor = extractor.extract()
                
                extraction_time = time.time() - start_time
                performance_results[sample_size] = {
                    'time': extraction_time,
                    'tensor_shape': state_tensor.shape
                }
                
                print(f"      ⏱️  耗时: {extraction_time:.2f}秒")
            
            print("  ✅ 性能基准测试完成!")
            
            # 显示性能总结
            print("     性能总结:")
            for size, result in performance_results.items():
                print(f"       {size} 样本: {result['time']:.2f}秒")
            
            return True
            
        except Exception as e:
            print(f"  ❌ 性能测试失败: {e}")
            traceback.print_exc()
            return False
    
    def run_all_tests(self) -> bool:
        """
        运行所有测试
        """
        print("🧪 开始 FeatureExtractor 完整测试流程")
        print("=" * 60)
        
        tests = [
            ("模型加载", self.test_model_loading),
            ("模块检测", self.test_module_detection),
            ("数据准备", self.test_data_preparation),
            ("特征提取", self.test_feature_extraction),
            ("性能基准", self.test_performance_benchmark),
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed_tests += 1
                    print(f"✅ {test_name} 测试通过")
                else:
                    print(f"❌ {test_name} 测试失败")
            except Exception as e:
                print(f"❌ {test_name} 测试异常: {e}")
        
        print("\n" + "=" * 60)
        print(f"🏁 测试完成! 通过率: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
        
        if passed_tests == total_tests:
            print("🎉 所有测试通过！FeatureExtractor 工作正常")
            print("💡 您现在可以安全地将其集成到训练流程中")
            return True
        else:
            print("⚠️  部分测试失败，建议检查和修复问题")
            return False


def main():
    """
    主函数
    """
    print("🚀 FeatureExtractor 测试程序启动")
    print(f"📅 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"🎮 检测到CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("💻 使用CPU模式")
    
    print()
    
    try:
        # 创建测试器并运行测试
        tester = FeatureExtractorTester()
        success = tester.run_all_tests()
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if success:
            print("\n🎊 测试成功完成！")
            sys.exit(0)
        else:
            print("\n❌ 测试失败，请检查错误信息")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 测试程序异常: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
