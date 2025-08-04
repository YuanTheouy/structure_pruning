#!/usr/bin/env python3
"""
测试轻量级评估系统
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.lightweight_eval import LightweightEvaluator
import torch

class MockModel:
    """模拟模型用于测试"""
    def __init__(self):
        self.device = "cpu"
        
    def eval(self):
        pass
        
    def __call__(self, **kwargs):
        # 模拟模型输出
        batch_size = kwargs['input_ids'].shape[0]
        seq_len = kwargs['input_ids'].shape[1]
        vocab_size = 50265  # OPT的词汇表大小
        
        # 返回随机logits
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        class MockOutput:
            def __init__(self, logits):
                self.logits = logits
                
        return MockOutput(logits)

class MockTokenizer:
    """模拟tokenizer用于测试"""
    def __init__(self):
        self.eos_token = "</s>"
        self.pad_token = "</s>"
        
    def __call__(self, text, **kwargs):
        # 简单的模拟tokenization
        # 假设每个单词对应一个token
        words = text.split()
        input_ids = list(range(1, len(words) + 1))  # 简单的token ids
        
        # 截断或填充到指定长度
        max_length = kwargs.get('max_length', 512)
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        
        return {
            'input_ids': torch.tensor([input_ids])
        }

def test_evaluator():
    """测试评估器的各个方法"""
    print("=== 测试轻量级评估系统 ===")
    
    # 创建模拟的模型和tokenizer
    mock_model = MockModel()
    mock_tokenizer = MockTokenizer()
    
    # 创建评估器
    evaluator = LightweightEvaluator(mock_model, mock_tokenizer, device="cpu")
    
    print("\n1. 测试单个任务评估方法是否可用:")
    
    # 检查所有评估方法是否可调用
    methods = [
        ('BoolQ', 'evaluate_boolq'),
        ('PIQA', 'evaluate_piqa'), 
        ('HellaSwag', 'evaluate_hellaswag'),
        ('WinoGrande', 'evaluate_winogrande'),
        ('ARC-Easy', 'evaluate_arc_easy'),
        ('ARC-Challenge', 'evaluate_arc_challenge'),
        ('OBQA', 'evaluate_obqa')
    ]
    
    for task_name, method_name in methods:
        if hasattr(evaluator, method_name):
            print(f"   ✓ {task_name} evaluation method available")
        else:
            print(f"   ✗ {task_name} evaluation method missing")
    
    print("\n2. 测试单个任务评估 (使用极小样本):")
    
    try:
        # 测试单个任务 - PIQA
        print("   Testing PIQA...")
        piqa_results = evaluator.evaluate_piqa(num_samples=2)
        print(f"   PIQA results: {piqa_results}")
        
        # 测试单个任务 - BoolQ
        print("   Testing BoolQ...")
        boolq_results = evaluator.evaluate_boolq(num_samples=2)
        print(f"   BoolQ results: {boolq_results}")
        
    except Exception as e:
        print(f"   Error in individual task testing: {e}")
    
    print("\n3. 测试综合评估方法:")
    
    try:
        print("   Running evaluate_all with 1 sample per task...")
        all_results = evaluator.evaluate_all(num_samples_per_task=1)
        
        print(f"\n   Complete evaluation results:")
        for key, value in all_results.items():
            print(f"     {key}: {value:.4f}")
            
        print(f"\n   ✓ 评估系统测试成功!")
        print(f"   ✓ 支持的任务数量: {len([k for k in all_results.keys() if k != 'avg_score'])}")
        
        expected_tasks = ['boolq_acc', 'piqa_acc', 'hellaswag_acc', 'winogrande_acc', 
                         'arc_easy_acc', 'arc_challenge_acc', 'obqa_acc', 'avg_score']
        
        print(f"\n   预期的任务结果键: {expected_tasks}")
        print(f"   实际的任务结果键: {list(all_results.keys())}")
        
        missing_keys = set(expected_tasks) - set(all_results.keys())
        if missing_keys:
            print(f"   ⚠️  缺失的任务结果: {missing_keys}")
        else:
            print(f"   ✓ 所有预期任务都有结果")
            
    except Exception as e:
        print(f"   ✗ Error in comprehensive evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_evaluator()
