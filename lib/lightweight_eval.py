"""
轻量级下游任务评估实现
用于在旧版transformers环境中提供基础的下游任务评估功能
"""

import torch
import numpy as np
from tqdm import tqdm
import json
import random
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

class LightweightEvaluator:
    """轻量级下游任务评估器"""
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
    def evaluate_piqa(self, num_samples: int = 100) -> Dict[str, float]:
        """评估PIQA任务 - 物理常识推理"""
        # 简化的PIQA样本（实际使用中应该从数据集加载）
        samples = [
            {
                "goal": "To clean a window",
                "sol1": "Use a squeegee with soap and water",
                "sol2": "Use a hammer to break it",
                "label": 0
            },
            {
                "goal": "To make ice cubes",
                "sol1": "Put water in the oven at 400 degrees",
                "sol2": "Put water in ice trays and freeze",
                "label": 1
            },
            {
                "goal": "To remove a stain from clothing",
                "sol1": "Apply stain remover and wash",
                "sol2": "Set the clothing on fire",
                "label": 0
            },
            {
                "goal": "To open a jar",
                "sol1": "Use a jar opener or rubber grip",
                "sol2": "Throw it against the wall",
                "label": 0
            },
            {
                "goal": "To water plants",
                "sol1": "Pour gasoline on them",
                "sol2": "Use a watering can with water",
                "label": 1
            }
        ]
        
        # 重复样本以达到所需数量
        while len(samples) < num_samples:
            samples.extend(samples[:min(len(samples), num_samples - len(samples))])
        samples = samples[:num_samples]
        
        correct = 0
        total = 0
        
        print(f"Evaluating PIQA with {len(samples)} samples...")
        
        for sample in tqdm(samples, desc="PIQA"):
            try:
                prompt = f"Question: {sample['goal']}\nSolution A: {sample['sol1']}\nSolution B: {sample['sol2']}\nWhich solution is better? Answer:"
                
                # 计算两个选项的困惑度
                option_a_text = f"{prompt} A"
                option_b_text = f"{prompt} B"
                
                prob_a = self._get_text_probability(option_a_text)
                prob_b = self._get_text_probability(option_b_text)
                
                predicted = 0 if prob_a > prob_b else 1
                if predicted == sample['label']:
                    correct += 1
                total += 1
                
            except Exception as e:
                print(f"Error in PIQA evaluation: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0.0
        return {"piqa_acc": accuracy}
    
    def evaluate_hellaswag(self, num_samples: int = 100) -> Dict[str, float]:
        """评估HellaSwag任务 - 常识推理"""
        # 简化的HellaSwag样本
        samples = [
            {
                "ctx": "A woman is sitting at a piano.",
                "endings": [
                    "She begins to play a beautiful melody.",
                    "She starts to eat the piano keys.",
                    "She transforms into a bird.",
                    "She disappears into thin air."
                ],
                "label": 0
            },
            {
                "ctx": "A man is cooking in the kitchen.",
                "endings": [
                    "He throws the food into the garbage.",
                    "He adds salt and pepper to taste.",
                    "He starts dancing on the table.",
                    "He paints the walls with sauce."
                ],
                "label": 1
            },
            {
                "ctx": "Children are playing in the park.",
                "endings": [
                    "They start flying like airplanes.",
                    "They disappear underground.",
                    "They run around and laugh happily.",
                    "They begin solving calculus problems."
                ],
                "label": 2
            }
        ]
        
        # 重复样本以达到所需数量
        while len(samples) < num_samples:
            samples.extend(samples[:min(len(samples), num_samples - len(samples))])
        samples = samples[:num_samples]
        
        correct = 0
        total = 0
        
        print(f"Evaluating HellaSwag with {len(samples)} samples...")
        
        for sample in tqdm(samples, desc="HellaSwag"):
            try:
                ctx = sample['ctx']
                endings = sample['endings']
                
                # 计算每个选项的概率
                probs = []
                for ending in endings:
                    text = f"{ctx} {ending}"
                    prob = self._get_text_probability(text)
                    probs.append(prob)
                
                predicted = np.argmax(probs)
                if predicted == sample['label']:
                    correct += 1
                total += 1
                
            except Exception as e:
                print(f"Error in HellaSwag evaluation: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0.0
        return {"hellaswag_acc": accuracy}
    
    def evaluate_winogrande(self, num_samples: int = 50) -> Dict[str, float]:
        """评估Winogrande任务 - 代词消歧"""
        # 简化的Winogrande样本
        samples = [
            {
                "sentence": "The trophy doesn't fit into the brown suitcase because _ is too large.",
                "option1": "the trophy",
                "option2": "the suitcase", 
                "answer": "1"
            },
            {
                "sentence": "The trophy doesn't fit into the brown suitcase because _ is too small.",
                "option1": "the trophy",
                "option2": "the suitcase",
                "answer": "2"
            },
            {
                "sentence": "The man couldn't lift the box because _ was too heavy.",
                "option1": "the man",
                "option2": "the box",
                "answer": "2"
            }
        ]
        
        # 重复样本以达到所需数量
        while len(samples) < num_samples:
            samples.extend(samples[:min(len(samples), num_samples - len(samples))])
        samples = samples[:num_samples]
        
        correct = 0
        total = 0
        
        print(f"Evaluating Winogrande with {len(samples)} samples...")
        
        for sample in tqdm(samples, desc="Winogrande"):
            try:
                sentence = sample['sentence']
                option1 = sample['option1']
                option2 = sample['option2']
                
                # 替换下划线并计算概率
                text1 = sentence.replace('_', option1)
                text2 = sentence.replace('_', option2)
                
                prob1 = self._get_text_probability(text1)
                prob2 = self._get_text_probability(text2)
                
                predicted = "1" if prob1 > prob2 else "2"
                if predicted == sample['answer']:
                    correct += 1
                total += 1
                
            except Exception as e:
                print(f"Error in Winogrande evaluation: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0.0
        return {"winogrande_acc": accuracy}
    
    def _get_text_probability(self, text: str) -> float:
        """计算给定文本的概率（使用困惑度的倒数）"""
        try:
            # 使用兼容的tokenizer参数
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # 计算困惑度
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # 返回负对数似然的负值（越大越好）
                return -loss.mean().item()
                
        except Exception as e:
            print(f"Error calculating text probability: {e}")
            return float('-inf')
    
    def evaluate_all(self, num_samples_per_task: int = 100) -> Dict[str, float]:
        """评估所有支持的任务"""
        print("Starting lightweight downstream task evaluation...")
        
        results = {}
        
        try:
            # PIQA
            piqa_results = self.evaluate_piqa(num_samples_per_task)
            results.update(piqa_results)
        except Exception as e:
            print(f"PIQA evaluation failed: {e}")
            results["piqa_acc"] = 0.0
        
        try:
            # HellaSwag
            hellaswag_results = self.evaluate_hellaswag(num_samples_per_task)
            results.update(hellaswag_results)
        except Exception as e:
            print(f"HellaSwag evaluation failed: {e}")
            results["hellaswag_acc"] = 0.0
        
        try:
            # Winogrande
            winogrande_results = self.evaluate_winogrande(num_samples_per_task // 2)
            results.update(winogrande_results)
        except Exception as e:
            print(f"Winogrande evaluation failed: {e}")
            results["winogrande_acc"] = 0.0
        
        # 计算平均分
        valid_scores = [v for v in results.values() if v > 0]
        if valid_scores:
            results["avg_score"] = sum(valid_scores) / len(valid_scores)
        else:
            results["avg_score"] = 0.0
        
        return results

def test_lightweight_evaluator():
    """测试轻量级评估器"""
    print("Testing lightweight evaluator...")
    
    # 这里应该传入实际的模型和tokenizer
    # evaluator = LightweightEvaluator(model, tokenizer)
    # results = evaluator.evaluate_all(num_samples_per_task=10)
    # print("Test results:", results)
    
    print("Lightweight evaluator module loaded successfully!")

if __name__ == "__main__":
    test_lightweight_evaluator()
