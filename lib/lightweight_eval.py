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
    
    def evaluate_boolq(self, num_samples: int = 100) -> Dict[str, float]:
        """评估BoolQ任务 - 布尔问答"""
        # 简化的BoolQ样本
        samples = [
            {
                "passage": "The Pacific Ocean is the largest ocean on Earth.",
                "question": "Is the Pacific Ocean the largest ocean?",
                "answer": True
            },
            {
                "passage": "Cats are known for their ability to fly.",
                "question": "Can cats fly?",
                "answer": False
            },
            {
                "passage": "Water boils at 100 degrees Celsius at sea level.",
                "question": "Does water boil at 100 degrees Celsius?",
                "answer": True
            },
            {
                "passage": "The sun rises in the west and sets in the east.",
                "question": "Does the sun rise in the west?",
                "answer": False
            },
            {
                "passage": "Python is a programming language.",
                "question": "Is Python a programming language?",
                "answer": True
            }
        ]
        
        # 重复样本以达到所需数量
        while len(samples) < num_samples:
            samples.extend(samples[:min(len(samples), num_samples - len(samples))])
        samples = samples[:num_samples]
        
        correct = 0
        total = 0
        
        print(f"Evaluating BoolQ with {len(samples)} samples...")
        
        for sample in tqdm(samples, desc="BoolQ"):
            try:
                passage = sample['passage']
                question = sample['question']
                
                # 构建提示
                prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
                
                # 计算两个选项的概率
                true_text = f"{prompt} True"
                false_text = f"{prompt} False"
                
                prob_true = self._get_text_probability(true_text)
                prob_false = self._get_text_probability(false_text)
                
                predicted = prob_true > prob_false
                if predicted == sample['answer']:
                    correct += 1
                total += 1
                
            except Exception as e:
                print(f"Error in BoolQ evaluation: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0.0
        return {"boolq_acc": accuracy}

    def evaluate_arc_easy(self, num_samples: int = 100) -> Dict[str, float]:
        """评估ARC-Easy任务 - 简单科学推理"""
        # 简化的ARC-Easy样本
        samples = [
            {
                "question": "What happens to water when it is heated to 100°C?",
                "choices": ["It freezes", "It boils", "It becomes solid", "It disappears"],
                "answer": 1
            },
            {
                "question": "Which planet is closest to the Sun?",
                "choices": ["Earth", "Venus", "Mercury", "Mars"],
                "answer": 2
            },
            {
                "question": "What do plants need to make their own food?",
                "choices": ["Only water", "Only sunlight", "Sunlight and water", "Only soil"],
                "answer": 2
            },
            {
                "question": "What is the main gas in the air we breathe?",
                "choices": ["Oxygen", "Carbon dioxide", "Nitrogen", "Hydrogen"],
                "answer": 2
            },
            {
                "question": "What happens when you mix red and blue paint?",
                "choices": ["Green", "Purple", "Yellow", "Orange"],
                "answer": 1
            }
        ]
        
        return self._evaluate_multiple_choice("ARC-Easy", samples, num_samples)

    def evaluate_arc_challenge(self, num_samples: int = 50) -> Dict[str, float]:
        """评估ARC-Challenge任务 - 困难科学推理"""
        # 简化的ARC-Challenge样本
        samples = [
            {
                "question": "Which property of a mineral can be determined just by looking at it?",
                "choices": ["hardness", "color", "melting point", "density"],
                "answer": 1
            },
            {
                "question": "A student is given a mixture of sand and salt. Which would be the best way to separate the mixture?",
                "choices": ["heating the mixture", "adding water and filtering", "using a magnet", "shaking the mixture"],
                "answer": 1
            },
            {
                "question": "What causes the phases of the Moon?",
                "choices": ["Earth's shadow on the Moon", "The Moon's rotation", "The Moon's position relative to Earth and Sun", "Clouds covering the Moon"],
                "answer": 2
            }
        ]
        
        return self._evaluate_multiple_choice("ARC-Challenge", samples, num_samples)

    def evaluate_obqa(self, num_samples: int = 100) -> Dict[str, float]:
        """评估OBQA任务 - 开放书问答"""
        # 简化的OBQA样本
        samples = [
            {
                "question": "The sun is responsible for",
                "choices": ["puppies learning new tricks", "children growing up and getting old", "flowers wilting in a garden", "plants and flowers blooming"],
                "answer": 3
            },
            {
                "question": "When food is reduced in the stomach",
                "choices": ["the mind needs time to digest", "take a second to digest what I said", "nutrients are being deconstructed", "reader's digest is a magazine"],
                "answer": 2
            },
            {
                "question": "You can make a telescope with a",
                "choices": ["stained glass window", "broken mirror", "paper towel tube", "kaleidoscope"],
                "answer": 2
            },
            {
                "question": "A thing's position is not altered when",
                "choices": ["it's moving", "it's bent", "it's in a state of inaction", "it's falling"],
                "answer": 2
            }
        ]
        
        return self._evaluate_multiple_choice("OBQA", samples, num_samples)

    def _evaluate_multiple_choice(self, task_name: str, samples: List[Dict], num_samples: int) -> Dict[str, float]:
        """通用的多选题评估方法"""
        # 重复样本以达到所需数量
        while len(samples) < num_samples:
            samples.extend(samples[:min(len(samples), num_samples - len(samples))])
        samples = samples[:num_samples]
        
        correct = 0
        total = 0
        
        print(f"Evaluating {task_name} with {len(samples)} samples...")
        
        for sample in tqdm(samples, desc=task_name):
            try:
                question = sample['question']
                choices = sample['choices']
                
                # 计算每个选项的概率
                probs = []
                for choice in choices:
                    text = f"Question: {question}\nAnswer: {choice}"
                    prob = self._get_text_probability(text)
                    probs.append(prob)
                
                predicted = np.argmax(probs)
                if predicted == sample['answer']:
                    correct += 1
                total += 1
                
            except Exception as e:
                print(f"Error in {task_name} evaluation: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0.0
        task_key = task_name.lower().replace("-", "_") + "_acc"
        return {task_key: accuracy}

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
        """评估所有支持的下游任务"""
        print("Starting comprehensive lightweight downstream task evaluation...")
        print("Tasks: BoolQ, PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA")
        
        results = {}
        
        # 评估任务列表及其样本数量
        tasks = [
            ("BoolQ", self.evaluate_boolq, num_samples_per_task),
            ("PIQA", self.evaluate_piqa, num_samples_per_task),
            ("HellaSwag", self.evaluate_hellaswag, num_samples_per_task),
            ("WinoGrande", self.evaluate_winogrande, max(1, num_samples_per_task // 2)),
            ("ARC-Easy", self.evaluate_arc_easy, num_samples_per_task),
            ("ARC-Challenge", self.evaluate_arc_challenge, max(1, num_samples_per_task // 2)),
            ("OBQA", self.evaluate_obqa, num_samples_per_task)
        ]
        
        for task_name, eval_func, sample_count in tasks:
            try:
                print(f"\n=> Evaluating {task_name}...")
                task_results = eval_func(sample_count)
                results.update(task_results)
                
                # 显示单个任务结果
                for key, value in task_results.items():
                    print(f"   {key}: {value:.4f}")
                    
            except Exception as e:
                print(f"{task_name} evaluation failed: {e}")
                # 为失败的任务添加默认值
                task_key = task_name.lower().replace("-", "_") + "_acc"
                results[task_key] = 0.0
        
        # 计算平均分
        accuracy_scores = [v for k, v in results.items() if k.endswith('_acc')]
        if accuracy_scores:
            results["avg_score"] = sum(accuracy_scores) / len(accuracy_scores)
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
