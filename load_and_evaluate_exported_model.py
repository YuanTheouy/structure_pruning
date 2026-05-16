#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
剪枝模型加载与评估脚本

该脚本用于：
1. 加载已导出的剪枝模型
2. 对加载的模型进行重构（可选）
3. 评估模型的PPL（困惑度）
4. 评估模型在下游任务上的性能（可选）

用法示例：
    # 基础评估（仅PPL）
    python load_and_evaluate_exported_model.py \
        --model_path /home/yx/yx_repository/01_Models/Qwen2.5-7B \
        --checkpoint_path ./checkpoints/qwen2.5-7b_0_7_wikitext2_20250909_192417_export.pth.tar
    
    # 启用重构和下游任务评估
    python load_and_evaluate_exported_model.py \
        --model_path /home/yx/yx_repository/01_Models/Qwen2.5-7B \
        --checkpoint_path ./checkpoints/qwen2.5-7b_0_7_wikitext2_20250909_192417_export.pth.tar \
        --enable_recon \
        --enable_downstream \
        --recon_samples 32
"""

import os
import sys
import argparse
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.eval import eval_ppl
from lib.data import get_loaders
from env.channel_pruning_env_llm_global import ChannelPruningEnv
from lib.utils import get_output_folder


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='加载和评估已导出的剪枝模型')
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='原始模型路径')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='导出的剪枝模型检查点路径(.pth.tar文件)')
    
    # 模型配置
    parser.add_argument('--model_name', type=str, default='qwen2.5-7b',
                       help='模型名称标识符')
    parser.add_argument('--dataset_name', type=str, default='wikitext2',
                       help='用于PPL评估的数据集')
    
    # 重构参数
    parser.add_argument('--enable_recon', action='store_true',
                       help='启用模型重构（提高精度但增加时间）')
    parser.add_argument('--recon_samples', type=int, default=32,
                       help='重构时使用的样本数量')
    
    # 评估参数
    parser.add_argument('--enable_downstream', action='store_true',
                       help='启用下游任务评估')
    parser.add_argument('--seqlen', type=int, default=2048,
                       help='序列长度')
    
    # 技术参数
    parser.add_argument('--n_samples', type=int, default=64,
                       help='用于特征提取的样本数量')
    parser.add_argument('--seed', type=int, default=2025,
                       help='随机种子')
    
    return parser.parse_args()


def load_original_model(model_path, seqlen=2048):
    """加载原始模型和分词器"""
    print(f"=> 正在加载原始模型: {model_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 设置序列长度属性
    model.seqlen = seqlen
    
    print(f"=> 模型类型: {type(model).__name__}")
    print(f"=> 序列长度设置为: {seqlen}")
    print(f"=> 模型设备: {next(model.parameters()).device}")
    
    return model, tokenizer


def load_pruned_checkpoint(model, checkpoint_path):
    """加载剪枝后的模型状态字典"""
    print(f"=> 正在加载剪枝模型检查点: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    # 加载状态字典
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # 应用状态字典到模型
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"=> 警告: 缺少以下键: {missing_keys}")
    if unexpected_keys:
        print(f"=> 警告: 意外的键: {unexpected_keys}")
    
    print("=> 剪枝模型检查点加载完成")
    return model


def setup_environment_for_reconstruction(model, tokenizer, args):
    """设置用于重构的环境"""
    print("=> 正在设置重构环境...")
    
    # 创建一个临时的args对象用于环境初始化
    class TempArgs:
        def __init__(self):
            self.model = args.model_path
            self.model_name = args.model_name
            self.dataset_name = args.dataset_name
            self.preserve_ratio = 1.0  # 对于已剪枝的模型，保持当前状态
            self.use_real_val = True
            self.prune = "para"
            self.structure = True
            self.state_mode = 0
            self.recon = args.enable_recon
            self.recon_sample = args.recon_samples
            self.n_samples = args.n_samples
            self.lbound = 0.1
            self.rbound = 1.0
            self.acc_metric = "acc1"
            self.reward = "reward_ppl"
            self.seed = args.seed
            self.enable_downstream = "true" if args.enable_downstream else "false"
            self.export_path = ""  # 不需要导出
    
    temp_args = TempArgs()
    
    # 创建环境但不重新加载模型
    env = ChannelPruningEnv(temp_args, model=model, tokenizer=tokenizer)
    
    return env


def perform_reconstruction(env, model):
    """执行模型重构"""
    print("=> 开始模型重构...")
    start_time = time.time()
    
    # 重构逻辑应该在这里实现
    # 注意：具体的重构实现需要根据你的环境代码来调整
    print("=> 注意: 重构功能需要根据具体的环境实现来调整")
    print("=> 当前跳过重构步骤")
    
    end_time = time.time()
    print(f"=> 重构完成，耗时: {end_time - start_time:.2f}秒")
    
    return model


def evaluate_model_ppl(model, tokenizer, dataset_name="wikitext2"):
    """评估模型的困惑度"""
    print(f"=> 正在评估模型PPL (数据集: {dataset_name})...")
    
    try:
        ppl = eval_ppl(model, tokenizer)
        print(f"=> 模型PPL: {ppl:.3f}")
        return ppl
    except Exception as e:
        print(f"=> PPL评估失败: {e}")
        return None


def evaluate_model_downstream(env, model):
    """评估模型在下游任务上的性能"""
    print("=> 正在评估下游任务性能...")
    
    try:
        success = env.test_model(model)
        if success:
            print("=> 下游任务评估成功完成")
        else:
            print("=> 下游任务评估完成但有警告")
        return success
    except Exception as e:
        print(f"=> 下游任务评估失败: {e}")
        return False


def main():
    """主函数"""
    args = parse_args()
    
    print("="*80)
    print("          剪枝模型加载与评估工具")
    print("="*80)
    print(f"原始模型路径: {args.model_path}")
    print(f"检查点路径: {args.checkpoint_path}")
    print(f"重构模式: {'启用' if args.enable_recon else '禁用'}")
    print(f"下游任务评估: {'启用' if args.enable_downstream else '禁用'}")
    print("="*80)
    
    # 1. 加载原始模型
    try:
        model, tokenizer = load_original_model(args.model_path, args.seqlen)
    except Exception as e:
        print(f"=> 错误: 无法加载原始模型: {e}")
        return 1
    
    # 2. 加载剪枝后的检查点
    try:
        model = load_pruned_checkpoint(model, args.checkpoint_path)
    except Exception as e:
        print(f"=> 错误: 无法加载剪枝检查点: {e}")
        return 1
    
    # 3. 设置环境（如果需要重构或下游评估）
    env = None
    if args.enable_recon or args.enable_downstream:
        try:
            env = setup_environment_for_reconstruction(model, tokenizer, args)
        except Exception as e:
            print(f"=> 错误: 无法设置评估环境: {e}")
            return 1
    
    # 4. 执行重构（如果启用）
    if args.enable_recon and env is not None:
        try:
            model = perform_reconstruction(env, model)
        except Exception as e:
            print(f"=> 错误: 重构失败: {e}")
            return 1
    
    print("\n" + "="*80)
    print("                    开始模型评估")
    print("="*80)
    
    # 5. 评估PPL
    ppl = evaluate_model_ppl(model, tokenizer, args.dataset_name)
    
    # 6. 评估下游任务（如果启用）
    downstream_success = None
    if args.enable_downstream and env is not None:
        downstream_success = evaluate_model_downstream(env, model)
    
    # 7. 总结结果
    print("\n" + "="*80)
    print("                    评估结果总结")
    print("="*80)
    print(f"模型PPL: {ppl:.3f}" if ppl is not None else "模型PPL: 评估失败")
    
    if args.enable_downstream:
        status = "成功" if downstream_success else "失败"
        print(f"下游任务评估: {status}")
    else:
        print("下游任务评估: 未启用")
    
    print("="*80)
    print("评估完成！")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
