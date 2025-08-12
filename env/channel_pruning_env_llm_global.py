import time
import random
import torch
import torch.nn as nn
from torch.nn import functional as f
from lib.utils import AverageMeter, accuracy, prGreen
from lib.arch import get_layers, get_mha_proj, get_ffn2, get_ffn1, get_mha,  get_down, get_up, get_gate
from lib.layerwrapper import WrappedGPT
from lib.eval import eval_ppl
from lib.data_utils import DataSaverHook, StopForwardException
from lib.data import get_loaders
from lib.Ridge import Ridge_Regression
from copy import deepcopy



from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
# 启用下游任务评估需要的导入 - 使用新版lm-eval-harness
# 注释掉旧版本导入
# from lib.lm_eval.evaluator import evaluate, make_table  
# from lib.lm_eval.tasks import get_task_dict, ALL_TASKS
# from lib.lm_eval.utils import pattern_match
# from lib.lm_eval.models import get_model

# 新版本lm-eval-harness导入 - 智能条件化导入
LMEVAL_AVAILABLE = False
LIGHTWEIGHT_EVAL_AVAILABLE = False
lm_eval = None
evaluator = None
HFLM = None
LightweightEvaluator = None

def _check_transformers_compatibility():
    """检查transformers版本兼容性"""
    try:
        import transformers
        version = transformers.__version__
        major, minor = map(int, version.split('.')[:2])
        return major > 4 or (major == 4 and minor >= 40)
    except:
        return False

def _load_lmeval():
    """智能加载评估框架 - 优先使用lm-eval-harness，回退到轻量级实现"""
    global LMEVAL_AVAILABLE, LIGHTWEIGHT_EVAL_AVAILABLE, lm_eval, evaluator, HFLM, LightweightEvaluator
    
    # 首先尝试加载完整的lm-eval-harness
    try:
        import lm_eval as _lm_eval
        from lm_eval import evaluator as _evaluator
        from lm_eval.models.huggingface import HFLM as _HFLM
        
        lm_eval = _lm_eval
        evaluator = _evaluator
        HFLM = _HFLM
        LMEVAL_AVAILABLE = True
        print("=> Using full lm-eval-harness framework")
        return "full"
    # except ImportError:
    #     # 静默处理ImportError，直接尝试轻量级实现
    #     pass
    except Exception:
        # 静默处理其他兼容性问题
        # pass
        import traceback
        print("\n" + "="*60)
        print("=> [DEBUG] 'lm-eval-harness' import failed. Printing full traceback:")
        traceback.print_exc()
        print("="*60 + "\n")
        # 保持pass，让程序继续回退到轻量级实现，避免程序崩溃
    
    # 回退到轻量级评估实现
    try:
        from lib.lightweight_eval import LightweightEvaluator as _LightweightEvaluator
        LightweightEvaluator = _LightweightEvaluator
        LIGHTWEIGHT_EVAL_AVAILABLE = True
        print("=> Using lightweight evaluation implementation")
        return "lightweight"
    except ImportError:
        print("=> Continuing without downstream task evaluation")
        return "none"
from sklearn.metrics import pairwise_distances
from lib.linalg import lsmr_cupy_solver
from lib.mac import mac_per_head, mac_per_neuron, get_layer_param, get_norm_param
from scipy.spatial import distance
# import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoConfig, OPTForCausalLM
# from sklearn.linear_model import Ridge
from env.rewards import *
import math
from tqdm import tqdm

import numpy as np
import cupy as cp

import sys  # 导入sys模块
import os
sys.setrecursionlimit(10000)

os.environ['TOKENIZERS_PARALLELISM']="false"
# 注释掉离线模式设置以允许下游任务评估下载数据集
# os.environ['HF_DATASETS_OFFLINE']="1"
# os.environ['TRANSFORMERS_OFFLINE']="1"

# class Ridge:
#     def __init__(self, alpha=1, fit_intercept=False):
#         self.weight = None
#         self.lambda_ = alpha
#         self.fit_bias = fit_intercept
        
#     def fit(self, X, y):
#         torch.cuda.empty_cache()
#         if self.fit_bias:
#             X = cp.c_[cp.ones(X.shape[0]), X]
#         A = self.lambda_ * cp.eye(X.shape[1])
#         X = cp.asarray(X.cpu().numpy())
#         y = cp.asarray(y.cpu().numpy())
#         mat = X.T @ X + A
#         pseudo_inverse = cp.linalg.inv(mat) @ X.T
#         self.coef_ = pseudo_inverse @ y
        
#     def predict(self, X):
#         if self.fit_bias:
#             X = cp.c_[cp.ones(X.shape[0]), X]
#         return cp.matmul(X, self.weight)


class ChannelPruningEnv:
    """
    Env for channel pruning search
    """
    def __init__(self, model, data, preserve_ratio, args, n_data_worker=4,
                 batch_size=256, export_model=False, use_new_input=False):

        self.args = args
        self.model_path = args.model
        self._get_model()

        # 简单设备分配 - 遵循CUDA_VISIBLE_DEVICES设置
        if torch.cuda.is_available():
            # 获取模型实际所在的设备
            model_device = next(self.model.parameters()).device
            self.device = model_device
            print(f"=> Model device: {self.device}")
        else:
            self.device = torch.device("cpu")
            print(f"=> Using CPU device (CUDA not available)")
            
        self.dataset = args.dataset_name
        self.n_data_worker = n_data_worker
        self.batch_size = batch_size
        self.data_type = data
        self.preserve_ratio = preserve_ratio

        ## llm
        self.num_hidden_layers = self.model.config.num_hidden_layers
        self.num_attention_heads = self.model.config.num_attention_heads
        self.num_key_value_heads = self.num_attention_heads if 'opt' in self.model.config.model_type else self.model.config.num_key_value_heads
        self.intermediate_size = self.model.config.ffn_dim if 'opt' in self.model.config.model_type else self.model.config.intermediate_size
        self.hidden_size = self.model.config.hidden_size
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)

        self.lbound = args.lbound
        self.rbound = args.rbound
        self.use_real_val = args.use_real_val
        self.n_samples = args.n_samples
        self.channel_round = args.channel_round
        self.acc_metric = args.acc_metric
        self.recon = args.recon
        self.recon_prune = args.recon

        self.export_model = export_model
        self.use_new_input = use_new_input

        # 根据状态模式设置状态维度
        if self.use_new_input:
            # 特征提取状态模式：所有模块特征拼接 (48个模块 × 8维特征 = 384维)
            self.state_dim = 48 * 8  # 48个模块，每个8维特征
        else:
            # 全局剪枝率状态模式：1维
            self.state_dim = 1
        
        print(f"=> Environment state dimension: {self.state_dim} (use_new_input: {self.use_new_input})")

        # prepare data
        self._init_data()
  
        # extract information for preparing
        self._extract_layer_information()

        # build reward
        self.reset()  # restore weight
        self.org_ppl = self._validate(self.model)  # 计算真实的原始模型PPL
        print('=> original ppl: {:.3f}'.format(self.org_ppl))
        self.org_para = sum(self.param_list)
        print('=> Params:')
        print(self.param_list)
        print('=> original weight size: {:.4f} M param'.format(self.org_para))
        self.org_flops = sum(self.flops_list)
        print('=> FLOPs:')
        print(self.flops_list)
        print('=> original FLOPs: {:.4f} M'.format(self.org_flops))

        # if self.args.prune == 'para':
        #     self.expected_preserve_computation = self.preserve_ratio * self.org_para
        # elif self.args.prune == 'flops':
        #     self.expected_preserve_computation = self.preserve_ratio * self.org_flops
        # else:
        #     raise NotImplementedError
        self.update_target_ratio(self.preserve_ratio)
        
        # 初始化奖励函数
        self.reward = eval(args.reward)

        self.best_reward = -math.inf
        self.best_strategy = None
        self.best_d_prime_list = None
        
        # 根据状态模式设置状态维度
        if self.use_new_input:
            # 特征提取状态模式：所有模块特征拼接 (48个模块 × 8维特征 = 384维)
            self.state_dim = 48 * 8
        else:
            # 全局剪枝率状态模式：1维
            self.state_dim = 1
            
        print(f'=> 状态维度设置为: {self.state_dim} (use_new_input={self.use_new_input})')
        self.best_reward = -math.inf
        self.best_strategy = None
        self.best_d_prime_list = None
        
        # ... (您之前的 state_dim 设置代码保持不变) ...
        print(f'=> 状态维度设置为: {self.state_dim} (use_new_input={self.use_new_input})')

        # --- 新增代码：在初始化完成时，备份原始模型权重 ---
        print("=> Storing original model weights in memory for efficient reset...")
        self.original_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        print("=> Original weights stored.")
        # --------------------------------------------------

    def update_target_ratio(self, new_preserve_ratio):
        """
        允许外部调用以更新环境的全局目标保留率。
        """
        self.preserve_ratio = new_preserve_ratio
        
        # 重新计算总预算
        if self.args.prune == 'para':
            self.expected_preserve_computation = self.preserve_ratio * self.org_para
        elif self.args.prune == 'flops':
            self.expected_preserve_computation = self.preserve_ratio * self.org_flops
        
        # (可选) 打印日志以确认更新
        print(f"=> Updated target preserve ratio to: {self.preserve_ratio:.4f}")

    def _get_model(self):
        self._get_model_local()
        if "opt" in self.args.model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        elif "llama-3" in self.args.model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        elif "llama" in self.args.model:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path, use_fast=False)


    def _get_model_local(self, verbose=True):
        # 简单的设备映射 - 严格遵循CUDA_VISIBLE_DEVICES设置
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if verbose:
                print(f"=> Detected {gpu_count} visible GPU(s)")
            
            # 简单策略：使用auto让transformers处理，但不覆盖用户的GPU绑定
            device_map = "auto"
            if verbose:
                print(f"=> Using automatic device mapping with {gpu_count} visible GPU(s)")
        else:
            device_map = "cpu"
            if verbose:
                print("=> Using CPU (no GPU available)")
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            cache_dir=self.args.cache_dir,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        
        # 根据模型类型动态设置序列长度
        if "OPT" in self.model.__class__.__name__:
            # OPT模型使用2048序列长度
            self.model.seqlen = 2048
            layers = self.model.model.decoder.layers
        else:
            # Llama等其他模型使用配置中的max_position_embeddings，但限制在合理范围内
            max_seq_len = getattr(self.model.config, 'max_position_embeddings', 2048)
            # 为了避免内存问题，我们将序列长度限制在2048
            self.model.seqlen = min(max_seq_len, 2048)
            layers = self.model.model.layers
            
        if verbose:
            print(f"=> Model type: {self.model.__class__.__name__}")
            print(f"=> Sequence length set to: {self.model.seqlen}")
            print(f"=> Model max_position_embeddings: {getattr(self.model.config, 'max_position_embeddings', 'N/A')}")
            print(f"=> Model has {len(layers)} transformer layers")
        else:
            # 静默模式，只打印关键信息
            print(f"=> Model reset: {self.model.__class__.__name__} with {len(layers)} layers")


    def step(self, action):
        self.action = self._action_wall(action)
        # with torch.no_grad():
        #     if "OPT" in self.model.__class__.__name__:
        #         self.inps, self.outs, self.attention_mask, self.position_ids = self.prepare_calibration_input_opt()
        #     else:
        #         self.inps, self.outs, self.attention_mask, self.position_ids = self.prepare_calibration_input()
        count=0
        total_steps = len(self.action)
        # print(f"=> Starting pruning process with {total_steps} steps...")
        
        # 创建进度条
        pbar = tqdm(total=total_steps, desc="Pruning Progress", 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        start_time = time.time()
        for i, r in enumerate(self.action):
            step_start_time = time.time()
            component = 'Head' if self.head else 'FFN'
            pbar.set_description(f"Layer {self.layer_idx:2d} {component}")
            
            if self.args.resume_path is not None:
                if count<self.args.start:
                    self.recon = False
                    self._prune_step(r)
                elif count == self.args.start:
                    checkpoint = torch.load(self.args.resume_path)
                    self.model.load_state_dict(checkpoint)
                    
                    layer = get_layers(self.model)
                    for i in range(self.layer_idx-1):
                        torch.cuda.empty_cache()
                        for j in range(len(self.recon_inps)):
                            with torch.no_grad():
                                if "OPT" in self.model.__class__.__name__:
                                    self.recon_outs[j] = layer[i](self.recon_inps[j].unsqueeze(0), attention_mask=self.attention_mask)[0]
                                else:
                                    self.recon_outs[j] = layer[i](self.recon_inps[j].unsqueeze(0), attention_mask=self.attention_mask,
                                        position_ids=self.position_ids)[0]
                        self.recon_inps = self.recon_outs

                    self.recon = True
                    self._prune_step(r)
                else:
                    self.recon = True
                    self._prune_step(r)
                count += 1
            else:
                self._prune_step(r)
            
            step_time = time.time() - step_start_time
            pbar.set_postfix({"Ratio": f"{r:.4f}", "Time": f"{step_time:.1f}s"})
            pbar.update(1)
        
        pbar.close()
        total_time = time.time() - start_time
        # print(f"=> Pruning completed in {total_time:.1f}s (avg: {total_time/total_steps:.1f}s/step)")

        assert len(self.action) == self.num_hidden_layers * 2
        # print("=> Calculating final metrics...")
        current_flops = self._cur_flops(self.strategy)
        compress_ratio = current_flops * 1. / self.org_flops
        current_para = self._cur_para(self.strategy)
        para_ratio = current_para * 1. / self.org_para

        # print("=> Validating pruned model (PPL + Downstream tasks)...")
        ppl = self._validate(self.model)
        if ppl is not None and np.isfinite(ppl) and ppl > 0:
            # 1. PPL 是一个有效的正数
            #    我们仍然使用您原来的奖励函数 self.reward()
            reward = self.reward(ppl)
        else:
            # 1. PPL是 nan, inf, 0 或 负数, 说明模型已崩溃
            # 2. 在日志中明确记录这次灾难性事件
            print(f"CRITICAL WARNING: Episode resulted in invalid PPL ({ppl}). Applying large penalty.")
            # 3. 给予一个固定的、巨大的、但有效的惩罚性奖励
            reward = -100.0  # 使用一个确定的坏分数，而不是nan

        # 双重保险：确保最终的 reward 自身不是nan
        if np.isnan(reward) or np.isinf(reward):
            print(f"CRITICAL WARNING: Reward calculation resulted in invalid value. Overriding with penalty.")
            reward = -100.0
        # --- 防火墙结束 ---

        info_set = {'compress_ratio': compress_ratio, 'para_ratio': para_ratio, 'ppl': ppl, 'strategy': self.action.copy()}
        
        # --- 关键修正 ---
        # 在函数的末尾，找到返回 observation 的地方
        if self.use_new_input:
            # 因为状态是静态的，所以下一个状态就是当前状态
            obs = self.state
        else:
            # 旧的逻辑，保持不变
            obs = np.array(self.preserve_ratio, dtype=np.float32)

        if reward > self.best_reward:
            self.best_reward = reward
            self.best_strategy = self.action.copy()
            self.best_d_prime_list = self.d_prime_list.copy()
            prGreen(
                'New best reward: {:.4f}, ppl: {:.4f}, compress: {:.4f}, para: {:.4f}'.format(self.best_reward, ppl,
                                                                                              compress_ratio, para_ratio))
            prGreen('New best policy: {}'.format(self.best_strategy))
            prGreen('New best d primes: {}'.format(self.best_d_prime_list))
            torch.save(self.model.state_dict(), self.export_path)

        done = True

        return obs, reward, done, info_set



    def _prune_step(self, action):
        # Pseudo prune and get the corresponding statistics. The real pruning happens till the end of all pseudo pruning
        idx = 2*self.layer_idx + int(not self.head)
        ratio, d_prime = self.prune(action, self.layer_idx, self.head, idx)

        self.d_prime_list.append(d_prime)
        self.strategy.append(ratio)

        # update to next layer
        if self.head:
            self.head = False
        else:
            self.head = True
            self.layer_idx += 1


    def create_feat_scaleing_attn(self, feat, ind, num):
        torch.cuda.empty_cache()
        num_neurons = ind.shape[0]
        scaling_mat = np.zeros([num, num_neurons])

        feature = torch.stack(feat)
        # feature = torch.mean(feature, dim=0).squeeze().to(self.device)
        feature = feature.reshape(-1, num).to(self.device)

        # 创建子进度条显示Ridge回归进度
        pbar = tqdm(range(scaling_mat.shape[0]), desc="Ridge Regression", leave=False, 
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')

        ridge_count = 0
        for i in pbar:
            if i in ind:  # chosen
                ind_i, = np.where(ind == i)
                assert (len(ind_i) == 1)  # check if only one index is found
                scaling_mat[i, ind_i] = 1
            else:  # not chosen
                # A = feature[:, chosen]
                B = feature[:, i]
                A = feature[:, ind]
                
                # 减少频繁的内存清理
                ridge_count += 1
                if ridge_count % 100 == 0:  # 每100次清理一次
                    torch.cuda.empty_cache()

                linear_cif = Ridge_Regression(A.to(dtype=torch.float32), B.to(dtype=torch.float32), alpha=0.9, fit_intercept=False, device=self.device)
                scale_factor = linear_cif.fit()

                for index, chosen_i in enumerate(ind):
                    scaling_mat[i, index] = scale_factor[index]
        
        # 最后清理一次
        torch.cuda.empty_cache()
        pbar.close()
        return scaling_mat


    def create_feat_scaleing_ffn(self, feat, ind, num, threshold):
        torch.cuda.empty_cache()
        num_neurons = ind.shape[0]
        scaling_mat = np.zeros([num, num_neurons])

        feature = torch.stack(feat)
        # feature = torch.mean(feature, dim=0).squeeze().to(self.device)
        feature = feature.reshape(-1, num).to(self.device)

        # 创建子进度条显示Ridge回归进度
        pbar = tqdm(range(scaling_mat.shape[0]), desc="Ridge Regression", leave=False,
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')

        ridge_count = 0
        for i in pbar:
            if i in ind:  # chosen
                ind_i, = np.where(ind == i)
                assert (len(ind_i) == 1)  # check if only one index is found
                scaling_mat[i, ind_i] = 1
            else:  # not chosen
                B = feature[:, i]
                # A = feature[:, chosen]
                A = feature[:, ind]

                # 减少频繁的内存清理
                ridge_count += 1
                if ridge_count % 100 == 0:  # 每100次清理一次
                    torch.cuda.empty_cache()

                linear_cif = Ridge_Regression(A.to(dtype=torch.float32), B.to(dtype=torch.float32), alpha=0.9, fit_intercept=False, device=self.device)
                scale_factor = linear_cif.fit()

                for index, chosen_i in enumerate(ind):
                    scaling_mat[i, index] = scale_factor[index]
        
        # 最后清理一次
        torch.cuda.empty_cache()
        pbar.close()
        return scaling_mat

    def prune(self, preserve_ratio, idx, head, global_idx):

        def format_rank(x):
            rank = int(np.around(x))
            return max(rank, 1)

        assert (preserve_ratio <= 1.)
        layer = get_layers(self.model)[idx]
        torch.cuda.empty_cache()

        if preserve_ratio == 1:
            if head:
                return preserve_ratio, self.num_key_value_heads
            else:
                if self.recon:
                    torch.cuda.empty_cache()
                    for j in range(len(self.recon_inps)):
                        with torch.no_grad():
                            if "OPT" in self.model.__class__.__name__:
                                self.recon_outs[j] = \
                                layer(self.recon_inps[j].unsqueeze(0), attention_mask=self.attention_mask)[0]
                            else:
                                self.recon_outs[j] = \
                                layer(self.recon_inps[j].unsqueeze(0), attention_mask=self.attention_mask,
                                      position_ids=self.position_ids)[0]
                    self.recon_inps = self.recon_outs

                return preserve_ratio, self.hidden_size

        if head:
            attn = get_mha(self.model, idx)
            target_layer = get_mha_proj(self.model, idx)
            d_prime = format_rank(preserve_ratio * self.num_key_value_heads)
            ratio = d_prime / self.num_key_value_heads
            head_metric = self.A_metric[global_idx]
            head_metric = head_metric.reshape(self.num_key_value_heads, -1)
            head_metric = torch.sum(head_metric, dim=-1)
            sorted_idx = torch.sort(-head_metric)
            preserve_idx = sorted_idx.indices[:d_prime]  # to preserve index
            preserve_idx,_ = torch.sort(preserve_idx)

            mask = torch.zeros_like(head_metric, dtype=bool)
            mask[preserve_idx] = True
            
            # 确保mask在与target_layer相同的设备上
            mask = mask.to(target_layer.weight.device)

            if self.recon:
                torch.cuda.empty_cache()
                data_saver = DataSaverHook(store_input=True, store_output=False, stop_forward=True)
                handles_inputs = target_layer.register_forward_hook(data_saver)
                inputs = []
                for j in range(len(self.recon_inps)):
                    with torch.no_grad():
                        try:
                            if "OPT" in self.model.__class__.__name__:
                                self.recon_outs[j] = layer(self.recon_inps[j].unsqueeze(0), attention_mask=self.attention_mask)[0]
                            else:
                                self.recon_outs[j] = layer(self.recon_inps[j].unsqueeze(0), attention_mask=self.attention_mask,
                                                     position_ids=self.position_ids)[0]
                        except StopForwardException:
                            pass
                        inputs.append(data_saver.input_store[0].detach().to(self.device))
                handles_inputs.remove()


            if "OPT" in self.model.__class__.__name__:
                attn.num_heads = d_prime
                attn.embed_dim = attn.head_dim * d_prime
                
                weight = attn.k_proj.weight.data.reshape(self.num_attention_heads, self.attention_head_size, -1)[mask, :, :]
                attn.k_proj.weight.data = weight.reshape(-1, weight.shape[2])
    
                weight = attn.q_proj.weight.data.reshape(self.num_attention_heads, self.attention_head_size, -1)[mask, :, :]
                attn.q_proj.weight.data = weight.reshape(-1, weight.shape[2])
    
                weight = attn.v_proj.weight.data.reshape(self.num_attention_heads, self.attention_head_size, -1)[mask, :, :]
                attn.v_proj.weight.data = weight.reshape(-1, weight.shape[2])
    
                if attn.k_proj.bias is not None:
                    bias = attn.k_proj.bias.data.reshape(self.num_attention_heads, -1)[mask, :]
                    attn.k_proj.bias.data = bias.reshape(-1)
    
                    bias = attn.q_proj.bias.data.reshape(self.num_attention_heads, -1)[mask, :]
                    attn.q_proj.bias.data = bias.reshape(-1)
    
                    bias = attn.v_proj.bias.data.reshape(self.num_attention_heads, -1)[mask, :]
                    attn.v_proj.bias.data = bias.reshape(-1)
                    
                if self.recon:
                    mask_proj = mask.unsqueeze(0)
                    mask_proj = mask_proj.repeat(self.attention_head_size, 1).t().reshape(-1)
                    # 确保mask_proj在正确的设备上
                    mask_proj = mask_proj.to(attn.out_proj.weight.device)
                    proj_idx = mask_proj.nonzero().squeeze()
                    scale_map = self.create_feat_scaleing_attn(inputs, np.array(proj_idx.cpu()), self.hidden_size)
                    scale_map = torch.from_numpy(scale_map).type(dtype=torch.float).to(self.device)
                    scale_map = scale_map.t()
    
                    weight = attn.out_proj.weight.data.clone().detach()
                    attn.out_proj.weight.data = weight[:, mask_proj]
                    for i, Cin in enumerate(weight):
                        Out = Cin.reshape(Cin.shape[0], -1).float().to(self.device)
                        Out = torch.mm(scale_map, Out).reshape(-1)
                        attn.out_proj.weight.data[i, :] = Out.to(attn.out_proj.weight.data.device)
                else:
                    weight = attn.out_proj.weight.data.reshape(-1, self.attention_head_size, self.num_attention_heads)[:, :,
                             mask]
                    attn.out_proj.weight.data = weight.reshape(weight.shape[0], -1)
            else:
                torch.cuda.empty_cache()
                attn.num_heads = d_prime * (attn.num_heads // attn.num_key_value_heads)
                attn.num_key_value_heads = d_prime
                attn.hidden_size = attn.head_dim * attn.num_heads
                attn.max_position_embeddings = attn.head_dim * attn.num_heads
    
                weight = attn.k_proj.weight.data.reshape(self.num_key_value_heads, self.attention_head_size, -1)[mask, :, :]
                attn.k_proj.weight.data = weight.reshape(-1, weight.shape[2])
    
                weight = attn.q_proj.weight.data.reshape(self.num_key_value_heads, self.attention_head_size*(attn.num_heads // attn.num_key_value_heads), -1)[mask, :, :]
                attn.q_proj.weight.data = weight.reshape(-1, weight.shape[2])
    
                weight = attn.v_proj.weight.data.reshape(self.num_key_value_heads, self.attention_head_size, -1)[mask, :, :]
                attn.v_proj.weight.data = weight.reshape(-1, weight.shape[2])
    
                if attn.k_proj.bias is not None:
                    bias = attn.k_proj.bias.data.reshape(self.num_attention_heads, -1)[mask, :]
                    attn.k_proj.bias.data = bias.reshape(-1)
    
                    bias = attn.q_proj.bias.data.reshape(self.num_attention_heads, -1)[mask, :]
                    attn.q_proj.bias.data = bias.reshape(-1)
    
                    bias = attn.v_proj.bias.data.reshape(self.num_attention_heads, -1)[mask, :]
                    attn.v_proj.bias.data = bias.reshape(-1)
                
                if self.recon:
                    torch.cuda.empty_cache()
                    mask_proj = mask.unsqueeze(0)
                    mask_proj = mask_proj.repeat(self.attention_head_size*(attn.num_heads // attn.num_key_value_heads), 1).t().reshape(-1)
                    # 确保mask_proj在正确的设备上
                    mask_proj = mask_proj.to(attn.o_proj.weight.device)
                    proj_idx = mask_proj.nonzero().squeeze()
                    scale_map = self.create_feat_scaleing_attn(inputs, np.array(proj_idx.cpu()), self.hidden_size)
                    scale_map = torch.from_numpy(scale_map).type(dtype=torch.float).to(self.device)
                    scale_map = scale_map.t()

                    torch.cuda.empty_cache()
                    weight = attn.o_proj.weight.data.clone().detach()
                    attn.o_proj.weight.data = weight[:, mask_proj]
                    for i, Cin in enumerate(weight):
                        Out = Cin.reshape(Cin.shape[0], -1).float().to(self.device)
                        Out = torch.mm(scale_map, Out).reshape(-1)
                        attn.o_proj.weight.data[i, :] = Out.to(attn.o_proj.weight.device)
                else:
                    # attn.o_proj.weight.data = attn.o_proj.weight.data.cuda()
                    weight = attn.o_proj.weight.data.reshape(-1, self.attention_head_size*(attn.num_heads // attn.num_key_value_heads), self.num_key_value_heads)[:, :,
                             mask]
                    attn.o_proj.weight.data = weight.reshape(weight.shape[0], -1)
                

        else:
            if "OPT" in self.model.__class__.__name__:
                pre_layer = get_ffn1(self.model, idx)
                target_layer = get_ffn2(self.model, idx)
            else:
                pre_layer_1 = get_gate(self.model, idx)
                pre_layer_2 = get_up(self.model, idx)
                target_layer = get_down(self.model, idx)
                
            d_prime = format_rank(preserve_ratio * target_layer.weight.data.shape[1])
            ratio = d_prime / target_layer.weight.data.shape[1]

            hidden_metric = self.A_metric[global_idx]
            sorted_idx = torch.sort(-hidden_metric)
            preserve_idx = sorted_idx.indices[:d_prime]  # to preserve index
            preserve_idx,_ = torch.sort(preserve_idx)

            mask = torch.zeros_like(hidden_metric, dtype=bool)
            mask[preserve_idx] = True
            
            # 确保mask在与target_layer相同的设备上
            mask = mask.to(target_layer.weight.device)

            if self.recon:
                torch.cuda.empty_cache()
                data_saver = DataSaverHook(store_input=True, store_output=False, stop_forward=True)
                handles_inputs = target_layer.register_forward_hook(data_saver)
                inputs = []
                for j in range(len(self.recon_inps)):
                    with torch.no_grad():
                        try:
                            if "OPT" in self.model.__class__.__name__:
                                self.recon_outs[j] = layer(self.recon_inps[j].unsqueeze(0), attention_mask=self.attention_mask)[0]
                            else:
                                self.recon_outs[j] = layer(self.recon_inps[j].unsqueeze(0), attention_mask=self.attention_mask,
                                                     position_ids=self.position_ids)[0]
                        except StopForwardException:
                            pass
                        inputs.append(data_saver.input_store[0].detach().to(self.device))
                handles_inputs.remove()

                torch.cuda.empty_cache()
                scale_map = self.create_feat_scaleing_ffn(inputs, np.array(preserve_idx.cpu()), self.intermediate_size, 0.01)
                scale_map = torch.from_numpy(scale_map).type(dtype=torch.float).to(self.device)
                scale_map = scale_map.t()

                torch.cuda.empty_cache()
                weight = target_layer.weight.data.clone().detach()
                target_layer.weight.data = weight[:, mask]
                for i, Cin in enumerate(weight):
                    Out = Cin.reshape(Cin.shape[0], -1).float().to(self.device)
                    Out = torch.mm(scale_map, Out).reshape(-1)
                    target_layer.weight.data[i, :] = Out.to(target_layer.weight.device)
            else:
                target_layer.weight.data = target_layer.weight.data[:, mask]

            if "OPT" in self.model.__class__.__name__:
                pre_layer.weight.data = pre_layer.weight.data[mask, :]
                if pre_layer.bias is not None:
                    pre_layer.bias.data = pre_layer.bias.data[mask]
            else:
                pre_layer_1.weight.data = pre_layer_1.weight.data[mask, :]
                if pre_layer_1.bias is not None:
                    pre_layer_1.bias.data = pre_layer_1.bias.data[mask]
                    
                pre_layer_2.weight.data = pre_layer_2.weight.data[mask, :]
                if pre_layer_2.bias is not None:
                    pre_layer_2.bias.data = pre_layer_2.bias.data[mask]
                    

            if self.recon:
                torch.cuda.empty_cache()
                for j in range(len(self.recon_inps)):
                    with torch.no_grad():
                        if "OPT" in self.model.__class__.__name__:
                            self.recon_outs[j] = layer(self.recon_inps[j].unsqueeze(0), attention_mask=self.attention_mask)[0]
                        else:
                            self.recon_outs[j] = layer(self.recon_inps[j].unsqueeze(0), attention_mask=self.attention_mask,
                                  position_ids=self.position_ids)[0]
                self.recon_inps = self.recon_outs

        # 移除冗余的打印，让进度条更清晰
        # print(f"=> Layer {idx} ({'Head' if head else 'FFN'}) pruned with ratio {ratio:.4f} -> d_prime: {d_prime}")
        if self.recon:
            torch.save(self.model.state_dict(), self.export_path)


        return ratio, d_prime



    def reset(self):
        self.layer_idx = 0
        self.head = True
        self.strategy = []
        self.d_prime_list = []

        # --- 核心改动：在重建模型后，立刻为其恢复 .seqlen 属性 ---
        if hasattr(self, 'original_state_dict'):
            # print("=> Resetting environment: creating a fresh model instance...")
            
            # 1. 在CPU上快速创建一个新的、未经修改的模型“骨架”
            fresh_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True 
            )
            
            # 2. 【新增的关键步骤】为新模型恢复 .seqlen 属性
            #    这里的逻辑与您原来 _get_model_local 中的逻辑保持一致
            max_seq_len = getattr(fresh_model.config, 'max_position_embeddings', 2048)
            fresh_model.seqlen = min(max_seq_len, 2048)
            # print(f"=> Restored .seqlen attribute to: {fresh_model.seqlen}")

            # 3. 将保存在内存中的原始权重加载到这个新骨架中
            fresh_model.load_state_dict(self.original_state_dict)
            
            # 4. 将这个恢复好的、完整的模型赋给 self.model 并移动到GPU
            self.model = fresh_model
            if self.device.type != 'cpu':
                # 自动计算设备映射
                max_memory = get_balanced_memory(self.model, max_memory=None, no_split_module_classes=self.model._no_split_modules)
                device_map = infer_auto_device_map(self.model, max_memory=max_memory, no_split_module_classes=self.model._no_split_modules)
                
                # 分发模型到多个GPU
                self.model = dispatch_model(self.model, device_map=device_map)

            # print("=> Model has been successfully reset to its original state.")
        else:
            # 这个分支只会在 __init__ 首次调用 reset 时执行
            # print("=> Initial reset call during __init__, model is already pristine.")
            self._get_model_local(verbose=False)  # 静默模式，避免重复输出 

        # --- 后续的 obs 返回逻辑不变 ---
        if self.use_new_input:
            if not hasattr(self, 'state'):
                print("=> Reset called during initialization, returning placeholder.")
                return np.array([0.0], dtype=np.float32)
            obs = self.state
        else:
            obs = np.array(self.preserve_ratio, dtype=np.float32)
        
        return obs

    def hijack_input(self, module, list_to_append):
        hook = lambda _, inputs: list_to_append.append(inputs)
        handle = module.register_forward_pre_hook(hook)
        return handle

    def set_export_path(self, path):
        self.export_path = path

    # def _action_wall(self, action):
    #     if self.export_model:
    #         return action

    #     actions = np.abs(action)
    #     actions = np.clip(actions, 0, 1)

    #     def format_rank(x):
    #         rank = int(np.around(x))
    #         return max(rank, 1)

    #     for i in range(len(self.dim_list)):
    #         d_prime = format_rank(actions[i] * self.dim_list[i])
    #         d_prime = int(np.ceil(d_prime * 1. / self.channel_round) * self.channel_round)
    #         actions[i] = d_prime / self.dim_list[i]

    #     actions = actions.clip(self.lbound, self.rbound)

    #     for idx in range(len(actions)):
    #         other_comp = 0
    #         this_comp = 0
    #         for i in range(self.num_hidden_layers * 2):
    #             if self.args.prune == 'flops':
    #                 if i == idx:  # this layer
    #                     this_comp += self.flops_list[i]
    #                 elif i < idx:
    #                     other_comp += actions[i] * self.flops_list[i]
    #                 else:
    #                     other_comp += self.lbound * self.flops_list[i]
    #             elif self.args.prune == 'para':
    #                 if i == idx:  # this layer
    #                     this_comp += self.param_list[i]
    #                 elif i < idx:
    #                     other_comp += actions[i] * (self.param_list[i] - self.norm_para[i]) + self.norm_para[i]
    #                 else:
    #                     other_comp += self.lbound * (self.param_list[i] - self.norm_para[i]) + self.norm_para[i]

    #         max_preserve_ratio = (self.expected_preserve_computation - other_comp) * 1. / this_comp
    #         actions[idx] = np.minimum(actions[idx], max_preserve_ratio)
    #         actions[idx] = np.maximum(actions[idx], self.lbound)

    #     return list(actions)

# channel_pruning_env_llm_global.py

    def _action_wall(self, action):
        if self.export_model:
            return action

        # 1. 基础处理
        actions = np.abs(action)
        actions = np.clip(actions, self.lbound, self.rbound)

        # 辅助函数：根据比例计算总计算量
        def get_computation(ratios):
            computation = 0
            num_layers = len(self.param_list)
            if self.args.prune == 'para':
                for i in range(num_layers):
                    computation += ratios[i] * (self.param_list[i] - self.norm_para[i]) + self.norm_para[i]
            else: # flops
                for i in range(num_layers):
                    computation += ratios[i] * self.flops_list[i]
            return computation

        current_computation = get_computation(actions)
        target_computation = self.expected_preserve_computation

        # 2. 双向调节 (浮点数层面)
        if current_computation < target_computation: # 智能放大
            deficit = target_computation - current_computation
            unsaturated_indices = [i for i, r in enumerate(actions) if r < self.rbound]
            if unsaturated_indices:
                cost_list = self.param_list if self.args.prune == 'para' else self.flops_list
                comp_headrooms = []
                for i in unsaturated_indices:
                    ratio_headroom = self.rbound - actions[i]
                    cost = (self.param_list[i] - self.norm_para[i]) if self.args.prune == 'para' else self.flops_list[i]
                    comp_headrooms.append(ratio_headroom * cost)

                total_comp_headroom = sum(comp_headrooms)
                if total_comp_headroom > 1e-6:
                    for i, original_idx in enumerate(unsaturated_indices):
                        comp_to_add = deficit * (comp_headrooms[i] / total_comp_headroom)
                        cost = (self.param_list[original_idx] - self.norm_para[original_idx]) if self.args.prune == 'para' else self.flops_list[original_idx]
                        if cost > 1e-6:
                            actions[original_idx] += comp_to_add / cost
                    actions = np.clip(actions, self.lbound, self.rbound)

        elif current_computation > target_computation: # 序贯缩小
            for idx in range(len(actions)):
                other_comp, this_comp = 0, 0
                for i in range(len(actions)):
                    cost = (self.param_list[i] - self.norm_para[i]) if self.args.prune == 'para' else self.flops_list[i]
                    norm_cost = self.norm_para[i] if self.args.prune == 'para' else 0
                    if i == idx: this_comp += self.param_list[i] if self.args.prune == 'para' else self.flops_list[i]
                    elif i < idx: other_comp += actions[i] * cost + norm_cost
                    else: other_comp += self.lbound * cost + norm_cost
                
                if this_comp > 1e-6:
                    max_preserve_ratio = (target_computation - other_comp) / this_comp
                    actions[idx] = np.minimum(actions[idx], max_preserve_ratio)
                    actions[idx] = np.maximum(actions[idx], self.lbound)

        # 3. 通道取整 (这是导致超预算的根源)
        d_primes = [max(1, int(np.around(r * d))) for r, d in zip(actions, self.dim_list)]
        if self.channel_round > 0:
            d_primes = [min(d_dim, int(np.ceil(d_p / self.channel_round) * self.channel_round)) for d_p, d_dim in zip(d_primes, self.dim_list)]
        
        actions_rounded = [d_p / d_dim if d_dim > 0 else 0 for d_p, d_dim in zip(d_primes, self.dim_list)]
        
        # --- 4. 最终预算修正：处理因向上取整导致的微小超支 ---
        rounded_computation = get_computation(actions_rounded)
        overshoot = rounded_computation - target_computation

        if overshoot > 0:
            # 从后向前，贪婪地削减那些可以削减的层的通道数，直到满足预算
            cost_list = self.param_list if self.args.prune == 'para' else self.flops_list
            for i in range(len(actions_rounded) - 1, -1, -1):
                if overshoot <= 0: break
                
                # 计算减去一个 rounding step 能节省多少计算量
                if self.dim_list[i] > 0 and self.channel_round > 0:
                    cost_per_channel_unit = cost_list[i] / self.dim_list[i]
                    cost_per_round_step = cost_per_channel_unit * self.channel_round
                    
                    # 检查削减后是否会低于 lbound
                    min_d_prime = max(1, int(np.around(self.lbound * self.dim_list[i])))
                    
                    while d_primes[i] > min_d_prime and overshoot > 0:
                        d_primes[i] -= self.channel_round
                        overshoot -= cost_per_round_step

            # 根据最终的 d_primes 重新计算 actions
            actions_final = [d_p / d_dim if d_dim > 0 else 0 for d_p, d_dim in zip(d_primes, self.dim_list)]
        else:
            actions_final = actions_rounded

        return list(actions_final)

    def _cur_flops(self, actions):
        flops = 0
        for i in range(self.num_hidden_layers*2):
            flops += actions[i] * self.flops_list[i]
        return flops

    def _cur_para(self, actions):
        param = 0
        for i in range(self.num_hidden_layers*2):
            param += actions[i] * (self.param_list[i] - self.norm_para[i]) + self.norm_para[i]
        return param


    def _init_data(self):
        self.dataloader, _ = get_loaders(self.dataset, nsamples=self.n_samples, seed=self.args.seed, seqlen=self.model.seqlen, tokenizer=self.tokenizer)
        # self.dataloader, _ = get_loaders(self.dataset, nsamples=self.n_samples, seed=self.args.seed, seqlen=2048, tokenizer=self.tokenizer)

    def prepare_calibration_input_opt(self):
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        if "OPT" in self.model.__class__.__name__:
            layers = self.model.model.decoder.layers
        else:
            layers = self.model.model.layers
        
        torch.cuda.empty_cache()

        # 智能设备选择 - 使用模型embedding层所在的设备
        device = self.device
        if hasattr(self.model, 'hf_device_map') and "model.embed_tokens" in self.model.hf_device_map:
            device = self.model.hf_device_map["model.embed_tokens"]
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            # 直接从embedding层获取设备
            device = next(self.model.model.embed_tokens.parameters()).device

        dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros((self.n_samples, self.model.seqlen, self.model.config.hidden_size), dtype=dtype, device=device)
        inps.requires_grad = False
        cache = {'i': 0, 'attention_mask': None, }

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']
                raise ValueError

        layers[0] = Catcher(layers[0])
        for batch in self.dataloader:
            try:
                self.model(batch[0].to(device))
            except ValueError:
                pass
        layers[0] = layers[0].module

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        self.model.config.use_cache = use_cache

        position_ids = None
        
        # 确保attention_mask与序列长度匹配
        if attention_mask is not None and attention_mask.shape[-1] != self.model.seqlen:
            print(f"=> Adjusting attention_mask from {attention_mask.shape} to match seqlen {self.model.seqlen}")
            # 重新生成正确尺寸的attention_mask
            batch_size = attention_mask.shape[0]
            # 对于OPT模型，我们通常只需要简单的padding mask
            new_attention_mask = torch.ones((batch_size, self.model.seqlen), 
                                          device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = new_attention_mask
            print(f"=> Generated new attention_mask with shape: {attention_mask.shape}")
        
        # 对于Llama等需要position_ids的模型，确保位置一致性
        if hasattr(cache, 'position_ids') and cache.get('position_ids') is not None:
            position_ids = cache['position_ids']
            if position_ids.shape[-1] != self.model.seqlen:
                print(f"=> Adjusting position_ids from {position_ids.shape} to match seqlen {self.model.seqlen}")
                if position_ids.shape[-1] > self.model.seqlen:
                    position_ids = position_ids[:, :self.model.seqlen]
                else:
                    batch_size = position_ids.shape[0]
                    extended_positions = torch.arange(position_ids.shape[-1], self.model.seqlen, device=position_ids.device)
                    extended_positions = extended_positions.unsqueeze(0).expand(batch_size, -1)
                    position_ids = torch.cat([position_ids, extended_positions], dim=1)

        return inps, outs, attention_mask, position_ids

    def prepare_calibration_input(self):
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        layers = self.model.model.layers

        torch.cuda.empty_cache()

        # 智能设备选择 - 使用模型embedding层所在的设备
        device = self.device
        if hasattr(self.model, 'hf_device_map') and "model.embed_tokens" in self.model.hf_device_map:
            device = self.model.hf_device_map["model.embed_tokens"]
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            # 直接从embedding层获取设备
            device = next(self.model.model.embed_tokens.parameters()).device

        dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros((self.n_samples, self.model.seqlen, self.model.config.hidden_size), dtype=dtype, device=device)
        inps.requires_grad = False
        cache = {'i': 0, 'attention_mask': None, "position_ids": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs['position_ids']
                raise ValueError

        layers[0] = Catcher(layers[0])
        for batch in self.dataloader:
            try:
                self.model(batch[0].to(device))
            except ValueError:
                pass
        layers[0] = layers[0].module

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']
        
        # 为Llama等模型重新生成正确长度的position_ids和attention_mask
        if position_ids is not None:
            batch_size = position_ids.shape[0]
            # 生成与序列长度匹配的position_ids
            position_ids = torch.arange(self.model.seqlen, device=position_ids.device).unsqueeze(0).expand(batch_size, -1)
            print(f"=> Generated position_ids with shape: {position_ids.shape} for seqlen: {self.model.seqlen}")
        
        # 确保attention_mask与序列长度匹配
        if attention_mask is not None and attention_mask.shape[-1] != self.model.seqlen:
            print(f"=> Adjusting attention_mask from {attention_mask.shape} to match seqlen {self.model.seqlen}")
            # 重新生成正确尺寸的attention_mask
            batch_size = attention_mask.shape[0]
            # 创建新的attention_mask，形状为[batch_size, 1, seq_len, seq_len]
            new_attention_mask = torch.ones((batch_size, 1, self.model.seqlen, self.model.seqlen), 
                                          device=attention_mask.device, dtype=attention_mask.dtype)
            # 设置为下三角矩阵（causal mask）
            new_attention_mask = torch.tril(new_attention_mask)
            # 转换为注意力分数的mask（0表示可以attention，-inf表示不能）
            new_attention_mask = new_attention_mask.masked_fill(new_attention_mask == 0, float('-inf'))
            new_attention_mask = new_attention_mask.masked_fill(new_attention_mask == 1, 0.0)
            attention_mask = new_attention_mask
            print(f"=> Generated new attention_mask with shape: {attention_mask.shape}")
        
        self.model.config.use_cache = use_cache

        return inps, outs, attention_mask, position_ids

    def _generate_prunable_module_names(self):
        """
        生成可剪枝模块的名称列表，供 FeatureExtractor 使用
        """
        self.prunable_module_names = []
        print("=> Generating prunable module names for FeatureExtractor...")
        
        # 根据模型类型确定层的访问路径
        if "opt" in self.model.config.model_type.lower():
            layer_prefix = "model.decoder.layers"
        else:
            # 为其他模型（如Llama）提供一个通用路径
            layer_prefix = "model.layers"

        for i in range(self.num_hidden_layers):
            # 注意力模块 (MHA)
            self.prunable_module_names.append(f"{layer_prefix}.{i}.self_attn")
            # 前馈网络 (FFN) - OPT使用fc1, Llama使用gate_proj
            if "opt" in self.model.config.model_type.lower():
                self.prunable_module_names.append(f"{layer_prefix}.{i}.fc1")
            else:
                self.prunable_module_names.append(f"{layer_prefix}.{i}.gate_proj")
        
        print(f"=> Found {len(self.prunable_module_names)} prunable modules.")


    def _extract_layer_information(self):
        # 首先生成可剪枝模块名称列表
        self._generate_prunable_module_names()
        
        with torch.no_grad():
            if "OPT" in self.model.__class__.__name__:
                print('Experiments with OPT models')
                self.inps, self.outs, self.attention_mask, self.position_ids = self.prepare_calibration_input_opt()
            else:
                self.inps, self.outs, self.attention_mask, self.position_ids = self.prepare_calibration_input()

        self.flops_list = []
        self.dim_list = []
        
        # 动态获取模型配置
        num_heads_per_layer = [self.num_attention_heads] * self.num_hidden_layers
        num_neurons_per_layer = [self.intermediate_size] * self.num_hidden_layers

        for num_heads, num_neurons in zip(num_heads_per_layer, num_neurons_per_layer):
            attention_mac = num_heads * mac_per_head(self.model.seqlen, self.hidden_size,  self.attention_head_size)
            ffn_mac = num_neurons * mac_per_neuron(self.model.seqlen,  self.hidden_size)
            self.flops_list.append(attention_mac / 1e6)
            self.flops_list.append(ffn_mac / 1e6)
            self.dim_list.append(self.num_attention_heads)
            self.dim_list.append(self.intermediate_size)

        print("=> Extracting layer information and computing metrics...")
        self.A_metric = []
        self.param_list = []
        self.norm_para = []
        layers = get_layers(self.model)
        
        if self.recon:
            idx = torch.randperm(self.n_samples)[:self.args.recon_sample]
            self.recon_inps = self.inps[idx]
            self.recon_outs = self.outs[idx]
        
        pbar = tqdm(range(len(layers)), desc="Processing Layers", 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        is_llama_model = "Llama" in self.model.__class__.__name__

        for i in pbar:
            layer_start_time = time.time()
            pbar.set_description(f"Layer {i+1}/{self.num_hidden_layers}")
            layer = layers[i]
            
            # --- [通用部分] 计算MHA重要性 ---
            mha = get_mha(self.model, i)
            mha_proj_layer = get_mha_proj(self.model, i)
            
            mha_para = get_layer_param(mha)
            mha_norm = get_norm_param(mha)
            self.param_list.append(mha_para / 1e6)
            self.norm_para.append(mha_norm / 1e6)
            
            wrapped_mha = WrappedGPT(mha_proj_layer)
            mha_handle = mha_proj_layer.register_forward_hook(
                lambda _, inp, out: wrapped_mha.add_batch(inp[0].data, out.data)
            )

            # --- [核心修改] 根据模型类型，差异化处理FFN重要性 ---
            if is_llama_model:
                # 获取 Llama FFN 的所有相关层
                down_layer = get_down(self.model, i)
                
                # FFN 参数统计 (逻辑不变)
                ffn_para = get_layer_param(layer) - mha_para
                ffn_norm = get_norm_param(layer) - mha_norm
                self.param_list.append(ffn_para / 1e6)
                self.norm_para.append(ffn_norm / 1e6)

                # --- 核心修正 ---
                # 为了评估中间神经元的重要性，我们只需要关注最终输出层 down_proj
                # 因此，我们只对 down_layer 的输入进行 hook，以获取中间激活的幅度
                wrapped_down = WrappedGPT(down_layer)
                down_handle = down_layer.register_forward_hook(
                    lambda _, inp, out: wrapped_down.add_batch(inp[0].data, out.data)
                )
                
                # 执行前向传播以收集 MHA 和 FFN 的激活数据
                for j in range(self.n_samples):
                    with torch.no_grad():
                        self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask, position_ids=self.position_ids)[0]
                
                # 移除钩子
                mha_handle.remove()
                down_handle.remove()

                # MHA 度量衡的计算逻辑不变
                W_metric_mha = torch.abs(mha_proj_layer.weight.data) * torch.sqrt(wrapped_mha.scaler_row.reshape((1, -1)))
                self.A_metric.append(torch.mean(W_metric_mha, dim=0))

                # --- Wanda for Llama FFN 的正确实现 ---
                # 1. 权重部分：使用 down_proj 层的权重 |W_down|，其形状为 [H, I]
                # 2. 激活部分：使用送入 down_proj 层的中间激活的幅度 ||A_inter||，其形状为 [I]
                #    (由 wrapped_down.scaler_row 提供)
                W_metric_ffn = torch.abs(down_layer.weight.data) * torch.sqrt(wrapped_down.scaler_row.reshape((1, -1)))
                
                # 3. 聚合：与 OPT 逻辑一样，我们在 dim=0 (H维度)上求平均，
                #    得到每个中间神经元（I维度）的唯一重要性分数。
                #    最终结果是一个长度为 I 的向量。
                self.A_metric.append(torch.mean(W_metric_ffn, dim=0))

            else: # OPT 及其他标准 FFN 模型的逻辑
                ffn_layer = get_ffn2(self.model, i)
                
                ffn_para = get_layer_param(layer) - mha_para
                ffn_norm = get_norm_param(layer) - mha_norm
                self.param_list.append(ffn_para / 1e6)
                self.norm_para.append(ffn_norm / 1e6)

                wrapped_ffn = WrappedGPT(ffn_layer)
                ffn_handle = ffn_layer.register_forward_hook(
                    lambda _, inp, out: wrapped_ffn.add_batch(inp[0].data, out.data)
                )
                
                # 执行前向传播
                for j in range(self.n_samples):
                    with torch.no_grad():
                        self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask)[0]
                
                mha_handle.remove()
                ffn_handle.remove()
                
                # 计算 MHA metric
                W_metric_mha = torch.abs(mha_proj_layer.weight.data) * torch.sqrt(wrapped_mha.scaler_row.reshape((1, -1)))
                self.A_metric.append(torch.mean(W_metric_mha, dim=0))

                # 计算 OPT FFN metric
                W_metric_ffn = torch.abs(ffn_layer.weight.data) * torch.sqrt(wrapped_ffn.scaler_row.reshape((1, -1)))
                self.A_metric.append(torch.mean(W_metric_ffn, dim=0))
            
            self.inps = self.outs
            layer_time = time.time() - layer_start_time
            pbar.set_postfix({"Time": f"{layer_time:.1f}s"})
        
        pbar.close()
        print("=> Layer information extraction completed!")
    
    def test_model(self, model):
        """
        在下游任务上评估模型性能
        智能选择评估方法：优先使用lm-eval-harness，回退到轻量级实现
        """
        # 尝试加载评估框架
        eval_type = _load_lmeval()
        
        if eval_type == "none":
            print("=> INFO: No evaluation framework available")
            print("=> To enable downstream evaluation, run: ./install_lmeval.sh")
            print("=> Or manually install: pip install lm-eval")
            print("=> Skipping downstream evaluation for now...")
            return False
            
        print("=> Setting up model for downstream evaluation...")
        
        try:
            if eval_type == "full":
                return self._evaluate_with_lmeval(model)
            elif eval_type == "lightweight":
                return self._evaluate_with_lightweight(model)
            else:
                print("=> Unknown evaluation type, skipping...")
                return False
                
        except Exception as e:
            print(f"=> WARNING: Downstream evaluation failed: {str(e)}")
            print("=> Trying backup generation test...")
            
            # 备选方案：进行简单的文本生成测试
            try:
                self._simple_generation_test(model)
                return True
            except Exception as backup_e:
                print(f"=> Backup evaluation also failed: {str(backup_e)}")
                print("=> Continuing without downstream evaluation...")
                return False

    def _evaluate_with_lmeval(self, model):
        """使用完整的lm-eval-harness进行评估"""
        try:
            # 使用新版lm-eval-harness API
            model_wrapper = HFLM(
                pretrained=model,
                tokenizer=self.tokenizer,
                batch_size=4,
                device=self.device
            )
            
            # 选择核心下游任务
            task_names = ["piqa", "hellaswag", "winogrande", "arc_easy"]
            
            print(f"=> Evaluating on {len(task_names)} downstream tasks: {task_names}")
            
            # 使用新版API进行评估
            results = evaluator.simple_evaluate(
                model=model_wrapper,
                tasks=task_names,
                num_fewshot=0,
                limit=100,  # 限制每个任务的样本数量
                bootstrap_iters=100,  # 减少bootstrap迭代次数
                no_cache=True,
                verbosity="INFO"
            )
            
            # 显示结果
            print("=> Downstream Task Results (lm-eval-harness):")
            task_scores = {}
            
            if "results" in results:
                for task_name, task_result in results["results"].items():
                    if isinstance(task_result, dict):
                        # 寻找主要指标
                        main_metrics = ["acc", "acc_norm", "exact_match", "f1"]
                        for metric in main_metrics:
                            if metric in task_result:
                                score = task_result[metric]
                                task_scores[task_name] = score
                                print(f"   {task_name}: {score:.4f}")
                                break
                
                if task_scores:
                    avg_score = sum(task_scores.values()) / len(task_scores)
                    print(f"=> Average downstream task performance: {avg_score:.4f}")
                    return True
                else:
                    print("=> WARNING: No valid task scores obtained!")
                    return False
            else:
                print("=> WARNING: No results returned from evaluation!")
                return False
                
        except Exception as e:
            print(f"=> lm-eval-harness evaluation failed: {str(e)}")
            raise e

    def _evaluate_with_lightweight(self, model):
        """使用轻量级评估器进行评估"""
        try:
            print("=> Using lightweight evaluation implementation")
            print("=> Setting up model for downstream evaluation...")
            evaluator_light = LightweightEvaluator(model, self.tokenizer, device=None)
            
            print("=> Starting evaluation with 50 samples per task...")
            # 进行评估
            results = evaluator_light.evaluate_all(num_samples_per_task=50)
            
            print("=> Evaluation completed successfully!")
            print(f"=> Total results obtained: {len(results)} items")
            
            # 显示结果
            print("\n=> Downstream Task Results (lightweight):")
            
            # 按任务类型分组显示
            reasoning_tasks = ["boolq_acc", "piqa_acc", "hellaswag_acc"]
            commonsense_tasks = ["winogrande_acc", "obqa_acc"]
            science_tasks = ["arc_easy_acc", "arc_challenge_acc"]
            
            if any(task in results for task in reasoning_tasks):
                print("   Reasoning Tasks:")
                for task in reasoning_tasks:
                    if task in results:
                        task_name = task.replace("_acc", "").upper()
                        print(f"     {task_name}: {results[task]:.4f}")
            
            if any(task in results for task in commonsense_tasks):
                print("   Commonsense Tasks:")
                for task in commonsense_tasks:
                    if task in results:
                        task_name = task.replace("_acc", "").title().replace("_", "-")
                        print(f"     {task_name}: {results[task]:.4f}")
                        
            if any(task in results for task in science_tasks):
                print("   Science Tasks:")
                for task in science_tasks:
                    if task in results:
                        task_name = task.replace("_acc", "").replace("_", "-").upper()
                        print(f"     {task_name}: {results[task]:.4f}")
            
            if "avg_score" in results:
                print(f"\n=> Average downstream task performance: {results['avg_score']:.4f}")
            
            # print("=> Downstream task evaluation completed successfully")
            return True
            
        except Exception as e:
            print(f"=> Lightweight evaluation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

    def _simple_generation_test(self, model):
        """
        备选的简单文本生成测试
        """
        print("=> Running simple generation test as backup...")
        
        test_prompts = [
            "The capital of France is",
            "2 + 2 equals",
            "The largest planet in our solar system is"
        ]
        
        model.eval()
        with torch.no_grad():
            for i, prompt in enumerate(test_prompts):
                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"   Test {i+1}: '{prompt}' -> '{generated_text[len(prompt):].strip()}'")
                except Exception as e:
                    print(f"   Test {i+1}: Failed ({str(e)})")
        
        print("=> Simple generation test completed")

    def _validate(self, model):
        ppl = eval_ppl(model, self.tokenizer)
        
        # 检查是否开启下游任务评估
        enable_downstream = getattr(self.args, 'enable_downstream', True)
        
        if enable_downstream:
            # 下游任务评估 - 使用智能框架选择，失败时不终止程序
            try:
                success = self.test_model(model)
                # if success:
                #     print("=> Downstream task evaluation completed successfully")
                # else:
                #     print("=> Downstream task evaluation completed with warnings")
            except Exception as e:
                print(f"=> WARNING: Downstream evaluation encountered an error: {str(e)}")
                print("=> Continuing with PPL-only validation...")
        else:
            # print("=> Downstream task evaluation is disabled")
            # print("=> Using PPL-only validation")
            pass
        
        return ppl

    def set_static_state(self, final_state_vector: np.ndarray):
        """
        接收并存储由主脚本组装好的、最终的、完整的静态状态向量。
        
        Args:
            final_state_vector (np.ndarray): 主脚本筛选并拼接好的一维状态向量
        """
        print("=> Environment received final static state vector.")
        
        # 验证输入是否为一维向量
        if final_state_vector.ndim != 1:
            raise ValueError(f"Expected a 1D state vector, but got shape {final_state_vector.shape}")
            
        self.state = final_state_vector
        self.state_dim = self.state.shape[0]
        
        print(f"=> Final state assembled in env. Shape: {self.state.shape}")
        print(f"=> Environment state_dim correctly set to: {self.state_dim}")

