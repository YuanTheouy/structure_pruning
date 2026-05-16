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
import json


from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
# 启用下游任务评估需要的导入 - 使用新版lm-eval-harness
# 注释掉旧版本导入
# from lib.lm_eval.evaluator import evaluate, make_table  
# from lib.lm_eval.tasks import get_task_dict, ALL_TASKS
# from lib.lm_eval.utils import pattern_match
# from lib.lm_eval.models import get_model
# ================== 新的、简化的调试工具 ==================
class NanInfDetectedError(Exception):
    """自定义异常，用于被钩子捕捉到NaN/Inf时抛出"""
    def __init__(self, message, module_name, module_full_name):
        super().__init__(message)
        self.module_name = module_name
        self.module_full_name = module_full_name

def nan_checker_hook_with_exception(module, a_input, a_output):
    """
    一个前向钩子函数，在检测到NaN/Inf时直接抛出我们自定义的异常。
    """
    # a_output 可能是 tensor, tuple, 或其他类型
    outputs_to_check = []
    if isinstance(a_output, torch.Tensor):
        outputs_to_check.append(a_output)
    elif isinstance(a_output, (list, tuple)):
        for item in a_output:
            if isinstance(item, torch.Tensor):
                outputs_to_check.append(item)

    for tensor in outputs_to_check:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            # 获取模块的完整名称 (例如 'model.layers.15.self_attn.o_proj')
            module_full_name = ""
            for name, mod in module.named_modules():
                if mod is module:
                    module_full_name = name
                    break # 通常内层循环一次就够了

            raise NanInfDetectedError(
                f"NaN or Inf detected in the output of module!",
                module_name=module.__class__.__name__,
                module_full_name=module_full_name # 传递完整的模块名
            )
# ==========================================================

def nan_checker_hook(module, a_input, a_output):
    """
    一个前向钩子函数，用于检查模块输出中是否存在NaN/Inf。
    """
    if not isinstance(a_output, torch.Tensor):
        # 有些模块可能输出tuple等，我们只关心Tensor
        return
    if torch.isnan(a_output).any() or torch.isinf(a_output).any():
        print(f"!!! NaN or Inf detected in the output of module: {module.__class__.__name__} !!!")
        # 您可以在这里设置断点进行调试
        # import pdb; pdb.set_trace()

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

# file: env/channel_pruning_env_llm_global.py

# file: env/channel_pruning_env_llm_global.py

def _load_lmeval():
    """
    智能加载评估框架 (终极离线修复版)
    通过猴子补丁(Monkey-Patching)劫持 evaluate.load 函数，强制其在离线环境中使用本地缓存。
    """
    global LMEVAL_AVAILABLE, LIGHTWEIGHT_EVAL_AVAILABLE, lm_eval, evaluator, HFLM, LightweightEvaluator

    if LMEVAL_AVAILABLE: return "full"
    if LIGHTWEIGHT_EVAL_AVAILABLE: return "lightweight"

    try:
        import evaluate as hf_evaluate
        import os
        import sys
        
        print("\n" + "="*70)
        print("🔬 ENTERING OFFLINE-FIRST EVALUATION LOADER (MONKEY-PATCH MODE)")
        print(f"Python executable: {sys.executable}")
        print("="*70)

        # --- 步骤 1: 预加载所有我们需要的度量到私有字典中 ---
        print("\n--- [STEP 1] Pre-loading metrics into a local dictionary...")
        HF_CACHE_HOME = os.path.expanduser(os.path.join('~', '.cache', 'huggingface'))
        metrics_to_preload = ["exact_match", "rouge", "bleu", "sacrebleu"]
        preloaded_metrics = {}

        for metric_name in metrics_to_preload:
            metric_path = os.path.join(HF_CACHE_HOME, 'evaluate', 'metrics', metric_name)
            if os.path.exists(metric_path):
                print(f"   -> Loading '{metric_name}' from: {metric_path}")
                preloaded_metrics[metric_name] = hf_evaluate.load(metric_path, trust_remote_code=True)
            else:
                raise FileNotFoundError(f"Required local cache for '{metric_name}' not found at {metric_path}")
        
        print("✅ [SUCCESS] All metrics loaded into local dictionary.")

        # --- 步骤 2: 准备猴子补丁 ---
        print("\n--- [STEP 2] Preparing to monkey-patch 'evaluate.load'...")
        original_evaluate_load = hf_evaluate.load  # 保存原始函数
        
        def custom_load(path, *args, **kwargs):
            """我们的自定义加载函数"""
            print(f"   🐵 INTERCEPTED call to evaluate.load(path='{path}')")
            # 如果是短名称调用，从我们的字典返回
            if path in preloaded_metrics:
                print(f"   ✅ Returning pre-loaded module for '{path}' from our dictionary.")
                return preloaded_metrics[path]
            
            # 否则，调用原始函数 (以防万一)
            print(f"   -> Path '{path}' not in our dictionary, falling back to original load function.")
            return original_evaluate_load(path, *args, **kwargs)

        # --- 步骤 3: 应用补丁并尝试导入 lm-eval ---
        hf_evaluate.load = custom_load # 应用猴子补丁
        print("✅ [SUCCESS] 'evaluate.load' has been temporarily replaced.")
        
        print("\n--- [STEP 3] Attempting to import 'lm-eval-harness' with the patch active...")
        try:
            import lm_eval as _lm_eval
            from lm_eval import evaluator as _evaluator
            from lm_eval.models.huggingface import HFLM as _HFLM
            
            lm_eval = _lm_eval
            evaluator = _evaluator
            HFLM = _HFLM
            LMEVAL_AVAILABLE = True
            print("\n🎉🎉🎉 [SUCCESS] 'lm-eval-harness' imported successfully under offline patch! 🎉🎉🎉")
            
            # 恢复原始函数
            hf_evaluate.load = original_evaluate_load
            print("✅ 'evaluate.load' has been restored to its original state.")
            return "full"

        except Exception as e:
            # 如果即使在打了补丁后仍然失败，打印错误
            import traceback
            print("\n🔥🔥🔥 [FAILURE] 'lm-eval-harness' import failed EVEN WITH a patch.")
            traceback.print_exc()
            raise e

    except Exception as final_exception:
        # 如果整个过程有任何问题，则回退到轻量级实现
        print(f"\n🔥🔥🔥 [CRITICAL FAILURE] Offline loader failed: {final_exception}")
        LMEVAL_AVAILABLE = False # 确保状态正确

    # 最终的回退逻辑
    if not LMEVAL_AVAILABLE:
        try:
            print("\n--- [FALLBACK] Reverting to lightweight evaluation implementation.")
            from lib.lightweight_eval import LightweightEvaluator as _LightweightEvaluator
            LightweightEvaluator = _LightweightEvaluator
            LIGHTWEIGHT_EVAL_AVAILABLE = True
            return "lightweight"
        except ImportError:
            print("=> Critical: No evaluation framework available at all.")
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


class ChannelPruningEnv:
    """
    Env for channel pruning search
    """
    def __init__(self, model, data, preserve_ratio, args, n_data_worker=4,
                 batch_size=256, export_model=False, use_new_input=False):

        # --- 步骤 1: 优先初始化所有基础依赖项 ---
        self.args = args
        self.model_path = args.model
        
        # _get_model会初始化 self.model 和 self.tokenizer
        self._get_model() 
        
        # !! 关键修正 !!: 立即从传入的参数中赋值 self.dataset
        # 在您的 amc_searchPPO.py 中，第二个参数'data'被传入了args.dataset_name
        self.dataset = data
        
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
        
        # # --- 步骤 2: 现在所有依赖项都已具备，可以安全地创建评估子集 ---
        # print("=> Loading validation set to create reward evaluation subsets...")
        
        # # 这个函数现在可以安全地使用 self.dataset, self.model.seqlen 等属性
        # _, val_tensor = get_loaders(
        #     self.dataset,
        #     nsamples=0, # nsamples=0 表示我们只需要测试集
        #     seed=self.args.seed,
        #     seqlen=self.model.seqlen,
        #     tokenizer=self.tokenizer
        # )
        
        # # 将测试集张量切分成样本列表
        # val_tensor_flat = val_tensor.input_ids.view(-1)
        # num_samples_total = val_tensor_flat.size(0) // self.model.seqlen
        # self.full_val_samples = [
        #     (val_tensor_flat[i*self.model.seqlen:(i+1)*self.model.seqlen].unsqueeze(0), None) 
        #     for i in range(num_samples_total)
        # ]
        # print(f"=> Successfully loaded and chunked {len(self.full_val_samples)} validation samples.")
        
        # random.shuffle(self.full_val_samples)
        
        
        # # 从args中读取比例
        # reward_subset_size_small = getattr(self.args, 'reward_subset_size_small', 0.05)
        # reward_subset_size_large = getattr(self.args, 'reward_subset_size_large', 0.2)

        # self.small_subset_size = int(len(self.full_val_samples) * reward_subset_size_small)
        # self.large_subset_size = int(len(self.full_val_samples) * reward_subset_size_large)
        
        # self.current_eval_mode = 'small' # 默认从 'small' 模式开始
        # self.active_reward_eval_set = None

        # self.resample_reward_eval_set() # Call the new method to create the first batch
        # print(f"Defaulting to SMALL reward evaluation set with {len(self.active_reward_eval_set)} samples.")
        # # --- [代码结束] ---
        
        # +++ [新增] 动态数据集初始化逻辑 +++
        print("=> Loading full validation set for dynamic sampling...")
        _, val_tensor = get_loaders(
            self.dataset,
            nsamples=0, # nsamples=0 表示加载完整的测试集
            seed=self.args.seed,
            seqlen=self.model.seqlen,
            tokenizer=self.tokenizer
        )
        val_tensor_flat = val_tensor.input_ids.view(-1)
        num_samples_total = val_tensor_flat.size(0) // self.model.seqlen
        self.full_val_samples = [
            (val_tensor_flat[i*self.model.seqlen:(i+1)*self.model.seqlen].unsqueeze(0), None) 
            for i in range(num_samples_total)
        ]
        print(f"=> Successfully loaded and chunked {len(self.full_val_samples)} validation samples.")
        
        # 初始化当前数据集比例
        self.current_dataset_ratio = 1.0 

        if self.args.use_dataset_growth:
            # 模式1: 启用数据集渐进增长
            print(f"=> Dataset Growth ENABLED. Initializing with ratio: {self.args.dataset_initial_ratio}")
            self.current_dataset_ratio = self.args.dataset_initial_ratio
            # 立即根据初始比例更新一次评估集
            self.resample_reward_eval_set() 
        else:
            # 模式2: 禁用数据集渐进增长，始终使用全集
            print("=> Dataset Growth DISABLED. Using FULL validation set for all evaluations.")
            self.active_reward_eval_set = self.full_val_samples

        print(f"=> Initial active evaluation set contains {len(self.active_reward_eval_set)} samples.")
        # +++ [新增结束] +++
        

        # 简单设备分配 - 遵循CUDA_VISIBLE_DEVICES设置
        if torch.cuda.is_available():
            # 获取模型实际所在的设备
            model_device = next(self.model.parameters()).device
            self.device = model_device
            print(f"=> Model device: {self.device}")
        else:
            self.device = torch.device("cpu")
            print(f"=> Using CPU device (CUDA not available)")
            
        

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
        
        # 如果启用了延迟下游任务评估，在初始化时只计算PPL
        if getattr(self.args, 'delayed_downstream_eval', False):
            print("=> Initial validation: PPL-only (downstream evaluation delayed)")
            self.org_ppl = self._validate_ppl_only(self.model)
        else:
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

    def update_dataset_ratio(self, new_ratio):
        """
        由外部训练循环调用，以更新用于奖励评估的数据集比例。
        """
        # 确保比例在0和1之间
        new_ratio = max(0.0, min(new_ratio, 1.0))
        
        if self.current_dataset_ratio != new_ratio:
            self.current_dataset_ratio = new_ratio
            # 比例变化后，需要重新采样评估集
            self.resample_reward_eval_set()
            # (可选) 日志输出，用于调试
            # print(f"=> Dataset ratio updated to: {self.current_dataset_ratio:.3f}, new eval set size: {len(self.active_reward_eval_set)}")
            
    # +++ [新增] 新的、基于比例的 resample_reward_eval_set 方法 +++
    def resample_reward_eval_set(self):
        """
        从完整的验证集中，根据当前的 self.current_dataset_ratio 比例，
        重新随机抽取样本，更新当前激活的评估子集。
        """
        # 1. 每次都随机打乱全集，确保抽样的随机性
        random.shuffle(self.full_val_samples)
        
        # 2. 根据当前比例计算需要的样本数量
        num_samples_to_take = int(len(self.full_val_samples) * self.current_dataset_ratio)
        
        # 3. 确保至少有1个样本，防止比例过小时取0
        num_samples_to_take = max(1, num_samples_to_take)
        
        # 4. 从打乱后的全集中切片，得到新的激活评估集
        self.active_reward_eval_set = self.full_val_samples[:num_samples_to_take]

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
        elif "qwen" or "Qwen" in self.args.model: # 为 Qwen 添加分支
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)


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
            
        # ==================== 核心修改 1: 使用 bfloat16 ====================
        print("=> [STABILITY] Loading model with torch_dtype=torch.bfloat16")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16, # <--- 修改这里
            # torch_dtype=torch.float16,
            cache_dir=self.args.cache_dir,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        # ===================================================================
        
        
        # 根据模型类型动态设置序列长度
        if "opt" in self.model.config.model_type.lower():
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
        #     if "opt" in self.model.config.model_type.lower():
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
                                if "opt" in self.model.config.model_type.lower():
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
        
        
        # --- START: CRITICAL FIX ---
        # 经过剪枝操作，模型各层的设备状态可能已被打乱。
        # 在进行验证（完整前向传播）之前，我们必须调用dispatch_model，
        # 让accelerate根据其内部的hf_device_map重新校准和分配所有层，
        # 确保模型恢复到设备一致的状态。
        print("=> Re-dispatching model to correct device map after pruning...")
        self.model = dispatch_model(self.model, device_map=self.model.hf_device_map)
        # --- END: CRITICAL FIX ---
        
        total_time = time.time() - start_time
        # print(f"=> Pruning completed in {total_time:.1f}s (avg: {total_time/total_steps:.1f}s/step)")

        assert len(self.action) == self.num_hidden_layers * 2
        # print("=> Calculating final metrics...")
        current_flops = self._cur_flops(self.strategy)
        compress_ratio = current_flops * 1. / self.org_flops
        current_para = self._cur_para(self.strategy)
        para_ratio = current_para * 1. / self.org_para

        
        # ==================== 核心修改 2: 在step函数中实现调试逻辑 ====================
        hooks = []
        try:
            # 验证前，给模型的所有相关模块挂上我们的“巡查员”——钩子
            # print("==> [DEBUG] Registering hooks to detect NaN during validation...")
            for name, module in self.model.named_modules():
                # 我们关心所有可能出现数值问题的层
                if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)) or 'LlamaRMSNorm' in module.__class__.__name__:
                    hooks.append(module.register_forward_hook(nan_checker_hook_with_exception))
            
            # 执行常规的PPL验证（eval_ppl内部会使用 torch.no_grad()，内存开销小）
            ppl = self._validate(self.model)

        except NanInfDetectedError as e:
            # 如果任何一层产生NaN，我们的自定义异常会被捕捉到
            print("\n" + "="*70)
            print(f"  [!!!] CRITICAL FAILURE: {e}")
            print(f"  [!!!] First module to produce NaN/Inf was: <{e.module_name}>")
            print(f"  [!!!] Full Module Path: {e.module_full_name}") # 打印完整的模块路径
            print("="*70 + "\n")
            ppl = float('nan') # 将PPL设为nan，以便RL知道这是个失败的尝试
        
        finally:
            # 无论如何，最后都要把所有钩子都移除，清理现场
            for handle in hooks:
                handle.remove()
            # print("==> [DEBUG] All hooks removed.")
        # =========================================================================
        
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

        # if reward > self.best_reward:
        #     self.best_reward = reward
        #     self.best_strategy = self.action.copy()
        #     self.best_d_prime_list = self.d_prime_list.copy()
        #     prGreen(
        #         'New best reward: {:.4f}, ppl: {:.4f}, compress: {:.4f}, para: {:.4f}'.format(self.best_reward, ppl,
        #                                                                                       compress_ratio, para_ratio))
        #     prGreen('New best policy: {}'.format(self.best_strategy))
        #     prGreen('New best d primes: {}'.format(self.best_d_prime_list))
        #     torch.save(self.model.state_dict(), self.export_path)

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
        # # --- 在这里加入下面这行打印语句 ---
        # print("\n\n--->>> 正在执行最新版本的 PRUNE 方法！ <<<---\n\n")
        # # ------------------------------------
        
        # # ==================== 最终探员：开始现场勘查 ====================
        # print(f"--- [DEBUG] 进入prune方法: layer_idx={idx}, is_head={head} ---")
        # try:
        #     print(f"--- [DEBUG] self.model.__class__.__name__ is: {self.model.__class__.__name__}")
        #     print(f"--- [DEBUG] self.model.config.model_type is: {self.model.config.model_type}")
        #     model_type_str_lower = self.model.config.model_type.lower()
        #     print(f"--- [DEBUG] model_type_str_lower is: '{model_type_str_lower}'")
        #     check_result = "opt" in model_type_str_lower
        #     print(f"--- [DEBUG] The check '\"opt\" in model_type_str_lower' 的结果是: {check_result}")
        # except Exception as e:
        #     print(f"--- [DEBUG] 打印调试信息时发生错误: {e}")
        # print("--------------------------------------------------------------------")
        # # =================================================================
        
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
                            if "opt" in self.model.config.model_type.lower():
                                self.recon_outs[j] = \
                                layer(self.recon_inps[j].unsqueeze(0), attention_mask=self.attention_mask)[0]
                            else:
                                self.recon_outs[j] = \
                                layer(self.recon_inps[j].unsqueeze(0), attention_mask=self.attention_mask,
                                      position_ids=self.position_ids)[0]
                    self.recon_inps = self.recon_outs

                return preserve_ratio, self.hidden_size

        # if head:
        #     attn = get_mha(self.model, idx)
        #     target_layer = get_mha_proj(self.model, idx)
        #     d_prime = format_rank(preserve_ratio * self.num_key_value_heads)
        #     ratio = d_prime / self.num_key_value_heads
            
        #     # --- START MODIFICATION ---
        #     # 1. 确定一个有效的计算设备 (例如主设备)
        #     compute_device = self.device 

        #     # 2. 在访问权重数据前，强制将可能被offload的模块移动到计算设备
        #     #    这将触发accelerate将权重从CPU加载回GPU
        #     attn.q_proj.to(compute_device)
        #     attn.k_proj.to(compute_device)
        #     attn.v_proj.to(compute_device)
        #     target_layer.to(compute_device) # 对于Llama/Qwen, 这是o_proj
            
        #     # 3. 确保重要性度量张量也在同一设备上
        #     head_metric = self.A_metric[global_idx].to(compute_device)
        #     # --- END MODIFICATION ---

        #     # head_metric = self.A_metric[global_idx]
        #     head_metric = head_metric.reshape(self.num_key_value_heads, -1)
        #     head_metric = torch.sum(head_metric, dim=-1)
        #     sorted_idx = torch.sort(-head_metric)
        #     preserve_idx = sorted_idx.indices[:d_prime]  # to preserve index
        #     preserve_idx,_ = torch.sort(preserve_idx)

        #     mask = torch.zeros_like(head_metric, dtype=bool)
        #     mask[preserve_idx] = True
            
        #     # 确保mask在与target_layer相同的设备上
        #     mask = mask.to(target_layer.weight.device)

        #     if self.recon:
        #         torch.cuda.empty_cache()
        #         data_saver = DataSaverHook(store_input=True, store_output=False, stop_forward=True)
        #         handles_inputs = target_layer.register_forward_hook(data_saver)
        #         inputs = []
        #         for j in range(len(self.recon_inps)):
        #             with torch.no_grad():
        #                 try:
        #                     if "opt" in self.model.config.model_type.lower():
        #                         self.recon_outs[j] = layer(self.recon_inps[j].unsqueeze(0), attention_mask=self.attention_mask)[0]
        #                     else:
        #                         self.recon_outs[j] = layer(self.recon_inps[j].unsqueeze(0), attention_mask=self.attention_mask,
        #                                              position_ids=self.position_ids)[0]
        #                 except StopForwardException:
        #                     pass
        #                 inputs.append(data_saver.input_store[0].detach().to(self.device))
        #         handles_inputs.remove()


        #     if "opt" in self.model.config.model_type.lower():
        #         attn.num_heads = d_prime
        #         attn.embed_dim = attn.head_dim * d_prime
                
        #         weight = attn.k_proj.weight.data.reshape(self.num_attention_heads, self.attention_head_size, -1)[mask, :, :]
        #         attn.k_proj.weight.data = weight.reshape(-1, weight.shape[2])
    
        #         weight = attn.q_proj.weight.data.reshape(self.num_attention_heads, self.attention_head_size, -1)[mask, :, :]
        #         attn.q_proj.weight.data = weight.reshape(-1, weight.shape[2])
    
        #         weight = attn.v_proj.weight.data.reshape(self.num_attention_heads, self.attention_head_size, -1)[mask, :, :]
        #         attn.v_proj.weight.data = weight.reshape(-1, weight.shape[2])
    
        #         if attn.k_proj.bias is not None:
        #             bias = attn.k_proj.bias.data.reshape(self.num_attention_heads, -1)[mask, :]
        #             attn.k_proj.bias.data = bias.reshape(-1)
    
        #             bias = attn.q_proj.bias.data.reshape(self.num_attention_heads, -1)[mask, :]
        #             attn.q_proj.bias.data = bias.reshape(-1)
    
        #             bias = attn.v_proj.bias.data.reshape(self.num_attention_heads, -1)[mask, :]
        #             attn.v_proj.bias.data = bias.reshape(-1)
                    
        #         if self.recon:
        #             mask_proj = mask.unsqueeze(0)
        #             mask_proj = mask_proj.repeat(self.attention_head_size, 1).t().reshape(-1)
        #             # 确保mask_proj在正确的设备上
        #             mask_proj = mask_proj.to(attn.out_proj.weight.device)
        #             proj_idx = mask_proj.nonzero().squeeze()
        #             scale_map = self.create_feat_scaleing_attn(inputs, np.array(proj_idx.cpu()), self.hidden_size)
        #             scale_map = torch.from_numpy(scale_map).type(dtype=torch.float).to(self.device)
        #             scale_map = scale_map.t()
    
        #             weight = attn.out_proj.weight.data.clone().detach()
        #             attn.out_proj.weight.data = weight[:, mask_proj]
        #             for i, Cin in enumerate(weight):
        #                 Out = Cin.reshape(Cin.shape[0], -1).float().to(self.device)
        #                 Out = torch.mm(scale_map, Out).reshape(-1)
        #                 attn.out_proj.weight.data[i, :] = Out.to(attn.out_proj.weight.data.device)
        #         else:
        #             weight = attn.out_proj.weight.data.reshape(-1, self.attention_head_size, self.num_attention_heads)[:, :,
        #                      mask]
        #             attn.out_proj.weight.data = weight.reshape(weight.shape[0], -1)
        #     else:
        #         torch.cuda.empty_cache()
        #         attn.num_heads = d_prime * (attn.num_heads // attn.num_key_value_heads)
        #         attn.num_key_value_heads = d_prime
        #         attn.hidden_size = attn.head_dim * attn.num_heads
        #         attn.max_position_embeddings = attn.head_dim * attn.num_heads
    
        #         weight = attn.k_proj.weight.data.reshape(self.num_key_value_heads, self.attention_head_size, -1)[mask, :, :]
        #         attn.k_proj.weight.data = weight.reshape(-1, weight.shape[2])
    
        #         weight = attn.q_proj.weight.data.reshape(self.num_key_value_heads, self.attention_head_size*(attn.num_heads // attn.num_key_value_heads), -1)[mask, :, :]
        #         attn.q_proj.weight.data = weight.reshape(-1, weight.shape[2])
    
        #         weight = attn.v_proj.weight.data.reshape(self.num_key_value_heads, self.attention_head_size, -1)[mask, :, :]
        #         attn.v_proj.weight.data = weight.reshape(-1, weight.shape[2])
    
        #     if attn.k_proj.bias is not None:
        #         # For K_proj bias, use num_key_value_heads
        #         bias = attn.k_proj.bias.data.reshape(self.num_key_value_heads, -1)[mask, :]
        #         attn.k_proj.bias.data = bias.reshape(-1)

        #         # For Q_proj bias, also use num_key_value_heads to align with GQA group pruning strategy
        #         # The q_proj bias has more elements, but it's structured in groups.
        #         num_q_groups = self.num_attention_heads // self.num_key_value_heads
        #         bias = attn.q_proj.bias.data.reshape(self.num_key_value_heads, -1)[mask, :]
        #         attn.q_proj.bias.data = bias.reshape(-1)

        #         # For V_proj bias, use num_key_value_heads
        #         bias = attn.v_proj.bias.data.reshape(self.num_key_value_heads, -1)[mask, :]
        #         attn.v_proj.bias.data = bias.reshape(-1)
                
        #         if self.recon:
        #             torch.cuda.empty_cache()
        #             mask_proj = mask.unsqueeze(0)
        #             mask_proj = mask_proj.repeat(self.attention_head_size*(attn.num_heads // attn.num_key_value_heads), 1).t().reshape(-1)
        #             # 确保mask_proj在正确的设备上
        #             mask_proj = mask_proj.to(attn.o_proj.weight.device)
        #             proj_idx = mask_proj.nonzero().squeeze()
        #             scale_map = self.create_feat_scaleing_attn(inputs, np.array(proj_idx.cpu()), self.hidden_size)
        #             scale_map = torch.from_numpy(scale_map).type(dtype=torch.float).to(self.device)
        #             scale_map = scale_map.t()

        #             torch.cuda.empty_cache()
        #             weight = attn.o_proj.weight.data.clone().detach()
        #             attn.o_proj.weight.data = weight[:, mask_proj]
        #             for i, Cin in enumerate(weight):
        #                 Out = Cin.reshape(Cin.shape[0], -1).float().to(self.device)
        #                 Out = torch.mm(scale_map, Out).reshape(-1)
        #                 attn.o_proj.weight.data[i, :] = Out.to(attn.o_proj.weight.device)
        #         else:
        #             # attn.o_proj.weight.data = attn.o_proj.weight.data.cuda()
        #             weight = attn.o_proj.weight.data.reshape(-1, self.attention_head_size*(attn.num_heads // attn.num_key_value_heads), self.num_key_value_heads)[:, :,
        #                      mask]
        #             attn.o_proj.weight.data = weight.reshape(weight.shape[0], -1)
                
        compute_device = self.device
        hidden_metric = self.A_metric[global_idx].to(compute_device)      
        if head:
            # --- [处理注意力头 ATTENTION HEAD] ---
            d_prime = format_rank(preserve_ratio * self.num_key_value_heads)
            ratio = d_prime / self.num_key_value_heads
            
            head_metric = hidden_metric.reshape(self.num_key_value_heads, -1)
            head_metric = torch.sum(head_metric, dim=-1)
            sorted_idx = torch.sort(-head_metric)
            preserve_idx = sorted_idx.indices[:d_prime]
            preserve_idx, _ = torch.sort(preserve_idx)

            mask = torch.zeros_like(head_metric, dtype=bool)
            mask[preserve_idx] = True
            
            attn = get_mha(self.model, idx)
            attn.to(compute_device)

            if "opt" in self.model.config.model_type.lower():
                # --- OPT 模型的专属、完整逻辑 ---
                target_layer = get_mha_proj(self.model, idx)  # This is out_proj for OPT
                target_layer.to(compute_device)
                mask = mask.to(target_layer.weight.device)
                
                attn.num_heads = d_prime
                attn.embed_dim = attn.head_dim * d_prime
                
                # Prune Q, K, V weights
                attn.k_proj.weight.data = attn.k_proj.weight.data.reshape(self.num_attention_heads, self.attention_head_size, -1)[mask, :, :].reshape(-1, self.hidden_size)
                attn.q_proj.weight.data = attn.q_proj.weight.data.reshape(self.num_attention_heads, self.attention_head_size, -1)[mask, :, :].reshape(-1, self.hidden_size)
                attn.v_proj.weight.data = attn.v_proj.weight.data.reshape(self.num_attention_heads, self.attention_head_size, -1)[mask, :, :].reshape(-1, self.hidden_size)

                if attn.k_proj.bias is not None:
                    # Prune Q, K, V biases
                    attn.k_proj.bias.data = attn.k_proj.bias.data.reshape(self.num_attention_heads, -1)[mask, :].reshape(-1)
                    attn.q_proj.bias.data = attn.q_proj.bias.data.reshape(self.num_attention_heads, -1)[mask, :].reshape(-1)
                    attn.v_proj.bias.data = attn.v_proj.bias.data.reshape(self.num_attention_heads, -1)[mask, :].reshape(-1)
                
                # Prune output projection (out_proj) and handle reconstruction
                if self.recon:
                    data_saver = DataSaverHook(store_input=True, store_output=False, stop_forward=True)
                    handles_inputs = target_layer.register_forward_hook(data_saver)
                    inputs = []
                    for j in range(len(self.recon_inps)):
                        with torch.no_grad():
                            try:
                                self.recon_outs[j] = layer(self.recon_inps[j].unsqueeze(0), attention_mask=self.attention_mask)[0]
                            except StopForwardException:
                                pass
                            inputs.append(data_saver.input_store[0].detach().to(self.device))
                    handles_inputs.remove()

                    mask_proj = mask.unsqueeze(0).repeat(self.attention_head_size, 1).t().reshape(-1)
                    mask_proj = mask_proj.to(target_layer.weight.device)
                    proj_idx = mask_proj.nonzero().squeeze()

                    scale_map = self.create_feat_scaleing_attn(inputs, np.array(proj_idx.cpu()), self.hidden_size).t()
                    scale_map = torch.from_numpy(scale_map).type(dtype=torch.float).to(self.device)
                    
                    weight = target_layer.weight.data.clone().detach()
                    target_layer.weight.data = weight[:, mask_proj]
                    for i, Cin in enumerate(weight):
                        Out = Cin.reshape(Cin.shape[0], -1).float().to(self.device)
                        Out = torch.mm(scale_map, Out).reshape(-1)
                        target_layer.weight.data[i, :] = Out.to(target_layer.weight.device)
                else:
                    weight = target_layer.weight.data.reshape(-1, self.attention_head_size, self.num_attention_heads)[:, :, mask]
                    target_layer.weight.data = weight.reshape(weight.shape[0], -1)
            
            else:
                # --- Llama/Qwen 等模型的专属、完整逻辑 ---
                target_layer = get_mha_proj(self.model, idx) # This is o_proj for Llama/Qwen
                target_layer.to(compute_device)
                mask = mask.to(target_layer.weight.device)
                
                num_q_groups = attn.num_heads // attn.num_key_value_heads
                attn.num_heads = d_prime * num_q_groups
                attn.num_key_value_heads = d_prime
                attn.hidden_size = attn.head_dim * attn.num_heads

                # Prune K, V weights
                attn.k_proj.weight.data = attn.k_proj.weight.data.reshape(self.num_key_value_heads, self.attention_head_size, -1)[mask, :, :].reshape(-1, self.hidden_size)
                attn.v_proj.weight.data = attn.v_proj.weight.data.reshape(self.num_key_value_heads, self.attention_head_size, -1)[mask, :, :].reshape(-1, self.hidden_size)
                
                # Prune Q weights (grouped by num_key_value_heads)
                attn.q_proj.weight.data = attn.q_proj.weight.data.reshape(self.num_key_value_heads, num_q_groups * self.attention_head_size, -1)[mask, :, :].reshape(-1, self.hidden_size)

                if attn.k_proj.bias is not None:
                    # Prune K, V, Q biases
                    attn.k_proj.bias.data = attn.k_proj.bias.data.reshape(self.num_key_value_heads, -1)[mask, :].reshape(-1)
                    attn.v_proj.bias.data = attn.v_proj.bias.data.reshape(self.num_key_value_heads, -1)[mask, :].reshape(-1)
                    attn.q_proj.bias.data = attn.q_proj.bias.data.reshape(self.num_key_value_heads, -1)[mask, :].reshape(-1)
                
                # Prune output projection (o_proj) and handle reconstruction
                if self.recon:
                    data_saver = DataSaverHook(store_input=True, store_output=False, stop_forward=True)
                    handles_inputs = target_layer.register_forward_hook(data_saver)
                    inputs = []
                    for j in range(len(self.recon_inps)):
                        with torch.no_grad():
                            try:
                                self.recon_outs[j] = layer(self.recon_inps[j].unsqueeze(0), attention_mask=self.attention_mask, position_ids=self.position_ids)[0]
                            except StopForwardException:
                                pass
                            inputs.append(data_saver.input_store[0].detach().to(self.device))
                    handles_inputs.remove()
                    
                    mask_proj = mask.unsqueeze(0).repeat(self.attention_head_size * num_q_groups, 1).t().reshape(-1)
                    mask_proj = mask_proj.to(target_layer.weight.device)
                    proj_idx = mask_proj.nonzero().squeeze()

                    scale_map = self.create_feat_scaleing_attn(inputs, np.array(proj_idx.cpu()), self.hidden_size).t()
                    scale_map = torch.from_numpy(scale_map).type(dtype=torch.float).to(self.device)

                    weight = target_layer.weight.data.clone().detach()
                    target_layer.weight.data = weight[:, mask_proj]
                    for i, Cin in enumerate(weight):
                        Out = Cin.reshape(Cin.shape[0], -1).float().to(self.device)
                        Out = torch.mm(scale_map, Out).reshape(-1)
                        target_layer.weight.data[i, :] = Out.to(target_layer.weight.device)
                else:
                    weight = target_layer.weight.data.reshape(-1, self.attention_head_size * num_q_groups, self.num_key_value_heads)[:, :, mask]
                    target_layer.weight.data = weight.reshape(weight.shape[0], -1)

        else:
            if "opt" in self.model.config.model_type.lower():
                pre_layer = get_ffn1(self.model, idx)
                target_layer = get_ffn2(self.model, idx)
            else:
                pre_layer_1 = get_gate(self.model, idx)
                pre_layer_2 = get_up(self.model, idx)
                target_layer = get_down(self.model, idx)
                
                # --- START MODIFICATION ---
                # 1. 确定一个有效的计算设备 (例如主设备)
                compute_device = self.device

                # 2. 在访问权重数据前，强制将可能被offload的模块移动到计算设备
                pre_layer_1.to(compute_device)
                pre_layer_2.to(compute_device)
                target_layer.to(compute_device)

                # 3. 确保重要性度量张量也在同一设备上
                hidden_metric = self.A_metric[global_idx].to(compute_device)
                # --- END MODIFICATION ---
                
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
                            if "opt" in self.model.config.model_type.lower():
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

            if "opt" in self.model.config.model_type.lower():
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
                        if "opt" in self.model.config.model_type.lower():
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
            # ==================== 核心修改 3: reset时也使用 bfloat16 ====================
            fresh_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16, # <--- 修改这里
                # torch_dtype=torch.float16,
                low_cpu_mem_usage=True 
            )
            # =========================================================================
            
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
        if "opt" in self.model.config.model_type.lower():
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
            if "opt" in self.model.config.model_type.lower():
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
        
        is_qwen_model = "Qwen" in self.model.__class__.__name__ # 添加Qwen的判断

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
                
            # --- START MODIFICATION ---
            elif is_qwen_model:
                # Qwen的逻辑与Llama非常相似，但访问路径不同
                down_layer = get_down(self.model, i) # get_down现在已经能处理Qwen
                
                ffn_para = get_layer_param(layer) - mha_para
                ffn_norm = get_norm_param(layer) - mha_norm
                self.param_list.append(ffn_para / 1e6)
                self.norm_para.append(ffn_norm / 1e6)

                wrapped_down = WrappedGPT(down_layer)
                down_handle = down_layer.register_forward_hook(
                    lambda _, inp, out: wrapped_down.add_batch(inp[0].data, out.data)
                )
                
                # 执行前向传播以收集激活
                for j in range(self.n_samples):
                    with torch.no_grad():
                        # Llama/Qwen的forward参数一致
                        self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask, position_ids=self.position_ids)[0]
                
                mha_handle.remove()
                down_handle.remove()

                # MHA 度量衡计算逻辑不变
                W_metric_mha = torch.abs(mha_proj_layer.weight.data) * torch.sqrt(wrapped_mha.scaler_row.reshape((1, -1)))
                self.A_metric.append(torch.mean(W_metric_mha, dim=0))

                # Wanda for Qwen FFN: 与Llama逻辑完全相同
                W_metric_ffn = torch.abs(down_layer.weight.data) * torch.sqrt(wrapped_down.scaler_row.reshape((1, -1)))
                self.A_metric.append(torch.mean(W_metric_ffn, dim=0))
            # --- END MODIFICATION ---
            
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
        """使用完整的lm-eval-harness进行评估 (通过猴子补丁支持本地数据集)"""
        import lm_eval.api.task
        # 保存原始的 Task 初始化函数
        original_init = lm_eval.api.task.ConfigurableTask.__init__

        # 定义我们的补丁函数
        def patched_configurable_task_init(self, config):
            """
            这是一个临时的替代品，用于替换 lm_eval.api.task.ConfigurableTask.__init__。
            它会检查正在初始化的任务是否为 'piqa'，如果是，则强行修改其配置以使用本地路径。
            """
            # 动态构建到本地 piqa 数据集的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            PIQA_LOCAL_PATH = os.path.join(current_dir, '..', 'local_datasets', 'piqa')
            
            # 检查任务名称是否为 piqa (或继承自 piqa)
            if config.get("task") == "piqa" or config.get("group") == "piqa":
                 # 检查本地路径是否存在，以决定是否打补丁
                if os.path.exists(PIQA_LOCAL_PATH):
                    print(f"🐵 Intercepted PIQA task config. Overriding dataset_path to: {PIQA_LOCAL_PATH}")
                    config["dataset_path"] = PIQA_LOCAL_PATH
                    config["dataset_name"] = None
            
            # 无论是否修改，都必须调用原始的 __init__ 函数来完成对象的初始化
            original_init(self, config=config)

        try:
            # --- 应用猴子补丁 ---
            lm_eval.api.task.ConfigurableTask.__init__ = patched_configurable_task_init
            print("✅ Monkey-patch for local PIQA dataset applied.")

            model_wrapper = HFLM(
                pretrained=model,
                tokenizer=self.tokenizer,
                batch_size=4,
            )
            
            task_names = ["boolq", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
            print(f"=> Evaluating on {len(task_names)} downstream tasks: {task_names}")
            
            # 使用原始的 simple_evaluate API，补丁会在后台生效
            results = evaluator.simple_evaluate(
                model=model_wrapper,
                tasks=task_names,
                num_fewshot=0,
                # limit=100,
                bootstrap_iters=100,
                verbosity="INFO"
            )

            # --- 处理结果 (逻辑不变) ---
            print("=> Downstream Task Results (lm-eval-harness):")
            task_scores = {}
            if "results" in results:
                for task_name, task_result in results["results"].items():
                    if isinstance(task_result, dict):
                        main_metrics = ["acc,none", "acc_norm,none", "acc", "acc_norm", "exact_match", "f1"]
                        found_metric = False
                        for metric in main_metrics:
                            if metric in task_result:
                                score = task_result[metric]
                                task_scores[task_name] = score
                                print(f"   ✅ {task_name}: {score:.4f} (found metric: '{metric}')")
                                found_metric = True
                                break
                        if not found_metric:
                            print(f"   ⚠️ Could not find a main metric for task '{task_name}' in its results.")
                if task_scores:
                    avg_score = sum(task_scores.values()) / len(task_scores)
                    print(f"=> Average downstream task performance: {avg_score:.4f}")
                    return True
                else:
                    print("=> WARNING: No valid task scores were extracted.")
                    return False
            else:
                print("=> WARNING: The key 'results' was not found in the raw output!")
                return False

        except Exception as e:
            import traceback
            print(f"=> lm-eval-harness evaluation failed: {str(e)}")
            traceback.print_exc()
            raise e
        finally:
            # --- 关键: 无论成功或失败，都恢复原始函数 ---
            lm_eval.api.task.ConfigurableTask.__init__ = original_init
            print("✅ Monkey-patch for local PIQA dataset restored.")
        
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

    def _validate_ppl_only(self, model):
        """仅计算PPL，不进行下游任务评估 (用于初始化阶段)"""
        return eval_ppl(model, self.tokenizer)

    def _validate(self, model):
        
        # ppl = eval_ppl(model, self.tokenizer)
        ppl = eval_ppl(model, self.tokenizer, dataset_override=self.active_reward_eval_set)
        
        # 检查是否开启下游任务评估以及是否延迟评估
        enable_downstream = getattr(self.args, 'enable_downstream', True)
        delayed_eval = getattr(self.args, 'delayed_downstream_eval', False)
        
        # 如果是延迟评估模式且在导出作业中，跳过下游任务评估
        if enable_downstream and not delayed_eval:
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
        elif delayed_eval:
            print("=> Downstream evaluation delayed until post-pruning+reconstruction")
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

