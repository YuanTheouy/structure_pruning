import time
import random
import torch
import torch.nn as nn
from torch.nn import functional as f
from lib.utils import AverageMeter, accuracy, prGreen
from lib.arch import get_layers, get_mha_proj, get_ffn2, get_ffn1, get_mha, find_layers
from lib.layerwrapper import WrappedGPT
from lib.sparsegpt import SparseGPT
from lib.eval import eval_ppl, eval_acc
from lib.data_utils import DataSaverHook, StopForwardException
from lib.data import get_loaders
from lib.Ridge import Ridge_Regression
from lib.linalg import lsmr_cupy_solver
from lib.mac import mac_per_head, mac_per_neuron, get_layer_param, get_norm_param
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoConfig, OPTForCausalLM
from env.rewards import *
import math

from lib.lm_eval.evaluator import evaluate, make_table
from lib.lm_eval.tasks import get_task_dict, ALL_TASKS
from lib.lm_eval.utils import pattern_match
from lib.lm_eval.models import get_model

import numpy as np
import copy

import sys  # 导入sys模块
sys.setrecursionlimit(10000)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WeightPruningEnv:
    """
    Env for channel pruning search
    """
    def __init__(self, model, data, preserve_ratio, args, prune_n=0, prune_m=0,
                 batch_size=256, export_model=False, use_new_input=False):

        self.args = args
        self.model_path = args.model
        self._get_model()

        # 智能设备分配 - 让PyTorch自动处理设备分配
        if torch.cuda.is_available():
            # 获取模型实际所在的设备
            model_device = next(self.model.parameters()).device
            self.device = model_device
            print(f"=> Auto-detected model device: {self.device}")
        else:
            self.device = torch.device("cpu")
            print(f"=> Using CPU device (CUDA not available)")
            
        self.dataset = args.dataset_name
        # self.n_data_worker = n_data_worker
        self.batch_size = batch_size
        self.data_type = data
        self.preserve_ratio = preserve_ratio

        ## llm
        self.num_hidden_layers = self.model.config.num_hidden_layers
        self.num_attention_heads = self.model.config.num_attention_heads
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
        if self.recon:
            self.prune_n = prune_n
            self.prune_m = prune_m

        self.export_model = export_model
        self.use_new_input = use_new_input


        # prepare data
        self._init_data()
  
        # extract information for preparing
        self._extract_layer_information()

        # build reward
        self.reset()  # restore weight
        self.org_ppl = self._validate(self.model)
        print('=> original ppl: {:.3f}%'.format(self.org_ppl))
        self.org_para = sum(self.param_list)
        print('=> Params:')
        print(self.param_list)
        print('=> original weight size: {:.4f} M param'.format(self.org_para * 1. / 1e6))
        # self.org_flops = sum(self.flops_list)
        # print('=> FLOPs:')
        # print(self.flops_list)
        # print('=> original FLOPs: {:.4f} M'.format(self.org_flops))

        self.expected_preserve_computation = self.preserve_ratio * self.org_para

        self.reward = eval(args.reward)

        self.best_reward = -math.inf
        self.best_strategy = None
        self.best_d_prime_list = None

        # self.org_w_size = sum(self.wsize_list)


    def _get_model(self):
        # config = AutoConfig.from_pretrained("./model/opt-125m")
        # self.model = AutoModelForCausalLM.from_pretrained("./llm_weights/models--facebook--opt-125m")
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.args.model,
        #     torch_dtype=torch.float16,
        #     cache_dir=self.args.cache_dir,
        #     low_cpu_mem_usage=True,
        #     device_map="auto"
        # )
        # self.model.seqlen = 2048

        self._get_model_local()

        if "opt" in self.args.model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        elif "llama" in self.args.model:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path, use_fast=False)


    def _get_model_local(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            cache_dir=self.args.cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        self.model.seqlen = 2048


    def step(self, action):
        self.action = self._action_wall(action)
        if self.recon:
            self.sparegpt(self.action)
        else:
            self.prune(self.action)

        assert len(self.action) == self.num_hidden_layers * 7
        # current_flops = self._cur_flops(self.strategy)
        # compress_ratio = current_flops * 1. / self.org_flops
        current_para = self._cur_para(self.strategy)
        para_ratio = current_para * 1. / self.org_para

        ppl = self._validate(self.model)
        reward = self.reward(ppl)

        info_set = {'compress_ratio': para_ratio, 'para_ratio': para_ratio, 'ppl': ppl, 'strategy': self.action.copy()}
        obs = np.array(self.preserve_ratio, dtype=np.float32)

        if reward > self.best_reward:
            self.best_reward = reward
            self.best_strategy = self.action.copy()
            self.best_d_prime_list = self.d_prime_list.copy()
            prGreen(
                'New best reward: {:.4f}, ppl: {:.4f}, compress: {:.4f}, para: {:.4f}'.format(self.best_reward, ppl,
                                                                                              para_ratio, para_ratio))
            prGreen('New best policy: {}'.format(self.best_strategy))
            prGreen('New best d primes: {}'.format(self.best_d_prime_list))
            torch.save(self.model.state_dict(), self.export_path)

        done = True

        return obs, reward, done, info_set

    def sparegpt(self, preserve_ratio):

        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        with torch.no_grad():
            if "OPT" in self.model.__class__.__name__:
                self.inps, self.outs, self.attention_mask, self.position_ids = self.prepare_calibration_input_opt()
            else:
                self.inps, self.outs, self.attention_mask, self.position_ids = self.prepare_calibration_input()

        layers = get_layers(self.model)
        idx = 0
        torch.cuda.empty_cache()
        for layer_idx in range(len(layers)):
            layer = layers[layer_idx]
            subset = find_layers(layer)

            gpts = {}
            for name in subset:
                gpts[name] = SparseGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in gpts:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(self.n_samples):
                with torch.no_grad():
                    if "OPT" in self.model.__class__.__name__:
                        self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask)[0]
                    else:
                        self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask,
                                             position_ids=self.position_ids)[0]
            for h in handles:
                h.remove()

            for name in gpts:
                gpts[name].fasterprune((1-preserve_ratio[idx]), prune_n=self.prune_n, prune_m=self.prune_m, percdamp=0.01,
                                       blocksize=128)
                gpts[name].free()
                idx += 1

                W = gpts[name].layer.weight.data
                total = W.numel()
                d_prime = (W!=0).sum().item()
                self.d_prime_list.append(d_prime)
                self.strategy.append(d_prime / total)


            for j in range(self.n_samples):
                with torch.no_grad():
                    if "OPT" in self.model.__class__.__name__:
                        self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask)[0]
                    else:
                        self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask,
                                             position_ids=self.position_ids)[0]
            torch.cuda.empty_cache()
            self.inps = self.outs

        self.model.config.use_cache = use_cache
        torch.cuda.empty_cache()


    def prune(self, preserve_ratio):

        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        with torch.no_grad():
            if "OPT" in self.model.__class__.__name__:
                self.inps, self.outs, self.attention_mask, self.position_ids = self.prepare_calibration_input_opt()
            else:
                self.inps, self.outs, self.attention_mask, self.position_ids = self.prepare_calibration_input()

        def format_rank(x):
            rank = int(np.around(x))
            return max(rank, 1)

        layers = get_layers(self.model)
        idx = 0
        for layer_idx in range(len(layers)):
            layer = layers[layer_idx]
            subset = find_layers(layer)

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            torch.cuda.empty_cache()
            for j in range(self.n_samples):
                with torch.no_grad():
                    if "OPT" in self.model.__class__.__name__:
                        self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask)[0]
                    else:
                        self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask,
                                             position_ids=self.position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                    wrapped_layers[name].scaler_row.reshape((1, -1)))

                W_mask = (torch.zeros_like(W_metric) == 1)
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                d_prime = format_rank(W_metric.shape[1] * preserve_ratio[idx])
                indices = sort_res[1][:, :int(W_metric.shape[1]-d_prime)]
                W_mask.scatter_(1, indices, True)

                torch.cuda.empty_cache()
                subset[name].weight.data[W_mask] = 0

                W = subset[name].weight.data
                total = W.numel()
                d_prime = (W != 0).sum().item()
                self.d_prime_list.append(d_prime)
                self.strategy.append(d_prime / total)
                # self.d_prime_list.append(d_prime)
                # self.strategy.append(d_prime / W_metric.shape[1])
                idx += 1

            torch.cuda.empty_cache()
            for j in range(self.n_samples):
                with torch.no_grad():
                    if "OPT" in self.model.__class__.__name__:
                        self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask)[0]
                    else:
                        self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask,
                                             position_ids=self.position_ids)[0]
            self.inps = self.outs

        self.model.config.use_cache = use_cache
        torch.cuda.empty_cache()



    def reset(self):
        self.strategy = []
        self.d_prime_list = []
        self._get_model_local()

        obs = np.array(self.preserve_ratio, dtype=np.float32)
        return obs


    def hijack_input(self, module, list_to_append):
        hook = lambda _, inputs: list_to_append.append(inputs)
        handle = module.register_forward_pre_hook(hook)
        return handle

    def set_export_path(self, path):
        self.export_path = path

    def _action_wall(self, action):
        # actions = np.abs(action)
        # actions = np.clip(actions, 0, 1)
        action = (np.tanh(action)+1)/2
        action = (action)*(self.rbound-self.lbound)+self.lbound
        actions = np.clip(action, 0., 1.)

        def format_rank(x):
            rank = int(np.around(x))
            return max(rank, 1)

        for i in range(len(self.param_list)):
            d_prime = format_rank(actions[i] * self.param_list[i])
            d_prime = int(np.ceil(d_prime * 1. / self.channel_round) * self.channel_round)
            actions[i] = d_prime / self.param_list[i]

        # actions = actions.clip(self.lbound, self.rbound)

        for idx in range(len(actions)):
            other_comp = 0
            this_comp = 0
            for i in range(len(self.param_list)):
                if i == idx:  # this layer
                    this_comp += self.param_list[i]
                elif i < idx:
                    other_comp += actions[i] * self.param_list[i]
                else:
                    other_comp += self.lbound * self.param_list[i]

            max_preserve_ratio = (self.expected_preserve_computation - other_comp) * 1. / this_comp
            actions[idx] = np.minimum(actions[idx], max_preserve_ratio)
            actions[idx] = np.maximum(actions[idx], self.lbound)

        return list(actions)


    def _cur_para(self, actions):
        param = 0
        for i in range(len(self.param_list)):
            param += actions[i] * self.param_list[i]
        return param


    # def _init_data(self):
    #     self.dataloader, _ = get_loaders(self.dataset, nsamples=self.n_samples, seed=self.args.seed, seqlen=2048, tokenizer=self.tokenizer)

    def _init_data(self):
        print(self.dataset)
        dataloader = []
        seqlen=2048
        nsamples=self.n_samples
        random.seed(self.args.seed)

        self.dataloader_bench = []
        task_list = []
        task_list.append(self.dataset)
        task_dict = get_task_dict(task_list)

        task = task_dict[self.dataset]
        task_doc_func = task.training_docs
        doc = task_doc_func()

        for i in doc:
            dataloader.append(task.doc_to_text(i))
        
        trainenc = self.tokenizer(" ".join(dataloader), return_tensors='pt')
        for _ in range(nsamples):
            # print(trainenc.input_ids.shape[1])
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            self.dataloader_bench.append((inp, tar))

    def prepare_calibration_input_opt(self):
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        if "OPT" in self.model.__class__.__name__:
            layers = self.model.model.decoder.layers
        else:
            layers = self.model.model.layers

        # device = torch.device("cuda:0")
        if "model.embed_tokens" in self.model.hf_device_map:
            device = self.model.hf_device_map["model.embed_tokens"]

        dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros((self.n_samples, self.model.seqlen, self.model.config.hidden_size), dtype=dtype, device=self.device)
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
        for batch in self.dataloader_bench:
            try:
                self.model(batch[0].to(self.device))
            except ValueError:
                pass
        layers[0] = layers[0].module

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        self.model.config.use_cache = use_cache

        position_ids = None

        return inps, outs, attention_mask, position_ids

    def prepare_calibration_input(self):
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        layers = self.model.model.layers

        # device = torch.device("cuda:0")
        if "model.embed_tokens" in self.model.hf_device_map:
            device = self.model.hf_device_map["model.embed_tokens"]

        dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros((self.n_samples, self.model.seqlen, self.model.config.hidden_size), dtype=dtype, device=self.device)
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
        for batch in self.dataloader_bench:
            try:
                self.model(batch[0].to(self.device))
            except ValueError:
                pass
        layers[0] = layers[0].module

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']
        self.model.config.use_cache = use_cache

        return inps, outs, attention_mask, position_ids


    def _extract_layer_information(self):
        self.param_list = []
        layers = get_layers(self.model)
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)
            for name in subset:
                self.param_list.append(get_layer_param(subset[name]))



    def _validate(self, model):
        # ppl = eval_ppl(model, self.tokenizer, self.device)
        acc = eval_acc(model, self.dataset)
        return acc

