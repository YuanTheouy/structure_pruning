import time
import random
import torch
import torch.nn as nn
from torch.nn import functional as f
from lib.utils import AverageMeter, accuracy, prGreen
from lib.lm_eval.tasks import get_task_dict
from lib.arch import get_layers, get_mha_proj, get_ffn2, get_ffn1, get_mha,  get_down, get_up, get_gate
from lib.layerwrapper import WrappedGPT
from lib.eval import eval_ppl
from lib.data_utils import DataSaverHook, StopForwardException
from lib.data import get_loaders
from lib.Ridge import Ridge_Regression
from sklearn.metrics import pairwise_distances
from lib.linalg import lsmr_cupy_solver
from lib.mac import mac_per_head, mac_per_neuron, get_layer_param, get_norm_param
from scipy.spatial import distance
# import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoConfig, OPTForCausalLM
# from sklearn.linear_model import Ridge
from env.rewards import *
import math

import numpy as np
import cupy as cp

import sys  # 导入sys模块
sys.setrecursionlimit(10000)

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
        self.n_data_worker = n_data_worker
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
        self.recon_sample = args.recon_sample
        self.channel_round = args.channel_round
        self.acc_metric = args.acc_metric
        self.recon = args.recon

        self.export_model = export_model
        self.use_new_input = use_new_input


        # prepare data
        self._init_data_wiki()
        self._init_data()
  
        # extract information for preparing
        self._extract_layer_information()

        # build reward
        self.reset()  # restore weight
        self.org_ppl = 10#self._validate(self.model)
        print('=> original ppl: {:.3f}%'.format(self.org_ppl))
        self.org_para = sum(self.param_list)
        print('=> Params:')
        print(self.param_list)
        print('=> original weight size: {:.4f} M param'.format(self.org_para))
        self.org_flops = sum(self.flops_list)
        print('=> FLOPs:')
        print(self.flops_list)
        print('=> original FLOPs: {:.4f} M'.format(self.org_flops))

        if self.args.prune == 'para':
            self.expected_preserve_computation = self.preserve_ratio * self.org_para
        elif self.args.prune == 'flops':
            self.expected_preserve_computation = self.preserve_ratio * self.org_flops
        else:
            raise NotImplementedError

        self.reward = eval(args.reward)

        self.best_reward = -math.inf
        self.best_strategy = None
        self.best_d_prime_list = None



    def _get_model(self):
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
        # with torch.no_grad():
        #     if "OPT" in self.model.__class__.__name__:
        #         self.inps, self.outs, self.attention_mask, self.position_ids = self.prepare_calibration_input_opt()
        #     else:
        #         self.inps, self.outs, self.attention_mask, self.position_ids = self.prepare_calibration_input()
        count=0
        for r in self.action:
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

        assert len(self.action) == self.num_hidden_layers * 2
        current_flops = self._cur_flops(self.strategy)
        compress_ratio = current_flops * 1. / self.org_flops
        current_para = self._cur_para(self.strategy)
        para_ratio = current_para * 1. / self.org_para

        # checkpoint = torch.load(self.args.resume_path)
        # self.model.load_state_dict(checkpoint)

        # ppl = self._validate(self.model)
        # reward = self.reward(ppl)

        # info_set = {'compress_ratio': compress_ratio, 'para_ratio': para_ratio, 'ppl': ppl, 'strategy': self.action.copy()}
        # obs = np.array(self.preserve_ratio, dtype=np.float32)

        # if reward > self.best_reward:
            # self.best_reward = reward
            # self.best_strategy = self.action.copy()
            # self.best_d_prime_list = self.d_prime_list.copy()
            # prGreen(
            #     'New best reward: {:.4f}, ppl: {:.4f}, compress: {:.4f}, para: {:.4f}'.format(self.best_reward, ppl,
            #                                                                                   compress_ratio, para_ratio))
            # prGreen('New best policy: {}'.format(self.best_strategy))
            # prGreen('New best d primes: {}'.format(self.best_d_prime_list))
            # torch.save(self.model.state_dict(), self.export_path)
        return self.model

        # done = True

        # return obs, reward, done, info_set



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
        # feature = torch.mean(feature, dim=0).squeeze().to("cuda:0")
        feature = feature.reshape(-1, num).to("cuda:1")

        norm = torch.norm(feature, dim=0, p=2)
        theshold = torch.mean(norm) * self.args.m
        chosen_ind = []
        for i in range(num_neurons):
            if norm[ind[i]] < theshold:
                chosen_ind.append(i)
        chosen = ind[chosen_ind]

        for i in range(scaling_mat.shape[0]):
            if i in ind:  # chosen
                ind_i, = np.where(ind == i)
                assert (len(ind_i) == 1)  # check if only one index is found
                scaling_mat[i, ind_i] = 1
            else:  # not chosen
                A = feature[:, chosen]
                # A = feature[:, ind]
                B = feature[:, i]
                linear_cif = Ridge_Regression(A.to(dtype=torch.float32), B.to(dtype=torch.float32), alpha=0.9, fit_intercept=False)
                scale_factor = linear_cif.fit()

                torch.cuda.empty_cache()
                for index, chosen_i in enumerate(chosen_ind):
                    scaling_mat[i, chosen_i] = scale_factor[index]
                
                
                # for index, chosen_i in enumerate(ind):
                #     scaling_mat[i, index] = scale_factor[index]
                
        return scaling_mat


    def create_feat_scaleing_ffn(self, feat, ind, num, threshold):
        torch.cuda.empty_cache()
        num_neurons = ind.shape[0]
        scaling_mat = np.zeros([num, num_neurons])

        feature = torch.stack(feat)
        # feature = torch.mean(feature, dim=0).squeeze().to("cuda:0")
        feature = feature.reshape(-1, num).to("cuda:1")

        norm = torch.norm(feature, dim=0, p=2)
        theshold = torch.mean(norm) * self.args.m
        chosen_ind = []
        for i in range(num_neurons):
            if norm[ind[i]] < theshold:
                chosen_ind.append(i)
        chosen = ind[chosen_ind]

        for i in range(scaling_mat.shape[0]):
            if i in ind:  # chosen
                ind_i, = np.where(ind == i)
                assert (len(ind_i) == 1)  # check if only one index is found
                scaling_mat[i, ind_i] = 1
            else:  # not chosen
                B = feature[:, i]
                # A = feature[:, ind]

                A = feature[:, chosen]
                linear_cif = Ridge_Regression(A.to(dtype=torch.float32), B.to(dtype=torch.float32), alpha=0.9, fit_intercept=False)
                scale_factor = linear_cif.fit()

                torch.cuda.empty_cache()
                for index, chosen_i in enumerate(chosen_ind):
                    scaling_mat[i, chosen_i] = scale_factor[index]

                # for index, chosen_i in enumerate(ind):
                #     scaling_mat[i, index] = scale_factor[index]

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
                return preserve_ratio, self.num_attention_heads
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
            d_prime = format_rank(preserve_ratio * self.num_attention_heads)
            ratio = d_prime / self.num_attention_heads
            head_metric = self.A_metric[global_idx]
            head_metric = head_metric.reshape(self.num_attention_heads, -1)
            head_metric = torch.sum(head_metric, dim=-1)
            sorted_idx = torch.sort(-head_metric)
            preserve_idx = sorted_idx.indices[:d_prime]  # to preserve index
            preserve_idx,_ = torch.sort(preserve_idx)

            mask = torch.zeros_like(head_metric, dtype=bool)
            mask[preserve_idx] = True

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
                        inputs.append(data_saver.input_store[0].detach().to("cuda:1"))
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
                    proj_idx = mask_proj.nonzero().squeeze()
                    scale_map = self.create_feat_scaleing_attn(inputs, np.array(proj_idx.cpu()), self.hidden_size)
                    scale_map = torch.from_numpy(scale_map).type(dtype=torch.float).to("cuda:0")
                    scale_map = scale_map.t()
    
                    weight = attn.out_proj.weight.data.clone().detach()
                    attn.out_proj.weight.data = weight[:, mask_proj]
                    for i, Cin in enumerate(weight):
                        Out = Cin.reshape(Cin.shape[0], -1).float().to("cuda:0")
                        Out = torch.mm(scale_map, Out).reshape(-1)
                        attn.out_proj.weight.data[i, :] = Out.to(attn.out_proj.weight.data.device)
                else:
                    weight = attn.out_proj.weight.data.reshape(-1, self.attention_head_size, self.num_attention_heads)[:, :,
                             mask]
                    attn.out_proj.weight.data = weight.reshape(weight.shape[0], -1)
            else:
                attn.num_heads = d_prime
                attn.num_key_value_heads = d_prime
                attn.hidden_size = attn.head_dim * d_prime
                attn.max_position_embeddings = attn.head_dim * d_prime
    
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
                    proj_idx = mask_proj.nonzero().squeeze()
                    scale_map = self.create_feat_scaleing_attn(inputs, np.array(proj_idx.cpu()), self.hidden_size)
                    scale_map = torch.from_numpy(scale_map).type(dtype=torch.float).to("cuda:0")
                    scale_map = scale_map.t()
    
                    weight = attn.o_proj.weight.data.clone().detach()
                    attn.o_proj.weight.data = weight[:, mask_proj]
                    for i, Cin in enumerate(weight):
                        Out = Cin.reshape(Cin.shape[0], -1).float().to("cuda:0")
                        Out = torch.mm(scale_map, Out).reshape(-1)
                        attn.o_proj.weight.data[i, :] = Out.to(attn.o_proj.weight.device)
                else:
                    # attn.o_proj.weight.data = attn.o_proj.weight.data.cuda()
                    weight = attn.o_proj.weight.data.reshape(-1, self.attention_head_size, self.num_attention_heads)[:, :,
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

            mask = torch.zeros_like(hidden_metric, dtype=bool, device='cuda')
            mask[preserve_idx] = True

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
                        inputs.append(data_saver.input_store[0].detach().to("cuda:1"))
                handles_inputs.remove()

                # inputs = random.sample(inputs,24)
                
                scale_map = self.create_feat_scaleing_ffn(inputs, np.array(preserve_idx.cpu()), self.intermediate_size, 0.01)
                scale_map = torch.from_numpy(scale_map).type(dtype=torch.float).to("cuda:0")
                scale_map = scale_map.t()

                weight = target_layer.weight.data.clone().detach()
                target_layer.weight.data = weight[:, mask]
                for i, Cin in enumerate(weight):
                    Out = Cin.reshape(Cin.shape[0], -1).float().to("cuda:0")
                    Out = torch.mm(scale_map, Out).reshape(-1)
                    target_layer.weight.data[i, :] = Out.to(target_layer.weight)
            else:
                # target_layer.weight.data = target_layer.weight.data.cuda()
                # print(mask.device)
                # print(target_layer.weight.data.device)
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

        print(idx, head)
        torch.save(self.model.state_dict(), self.export_path)


        return ratio, d_prime


    def reset(self):
        self.layer_idx = 0
        self.head = True

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
        if self.export_model:
            return action

        actions = np.abs(action)
        actions = np.clip(actions, 0, 1)

        def format_rank(x):
            rank = int(np.around(x))
            return max(rank, 1)

        for i in range(len(self.dim_list)):
            d_prime = format_rank(actions[i] * self.dim_list[i])
            d_prime = int(np.ceil(d_prime * 1. / self.channel_round) * self.channel_round)
            actions[i] = d_prime / self.dim_list[i]

        actions = actions.clip(self.lbound, self.rbound)

        for idx in range(len(actions)):
            other_comp = 0
            this_comp = 0
            for i in range(self.num_hidden_layers * 2):
                if self.args.prune == 'flops':
                    if i == idx:  # this layer
                        this_comp += self.flops_list[i]
                    elif i < idx:
                        other_comp += actions[i] * self.flops_list[i]
                    else:
                        other_comp += self.lbound * self.flops_list[i]
                elif self.args.prune == 'para':
                    if i == idx:  # this layer
                        this_comp += self.param_list[i]
                    elif i < idx:
                        other_comp += actions[i] * (self.param_list[i] - self.norm_para[i]) + self.norm_para[i]
                    else:
                        other_comp += self.lbound * (self.param_list[i] - self.norm_para[i]) + self.norm_para[i]

            max_preserve_ratio = (self.expected_preserve_computation - other_comp) * 1. / this_comp
            actions[idx] = np.minimum(actions[idx], max_preserve_ratio)
            actions[idx] = np.maximum(actions[idx], self.lbound)

        return list(actions)

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

    def _init_data_wiki(self):
        self.dataloader, _ = get_loaders('wikitext2', nsamples=self.n_samples, seed=self.args.seed, seqlen=2048, tokenizer=self.tokenizer)

    def _init_data(self):
        print(self.dataset)
        dataloader = []
        seqlen=2048
        nsamples=self.recon_sample
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

        # print(self.dataloader)
        # raise RuntimeError

       
    def prepare_calibration_input_opt(self, dataset, n_samples):
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        if "OPT" in self.model.__class__.__name__:
            layers = self.model.model.decoder.layers
        else:
            layers = self.model.model.layers
        
        torch.cuda.empty_cache()

        # 使用动态设备或从模型的device_map中获取
        device = self.device if hasattr(self, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(self.model, 'hf_device_map') and "model.embed_tokens" in self.model.hf_device_map:
            device = self.model.hf_device_map["model.embed_tokens"]

        dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros((n_samples, self.model.seqlen, self.model.config.hidden_size), dtype=dtype, device=device)
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
        for batch in dataset:
            try:
                self.model(batch[0].to(device))
            except ValueError:
                pass
        layers[0] = layers[0].module

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        self.model.config.use_cache = use_cache

        position_ids = None

        return inps, outs, attention_mask, position_ids

    def prepare_calibration_input(self, dataset, n_samples):
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        layers = self.model.model.layers

        torch.cuda.empty_cache()

        # 使用动态设备或从模型的device_map中获取
        device = self.device if hasattr(self, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(self.model, 'hf_device_map') and "model.embed_tokens" in self.model.hf_device_map:
            device = self.model.hf_device_map["model.embed_tokens"]

        dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros((n_samples, self.model.seqlen, self.model.config.hidden_size), dtype=dtype, device=device)
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
        for batch in dataset:
            try:
                self.model(batch[0].to(device))
            except ValueError:
                pass
        layers[0] = layers[0].module

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']
        self.model.config.use_cache = use_cache

        return inps, outs, attention_mask, position_ids


    def _extract_layer_information(self):
        with torch.no_grad():
            if "OPT" in self.model.__class__.__name__:
                print('Experiments with OPT models')
                self.inps, self.outs, self.attention_mask, self.position_ids = self.prepare_calibration_input_opt(self.dataloader, self.n_samples)
                self.recon_inps, self.recon_outs, self.attention_mask_bench, self.position_ids_bench = self.prepare_calibration_input_opt(self.dataloader_bench, self.recon_sample)
            else:
                self.inps, self.outs, self.attention_mask, self.position_ids = self.prepare_calibration_input(self.dataloader, self.n_samples)
                self.recon_inps, self.recon_outs, self.attention_mask_bench, self.position_ids_bench = self.prepare_calibration_input(self.dataloader_bench, self.recon_sample)
        self.flops_att_list = []
        self.flops_ffn_list = []
        self.flops_list = []
        self.dim_list = []
        num_heads_per_layer = [self.num_attention_heads] * self.num_hidden_layers
        num_neurons_per_layer = [self.intermediate_size] * self.num_hidden_layers
        for num_heads, num_neurons in zip(num_heads_per_layer, num_neurons_per_layer):
            attention_mac = num_heads * mac_per_head(self.model.seqlen, self.hidden_size,  self.attention_head_size)
            ffn_mac = num_neurons * mac_per_neuron(self.model.seqlen,  self.hidden_size)
            # mac = attention_mac + ffn_mac
            self.flops_att_list.append(attention_mac * 1. /1e6)
            self.flops_ffn_list.append(ffn_mac * 1./ 1e6)
            self.flops_list.append(attention_mac * 1. / 1e6)
            self.flops_list.append(ffn_mac * 1. / 1e6)
            self.dim_list.append(self.num_attention_heads)
            self.dim_list.append(self.intermediate_size)

        self.A_metric = []
        # self.recon_sample = []
        self.param_list = []
        self.norm_para = []
        self.feat = []
        layers = get_layers(self.model)
        idx = torch.randperm(self.n_samples)[:self.args.recon_sample].cpu()
        # if self.recon:
        #     self.recon_inps = self.inps[idx]
        #     self.recon_outs = self.outs[idx]
            # self.attention_mask_bench = self.attention_mask[idx]
            # self.position_ids_bench = self.position_ids[idx]
        for i in range(len(layers)):
            layer = layers[i]
            mha = get_mha(self.model, i)
            mha_layer = get_mha_proj(self.model, i)
            if "OPT" in self.model.__class__.__name__:
                ffn_layer = get_ffn2(self.model, i)
            else:
                ffn_layer = get_down(self.model, i)

            mha_para = get_layer_param(mha)
            ffn_para = get_layer_param(layer) - mha_para

            mha_norm = get_norm_param(mha)
            ffn_norm = get_norm_param(layer) - mha_norm

            self.param_list.append(mha_para * 1. / 1e6)
            self.param_list.append(ffn_para * 1. / 1e6)

            self.norm_para.append(mha_norm * 1. / 1e6)
            self.norm_para.append(ffn_norm * 1. / 1e6)

            subset = {}
            subset.update({'mha': mha_layer})
            subset.update({'ffn': ffn_layer})
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

            for j in range(self.n_samples):
                with torch.no_grad():
                    if "OPT" in self.model.__class__.__name__:
                        self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask)[0]
                    else:
                        self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask, position_ids=self.position_ids)[
                            0]

            for h in handles:
                h.remove()

            for name in wrapped_layers:
                self.A_metric.append(torch.sqrt(wrapped_layers[name].scaler_row))
            
            # if self.recon:
            #     data_saver = DataSaverHook(store_input=True, store_output=False, stop_forward=True)
            #     handles_inputs = mha_layer.register_forward_hook(data_saver)
            #     inputs = []
            #     data_saver_ffn = DataSaverHook(store_input=True, store_output=True, stop_forward=True)
            #     handles_inputs_ffn = ffn_layer.register_forward_hook(data_saver_ffn)
            #     inputs_ffn = []
            #     for j in idx:
            #         with torch.no_grad():
            #             try:
            #                 if "OPT" in self.model.__class__.__name__:
            #                     self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask)[0]
            #                 else:
            #                     self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask,
            #                                          position_ids=self.position_ids)[0]
            #             except StopForwardException:
            #                 pass
            #             inputs.append(data_saver.input_store[0].detach())
            #             inputs_ffn.append(data_saver_ffn.input_store[0].detach())
            #     handles_inputs.remove()
            #     handles_inputs_ffn.remove()
            #     self.feat.append(inputs)
            #     self.feat.append(inputs_ffn)

            self.inps = self.outs

    def _validate(self, model):
        ppl = eval_ppl(model, self.tokenizer)
        return ppl
