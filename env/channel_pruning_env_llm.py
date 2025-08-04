import time
import random
import torch
import torch.nn as nn
from torch.nn import functional as f
from lib.utils import AverageMeter, accuracy, prGreen
from lib.arch import get_layers, get_mha_proj, get_ffn2, get_ffn1, get_mha
from lib.layerwrapper import WrappedGPT
from lib.eval import eval_ppl
from lib.data_utils import DataSaverHook, StopForwardException
from lib.data import get_loaders
from lib.Ridge import Ridge_Regression
from lib.linalg import lsmr_cupy_solver
from lib.mac import mac_per_head, mac_per_neuron, get_layer_param
from scipy.spatial import distance
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoConfig, OPTForCausalLM
from env.rewards import *
import math

import numpy as np
import copy

import sys  # 导入sys模块
sys.setrecursionlimit(10000)


class ChannelPruningEnv:
    """
    Env for channel pruning search
    """
    def __init__(self, model, data, preserve_ratio, args, n_data_worker=4,
                 batch_size=256, export_model=False, use_new_input=False):

        self.args = args
        self._get_model()

        self.device = torch.device("cuda:0")
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

        # options from args
        self.lbound = args.lbound
        self.rbound = args.rbound
        self.mask = [True, True, True]
        self.use_real_val = args.use_real_val
        self.n_samples = args.n_samples
        self.channel_round = args.channel_round
        self.acc_metric = args.acc_metric
        self.recon = args.recon

        self.export_model = export_model
        self.use_new_input = use_new_input

        # sanity check
        assert self.preserve_ratio > self.lbound, 'Error! You can make achieve preserve_ratio smaller than lbound!'

        # prepare data
        self._init_data()
  
        # extract information for preparing
        self._extract_layer_information()
        print('=> Initial min strategy dict: {}'.format(self.min_strategy_dict))

        # build reward
        self.reset()  # restore weight
        self.org_ppl = self._validate(self.model)
        print('=> original ppl: {:.3f}%'.format(self.org_ppl))
        # self.org_para = sum(self.wsize_list)
        # print('=> original weight size: {:.4f} M param'.format(self.org_para * 1. / 1e6))
        self.org_flops = sum(self.flops_list)
        print('=> FLOPs:')
        print([self.flops_list])
        print('=> original FLOPs: {:.4f} M'.format(self.org_flops))

        # if self.args.prune == 'para':
        #     self.expected_preserve_computation = self.preserve_ratio * self.org_para
        # elif self.args.prune == 'flops':
        self.expected_preserve_computation = self.preserve_ratio * self.org_flops
        # else:
        #     raise NotImplementedError

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
            self.tokenizer = AutoTokenizer.from_pretrained("./model/opt-125m", use_fast=False, force_download=False, resume_download=True)
        elif "llama" in self.args.model:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.args.model, use_fast=False, force_download=False, resume_download=True)

    def _get_model_local(self):

        self.model = AutoModelForCausalLM.from_pretrained(
            "./model/opt-125m",
            torch_dtype=torch.float16,
            cache_dir=self.args.cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        self.model.seqlen = 2048



    def step(self, action):
        # Pseudo prune and get the corresponding statistics. The real pruning happens till the end of all pseudo pruning
        idx = 2*self.layer_idx + int(not self.head)

        action = self._action_wall(action)
        preserve_ratio, d_prime = self.prune(action, self.layer_idx, self.head, idx)

        self.strategy.append(preserve_ratio)  # save action to strategy
        self.d_prime_list.append(d_prime)
        self.strategy_dict[idx] = preserve_ratio

        # update to next layer
        if self.head:
            self.head = False
        else:
            self.head = True
            self.layer_idx += 1

        if self._is_final_layer():
            assert len(self.strategy) == self.num_hidden_layers*2
            current_flops = self._cur_flops()
            compress_ratio = current_flops * 1. / self.org_flops

            ppl = self._validate(self.model)
            reward = self.reward(ppl)

            info_set = {'compress_ratio': compress_ratio, 'ppl': ppl, 'strategy': self.strategy.copy()}
            obs = self.layer_embedding

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_strategy = self.strategy.copy()
                self.best_d_prime_list = self.d_prime_list.copy()
                prGreen(
                    'New best reward: {:.4f}, ppl: {:.4f}, compress: {:.4f}'.format(self.best_reward, ppl, compress_ratio))
                prGreen('New best policy: {}'.format(self.best_strategy))
                prGreen('New best d primes: {}'.format(self.best_d_prime_list))
                torch.save(self.model.state_dict(), self.export_path)

            done = True
            self._get_model_local()

            return obs, reward, done, info_set


        obs = self._build_observe(self.layer_idx, self.head)
        obs[-1] = preserve_ratio
        self.layer_embedding = obs

        info_set = None
        reward = 0
        done = False

        return obs, reward, done, info_set

    def create_feat_scaleing_attn(self, feat, ind, num):
        num_neurons = ind.shape[0]
        scaling_mat = np.zeros([num, num_neurons])

        for i in range(scaling_mat.shape[0]):
            if i in ind:  # chosen
                ind_i, = np.where(ind == i)
                assert (len(ind_i) == 1)  # check if only one index is found
                scaling_mat[i, ind_i] = 1
            else:  # not chosen
                preserved = feat[0][:, :, ind].squeeze()  # torch.zeros(num_neurons, num_neurons).cuda()
                pruned = feat[0][:, :, i].squeeze().unsqueeze(1)  # torch.zeros(num_neurons).cuda()
                A = preserved
                B = pruned

                for feature in feat[1:]:
                    preserved = feature[:, :, ind].squeeze()
                    pruned = feature[:, :, i].squeeze().unsqueeze(1)
                    ATA = preserved
                    ATB = pruned
                    A = torch.cat((A, ATA), 0)
                    B = torch.cat((B, ATB), 0)

                B = B.squeeze()
                # linear_cif = Ridge_Regression(A, B, alpha=[1])
                # scale_factor = linear_cif.fit()
                scale_factor, success = lsmr_cupy_solver(A.float().cpu(), B.float().cpu())
                # solver = Ridge(alpha=0.8, fit_intercept=False)
                # solver.fit(A.cpu(), B.cpu())
                # scale_factor = solver.coef_

                if scale_factor.max() > 10 or scale_factor.min() < -10:
                    print(scale_factor.max(), scale_factor.min())
                else:
                    for index, chosen_i in enumerate(ind):
                        scaling_mat[i, index] = scale_factor[index]

        return scaling_mat

    def create_feat_scaleing_ffn(self, feat, ind):

        num_neurons = ind.shape[0]
        scaling_mat = np.zeros([feat[0].shape[1], num_neurons])

        for i in range(scaling_mat.shape[0]):
            if i in ind:  # chosen
                ind_i, = np.where(ind == i)
                assert (len(ind_i) == 1)  # check if only one index is found
                scaling_mat[i, ind_i] = 1
            else:  # not chosen
                preserved = feat[0][:, ind]  # torch.zeros(num_neurons, num_neurons).cuda()
                pruned = feat[0][:, i].unsqueeze(1)  # torch.zeros(num_neurons).cuda()
                A = preserved
                B = pruned

                for feature in feat[1:]:
                    preserved = feature[:, ind]
                    pruned = feature[:, i].unsqueeze(1)
                    ATA = preserved
                    ATB = pruned
                    A = torch.cat((A, ATA), 0)
                    B = torch.cat((B, ATB), 0)

                B = B.squeeze()
                scale_factor, success = lsmr_cupy_solver(A.float().cpu(), B.float().cpu())
                # solver = Ridge(alpha=0.8, fit_intercept=False)
                # solver.fit(A.cpu(), B.cpu())
                # scale_factor = solver.coef_

                if scale_factor.max() > 10 or scale_factor.min() < -10:
                    print(scale_factor.max(), scale_factor.min())
                else:
                    for index, chosen_i in enumerate(ind):
                        scaling_mat[i, index] = scale_factor[index]
        return scaling_mat

    def create_weight_scaleing_ffn(self, feat, ind):

        num_neurons = ind.shape[0]
        scaling_mat = np.zeros([feat.shape[0], num_neurons])

        for i in range(scaling_mat.shape[0]):
            if i in ind:  # chosen
                ind_i, = np.where(ind == i)
                assert (len(ind_i) == 1)  # check if only one index is found
                scaling_mat[i, ind_i] = 1
            else:  # not chosen
                preserved = feat[ind, :]
                pruned = feat[i, :].unsqueeze(0)
                A = preserved.t()
                B = pruned.t()

                B = B.squeeze()
                scale_factor, success = lsmr_cupy_solver(A.float().cpu(), B.float().cpu())

                if scale_factor.max() > 10 or scale_factor.min() < -10:
                    print(scale_factor.max(), scale_factor.min())
                else:
                    for index, chosen_i in enumerate(ind):
                        scaling_mat[i, index] = scale_factor[index]
        return scaling_mat

    def prune(self, preserve_ratio, idx, head, global_idx):

        def format_rank(x):
            rank = int(np.around(x))
            return max(rank, 1)

        assert (preserve_ratio <= 1.)

        if preserve_ratio == 1:
            if head:
                return preserve_ratio, self.num_attention_heads
            else:
                return preserve_ratio, self.hidden_size

        if head:
            attn = get_mha(self.model, idx)

            d_prime = format_rank(preserve_ratio * self.num_attention_heads)
            ratio = d_prime / self.num_attention_heads
            head_metric = self.A_metric[global_idx]
            head_metric = head_metric.reshape(self.num_attention_heads, -1)
            head_metric = torch.sum(head_metric, dim=-1)
            sorted_idx = torch.sort(-head_metric)
            preserve_idx = sorted_idx.indices[:d_prime]  # to preserve index

            mask = torch.zeros_like(head_metric, dtype=bool)
            mask[preserve_idx] = True

            attn.num_heads = d_prime
            attn.embed_dim = self.attention_head_size * d_prime

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
                scale_map = self.create_feat_scaleing_attn(self.feature[global_idx], np.array(proj_idx.cpu()), self.hidden_size)
                scale_map = torch.from_numpy(scale_map).type(dtype=torch.float).cuda()
                scale_map = scale_map.t()

                weight = attn.out_proj.weight.data.clone().detach()
                attn.out_proj.weight.data = weight[:, mask_proj]
                for i, Cin in enumerate(weight):
                    Out = Cin.reshape(Cin.shape[0], -1).float()
                    Out = torch.mm(scale_map, Out).reshape(-1)
                    attn.out_proj.weight.data[i, :] = Out
            else:
                weight = attn.out_proj.weight.data.reshape(-1, self.attention_head_size, self.num_attention_heads)[:, :,
                         mask]
                attn.out_proj.weight.data = weight.reshape(weight.shape[0], -1)

        else:
            pre_layer = get_ffn1(self.model, idx)
            target_layer = get_ffn2(self.model, idx)
            d_prime = format_rank(preserve_ratio * target_layer.weight.data.shape[1])
            ratio = d_prime / target_layer.weight.data.shape[1]

            hidden_metric = self.A_metric[global_idx]
            # hidden_metric = torch.abs(target_layer.weight.data).sum(dim=0)
            sorted_idx = torch.sort(-hidden_metric)
            preserve_idx = sorted_idx.indices[:d_prime]  # to preserve index

            mask = torch.zeros_like(hidden_metric, dtype=bool)
            mask[preserve_idx] = True

            if self.recon:
                scale_map = self.create_feat_scaleing_ffn(self.feature[global_idx], np.array(preserve_idx.cpu()))
                # scale_map = self.create_weight_scaleing_ffn(pre_layer.weight.data, np.array(preserve_idx.cpu()))
                scale_map = torch.from_numpy(scale_map).type(dtype=torch.float).cuda()
                scale_map = scale_map.t()

                weight = target_layer.weight.data.clone().detach()
                target_layer.weight.data = weight[:, mask]
                for i, Cin in enumerate(weight):
                    Out = Cin.reshape(Cin.shape[0], -1).float()
                    Out = torch.mm(scale_map, Out).reshape(-1)
                    target_layer.weight.data[i, :] = Out
            else:
                target_layer.weight.data = target_layer.weight.data[:, mask]

            pre_layer.weight.data = pre_layer.weight.data[mask, :]
            if pre_layer.bias is not None:
                pre_layer.bias.data = pre_layer.bias.data[mask]

        return ratio, d_prime


    def reset(self):
        self.layer_idx = 0
        self.head = True

        self.strategy = []  # pruning strategy
        self.d_prime_list = []
        self.strategy_dict = copy.deepcopy(self.min_strategy_dict)

        obs = self._build_observe(self.layer_idx, self.head)
        self.layer_embedding = obs.copy()
        return obs


    def hijack_input(self, module, list_to_append):
        hook = lambda _, inputs: list_to_append.append(inputs)
        handle = module.register_forward_pre_hook(hook)
        return handle


    def _build_observe(self, idx, head):
        obs = []
        obs.append(idx)
        layer = get_layers(self.model)[idx]

        if head:
            target_layer = get_mha_proj(self.model, idx)
            obs.append(1)  # head or ffn
            obs.append(target_layer.in_features*1./1e3)
            obs.append(target_layer.out_features*1./1e3)
            obs.append(self.num_attention_heads*1./1e2)  # head number
            obs.append(self.flops_att_list[idx]*1./1e4)  # flops
        else:
            target_layer = get_ffn2(self.model, idx)
            obs.append(0)
            obs.append(target_layer.in_features*1./1e3)
            obs.append(target_layer.out_features*1./1e3)
            obs.append(0)
            obs.append(self.flops_ffn_list[idx]*1./1e4)

        def add_batch():
            def tmp(_, inp, out):
                wrapped_layers.add_batch(inp[0].data, out.data)
            return tmp

        if head and idx > 0:
            pre_layer = get_layers(self.model)[idx-1]
            for j in range(self.n_samples):
                with torch.no_grad():
                    if "OPT" in self.model.__class__.__name__:
                        self.outs[j] = pre_layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask)[0]
                    else:
                        self.outs[j] = pre_layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask,
                                             position_ids=self.position_ids)[0]
            self.inps = self.outs
        elif head and idx == 0:
            with torch.no_grad():
                if "OPT" in self.model.__class__.__name__:
                    print('Experiments with OPT models')
                    self.inps, self.outs, self.attention_mask, self.position_ids = self.prepare_calibration_input_opt()
                else:
                    self.inps, self.outs, self.attention_mask, self.position_ids = self.prepare_calibration_input()

        handles = []
        wrapped_layers = WrappedGPT(target_layer)
        handles.append(target_layer.register_forward_hook(add_batch()))
        for j in range(self.n_samples):
            with torch.no_grad():
                if "OPT" in self.model.__class__.__name__:
                    self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask)[0]
                else:
                    self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask, position_ids=self.position_ids)[0]
        for h in handles:
            h.remove()
        # self.A_metric = torch.sqrt(wrapped_layers.scaler_row)
        self.W_metric = torch.abs(target_layer.weight.data) * torch.sqrt(wrapped_layers.scaler_row.reshape((1, -1)))

        data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)
        if head:
            # pre_ffn_layer = get_ffn2(self.model, idx)
            handles_inputs = target_layer.register_forward_hook(data_saver)
            inputs = []
            for j in range(self.n_samples):
                with torch.no_grad():
                    try:
                        if "OPT" in self.model.__class__.__name__:
                            self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask)[0]
                        else:
                            self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask,
                                                 position_ids=self.position_ids)[0]
                    except StopForwardException:
                        pass
                    inputs.append(data_saver.input_store[0].detach())
            handles_inputs.remove()
            self.feat_mha = random.sample(inputs, self.args.recon_sample)
        # else:
        #     # pre_ffn_layer = get_ffn2(self.model, idx)
        #     handles_inputs = target_layer.register_forward_hook(data_saver)
        #     outputs = []
        #     for j in range(self.n_samples):
        #         with torch.no_grad():
        #             try:
        #                 if "OPT" in self.model.__class__.__name__:
        #                     self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask)[0]
        #                 else:
        #                     self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask, position_ids=self.position_ids)[0]
        #             except StopForwardException:
        #                 pass
        #             outputs.append(data_saver.input_store[0].detach())
        #     handles_inputs.remove()
        #     self.feat_ffn = random.sample(outputs, self.args.recon_sample)

        out_ratio_layer = self.check_outlier_mean(self.W_metric, 7)
        obs.append(out_ratio_layer)
        obs.append(0)  # a t-1

        obs = np.array(obs, dtype=np.float32)
        return obs


    def set_export_path(self, path):
        self.export_path = path

    def _is_final_layer(self):
        return self.layer_idx == self.num_hidden_layers

    def _action_wall(self, action):
        idx = 2*self.layer_idx + int(not self.head)
        assert len(self.strategy) == idx

        action = float(action)
        action = np.clip(action, 0, 1)

        other_comp = 0
        this_comp = 0
        for i in range(self.num_hidden_layers*2):
            if i == idx: # this layer
                this_comp += self.flops_list[i]
            else:
                other_comp += self.strategy_dict[i] * self.flops_list[i]

        total_flops = self.org_flops * self.preserve_ratio
        max_preserve_ratio = (total_flops - other_comp) * 1. / this_comp

        action = np.minimum(action, max_preserve_ratio)
        action = np.maximum(action, self.strategy_dict[idx])

        return action

    def _cur_flops(self):
        flops = 0
        for i in range(self.num_hidden_layers*2):
            flops += self.strategy_dict[i] * self.flops_list[i]
        return flops


    def _init_data(self):
        self.dataloader, _ = get_loaders(self.dataset, nsamples=self.n_samples, seed=self.args.seed, seqlen=2048, tokenizer=self.tokenizer)

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
        for batch in self.dataloader:
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
        for batch in self.dataloader:
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
        with torch.no_grad():
            if "OPT" in self.model.__class__.__name__:
                print('Experiments with OPT models')
                self.inps, self.outs, self.attention_mask, self.position_ids = self.prepare_calibration_input_opt()
            else:
                self.inps, self.outs, self.attention_mask, self.position_ids = self.prepare_calibration_input()
        # self.wsize_list = []
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
            self.flops_att_list.append(attention_mac*1./1e6)
            self.flops_ffn_list.append(ffn_mac*1./1e6)
            self.flops_list.append(attention_mac * 1. /1e6)
            self.flops_list.append(ffn_mac * 1. / 1e6)
            self.dim_list.append(self.num_attention_heads)
            self.dim_list.append(self.intermediate_size)

        self.strategy_dict = {}
        for i in range(self.num_hidden_layers*2):
            self.strategy_dict[i] = self.lbound

        self.min_strategy_dict = copy.deepcopy(self.strategy_dict)


        self.A_metric = []
        self.recon_sample = []
        self.feature = []
        layers = get_layers(self.model)
        for i in range(len(layers)):
            layer = layers[i]
            mha_layer = get_mha_proj(self.model, i)
            ffn_layer = get_ffn2(self.model, i)

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
                        self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask,
                                             position_ids=self.position_ids)[
                            0]
            for h in handles:
                h.remove()

            for name in wrapped_layers:
                self.A_metric.append(torch.sqrt(wrapped_layers[name].scaler_row))

            data_saver_mha = DataSaverHook(store_input=True, store_output=True, stop_forward=False)
            handles_mha = mha_layer.register_forward_hook(data_saver_mha)
            inputs_mha = []

            data_saver_ffn = DataSaverHook(store_input=True, store_output=True, stop_forward=True)
            handles_ffn = ffn_layer.register_forward_hook(data_saver_ffn)
            inputs_ffn = []

            for j in range(self.n_samples):
                with torch.no_grad():
                    try:
                        if "OPT" in self.model.__class__.__name__:
                            self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask)[0]
                        else:
                            self.outs[j] = layer(self.inps[j].unsqueeze(0), attention_mask=self.attention_mask,
                                                 position_ids=self.position_ids)[0]
                    except StopForwardException:
                        pass
                    inputs_mha.append(data_saver_mha.input_store[0].detach())
                    inputs_ffn.append(data_saver_ffn.input_store[0].detach())

            handles_mha.remove()
            handles_ffn.remove()
            self.feature.append(random.sample(inputs_mha, self.args.recon_sample))
            self.feature.append(random.sample(inputs_ffn, self.args.recon_sample))

            self.inps = self.outs


    def _validate(self, model):
        ppl = eval_ppl(model, self.tokenizer, self.device)
        return ppl

