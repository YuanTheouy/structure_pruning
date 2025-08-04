# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import time
import torch
import torch.nn as nn
from torch.nn import functional as f
from lib.utils import AverageMeter, accuracy, prGreen
from lib.data import get_split_dataset
from memory.modelsize_estimate import modelsize
from models.mobilenet_v2 import InvertedResidual
from models.resnet import BasicBlock, LambdaLayer
from env.rewards import *
import math

import numpy as np
import copy

# import sys
# sys.setrecursionlimit(50000)

class ChannelPruningEnv:
    """
    Env for channel pruning search
    """
    def __init__(self, model, checkpoint, modelset, data, preserve_ratio, args, n_data_worker=4,
                 batch_size=256, export_model=False, use_new_input=False):
        # default setting
        self.prunable_layer_types = [torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear]#,LambdaLayer]

        # save options
        self.model = model
        self.checkpoint = checkpoint
        self.n_data_worker = n_data_worker
        self.batch_size = batch_size
        self.model_type = modelset
        self.data_type = data
        self.preserve_ratio = preserve_ratio

        # options from args
        self.args = args
        self.lbound = args.lbound
        self.rbound = args.rbound
        self.mask = [True, True, True]

        self.use_real_val = args.use_real_val

        self.n_calibration_batches = args.n_calibration_batches
        self.n_points_per_layer = args.n_points_per_layer
        self.channel_round = args.channel_round
        self.acc_metric = args.acc_metric
        self.data_root = args.data_root

        self.export_model = export_model
        self.use_new_input = use_new_input

        # sanity check
        assert self.preserve_ratio > self.lbound, 'Error! You can make achieve preserve_ratio smaller than lbound!'

        # prepare data
        self._init_data()
        if not self.export_model:
            self._get_model()

        # build indexs
        self._build_index()
        self.n_prunable_layer = len(self.prunable_idx)
        

  
        # extract information for preparing
        self._extract_layer_information()

            # build embedding (static part)
        self._build_state_embedding()

            # build reward
        self.reset()  # restore weight
        
        self.org_acc, self.org_time, self.org_memory = self._validate(self.val_loader, self.model)
        print('=> original acc: {:.3f}%'.format(self.org_acc))
        print('=> original time: {:.4f}ms'.format(self.org_time))
        print('=> original memory: {:.4f}MB'.format(self.org_memory))
        self.org_para = sum(self.wsize_list)
        print('=> params:')
        print([self.layer_info_dict[idx]['params'] for idx in sorted(self.layer_info_dict.keys())])
        print('=> original weight size: {:.4f} param'.format(self.org_para * 1.))

        self.org_flops = sum(self.flops_list)
        print('=> FLOPs:')
        print([self.layer_info_dict[idx]['flops'] / 1e6 for idx in sorted(self.layer_info_dict.keys())])
        print('=> original FLOPs: {:.4f} M'.format(self.org_flops * 1. / 1e6))

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

        self.org_w_size = sum(self.wsize_list)


    def _get_model(self):
        if self.model_type == 'mobilenet':
            if self.data_type == 'imagenet':
                from models.mobilenet import MobileNet
                self.model = MobileNet(n_class=1000)
            elif self.data_type == 'cifar10':
                from models.mobilenet import MobileNet
                self.model = MobileNet(n_class=10)
            elif self.data_type == 'cifar100':
                from models.mobilenet import MobileNet
                self.model = MobileNet(n_class=100)
            else:
                raise NotImplementedError
        elif self.model_type =='shufflenet':
            if self.data_type == 'imagenet':
                from models.ShuffleNetV2 import ShuffleNetV2
                self.model = ShuffleNetV2(n_class=1000,input_size=224)
            elif self.data_type == 'cifar10':
                from models.ShuffleNetV2 import ShuffleNetV2
                self.model = ShuffleNetV2(n_class=10,input_size=32)
            elif self.data_type == 'cifar100':
                from models.ShuffleNetV2 import ShuffleNetV2
                self.model = ShuffleNetV2(n_class=100,input_size=32)
            else:
                raise NotImplementedError
        elif self.model_type =='mobilenetv2':
            if self.data_type == 'imagenet':
                from models.mobilenet_v2 import MobileNetV2
                self.model = MobileNetV2(n_class=1000,input_size=224)
            elif self.data_type == 'cifar10':
                from models.mobilenet_v2 import MobileNetV2
                self.model = MobileNetV2(n_class=10,input_size=32)
            elif self.data_type == 'cifar100':
                from models.mobilenet_v2 import MobileNetV2
                self.model = MobileNetV2(n_class=100,input_size=32)
            else:
                raise NotImplementedError
        elif self.model_type =='resnet18':
            if self.data_type == 'imagenet':
                from models.resnet18 import ResNet18
                self.model_type = ResNet18(n_class=1000)
            elif self.data_type == 'cifar10':
                from models.resnet18 import ResNet18
                self.model_type = ResNet18(n_class=10)
            elif self.data_type == 'cifar100':
                from models.resnet18 import ResNet18
                self.model_type = ResNet18(n_class=100)
            else:
                raise NotImplementedError
        elif self.model_type =='vgg':
            if self.data_type == 'imagenet':
                from models.vgg import VGG
                self.model = VGG(n_class=1000)
            elif self.data_type == 'cifar10':
                from models.vgg import VGG
                self.model = VGG(n_class=10)
            elif self.data_type == 'cifar100':
                from models.vgg import VGG
                self.model = VGG(n_class=100)
            else:
                raise NotImplementedError
        elif self.model_type =='resnet':
            if self.data_type == 'imagenet':
                from models.resnet import resnet56
                self.model = resnet56()
            elif self.data_type == 'cifar10':
                from models.resnet import resnet56
                self.model = resnet56()
            elif self.data_type == 'cifar100':
                from models.resnet import resnet56
                self.model = resnet56()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        self.model.load_state_dict(self.checkpoint)
        self.model = self.model.cuda()


    def step(self, action):
        # Pseudo prune and get the corresponding statistics. The real pruning happens till the end of all pseudo pruning
        if self.visited[self.cur_ind]:
            action = self.strategy_dict[self.prunable_idx[self.cur_ind]][0]
            preserve_idx = self.index_buffer[self.cur_ind]
        else:
            if self.args.prune == 'para':
                action = self._action_wall_para(action)
            elif self.args.prune == 'flops':
                action = self._action_wall(action)  # percentage to preserve
            else:
                raise NotImplementedError
            preserve_idx = None


        # prune and update action
        if self.args.model == 'mobilenet' or self.args.model == 'mobilenetv2':
            action, d_prime, preserve_idx, mask = self.prune_kernel_mobilenet(self.prunable_idx[self.cur_ind], action, preserve_idx,last_mask=self.mask)
        else:
            action, d_prime, preserve_idx, mask = self.prune_kernel(self.prunable_idx[self.cur_ind], action, preserve_idx,last_mask=self.mask)
        self.mask = mask

        if not self.visited[self.cur_ind]:
            for group in self.shared_idx:
                if self.cur_ind in group:  # set the shared ones
                    for g_idx in group:
                        self.strategy_dict[self.prunable_idx[g_idx]][0] = action
                        self.strategy_dict[self.prunable_idx[g_idx - 1]][1] = action
                        self.visited[g_idx] = True
                        self.index_buffer[g_idx] = preserve_idx.copy()

        if self.export_model:  # export checkpoint
            print('# Pruning {}: ratio: {}, d_prime: {}'.format(self.cur_ind, action, d_prime))

        self.strategy.append(action)  # save action to strategy
        self.d_prime_list.append(d_prime)

        self.strategy_dict[self.prunable_idx[self.cur_ind]][0] = action
        if self.cur_ind > 0:
            self.strategy_dict[self.prunable_idx[self.cur_ind - 1]][1] = action

        # all the actions are made
        if self._is_final_layer():

            assert len(self.strategy) == len(self.prunable_idx)
            current_flops = self._cur_flops()
            current_para = self._cur_para()
            compress_ratio = current_flops * 1. / self.org_flops
            para_ratio = current_para * 1. / self.org_para

            if self.export_model:  # export state dict
                prGreen('compress: {:.4f}, para: {:.4f}'.format(compress_ratio, para_ratio))
                return None, None, None, None

            acc,val_time,val_memory = self._validate(self.val_loader, self.model)
            info_set = {'compress_ratio': compress_ratio, 'para_ratio': para_ratio, 'accuracy': acc, 'strategy': self.strategy.copy(), 'time': val_time,  'memory': val_memory}
            reward = self.reward(self, acc/100, current_flops, val_time/self.org_time, val_memory/self.org_memory)

            if self.export_model:  # export state dict
                prGreen('acc: {:.4f}, compress: {:.4f}, para: {:.4f}'.format(acc, compress_ratio, para_ratio))
                return None, None, None, None

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_strategy = self.strategy.copy()
                self.best_d_prime_list = self.d_prime_list.copy()
                prGreen('New best reward: {:.4f}, acc: {:.4f}, compress: {:.4f}'.format(self.best_reward, acc, compress_ratio))
                prGreen('New best policy: {}'.format(self.best_strategy))
                prGreen('New best d primes: {}'.format(self.best_d_prime_list))
                torch.save(self.model.state_dict(), self.export_path)

            obs = self.layer_embedding[self.cur_ind, :].copy()  # actually the same as the last state
            done = True

            self._get_model()
            
            # if self.export_model:  # export state dict
            #     torch.save(self.model.state_dict(), self.export_path)
            #     return None, None, None, None
            return obs, reward, done, info_set

        info_set = None
        reward = 0
        done = False
        self.visited[self.cur_ind] = True  # set to visited
        self.cur_ind += 1  # the index of next layer
        if self.args.prune == 'para':
            self.layer_embedding[self.cur_ind][-1] = self.strategy[-1]  # last action
            self.layer_embedding[self.cur_ind][-2] = sum(self.wsize_list[self.cur_ind + 1:]) * 1. / self.org_para  # rest
            self.layer_embedding[self.cur_ind][-3] = self._cur_reduced_para() * 1. / self.org_para  # reduced
        elif self.args.prune == 'flops':
            self.layer_embedding[self.cur_ind][-1] = self.strategy[-1]  # last action
            self.layer_embedding[self.cur_ind][-2] = sum(self.flops_list[self.cur_ind + 1:]) * 1. / self.org_flops  # rest
            self.layer_embedding[self.cur_ind][-3] = self._cur_reduced() * 1. / self.org_flops  # reduced
        else:
            raise NotImplementedError
        obs = self.layer_embedding[self.cur_ind, :].copy()

        return obs, reward, done, info_set

    def reset(self):
        # restore env by loading the checkpoint
        self.model.load_state_dict(self.checkpoint)
        self.cur_ind = 0
        self.strategy = []  # pruning strategy
        self.d_prime_list = []
        self.strategy_dict = copy.deepcopy(self.min_strategy_dict)
        # reset layer embeddings
        # self.layer_embedding[:, -1] = 1.
        # self.layer_embedding[:, -2] = 0.
        # self.layer_embedding[:, -3] = 0.
        obs = self.layer_embedding[0].copy()
        obs[-2] = sum(self.wsize_list[1:]) * 1. / sum(self.wsize_list)
        self.extract_time = 0
        self.fit_time = 0

        # for share index
        self.visited = [False] * len(self.prunable_idx)
        self.index_buffer = {}
        return obs

    def set_export_path(self, path):
        self.export_path = path

    def prune_kernel(self, op_idx, preserve_ratio, preserve_idx=None,last_mask=None):
        '''Return the real ratio'''
        m_list = list(self.model.modules())
        op = m_list[op_idx]
        assert (preserve_ratio <= 1.)

        def format_rank(x):
            rank = int(np.around(x))
            return max(rank, 1)

        n, c = op.weight.size(0), op.weight.size(1)
        d_prime = format_rank(c * preserve_ratio)
        d_prime = int(np.ceil(d_prime * 1. / self.channel_round) * self.channel_round)
        if d_prime > c:
            d_prime = int(np.floor(c * 1. / self.channel_round) * self.channel_round)

        weight = op.weight.data.cpu().numpy()
        # conv [C_out, C_in, ksize, ksize]
        # fc [C_out, C_in]
        op_type = 'Conv2D'
        if len(weight.shape) == 2:
            op_type = 'Linear'
            weight = weight[:, :, None, None]

        if preserve_idx is None:  # not provided, generate new
            importance = np.abs(weight).sum((0, 2, 3))
            sorted_idx = np.argsort(-importance)  # sum magnitude along C_in, sort descend
            preserve_idx = sorted_idx[:d_prime]  # to preserve index  
       
                                   
        assert len(preserve_idx) == d_prime
        mask = np.zeros(weight.shape[1], bool)
        mask[preserve_idx] = True                       
        
        if preserve_ratio == 1:  # do not prune
            mask[:] = True
            prev_idx = self.prunable_idx[self.prunable_idx.index(op_idx)-1]
            for idx in range(prev_idx, op_idx):
                m = m_list[idx]
                if type(m) == nn.Conv2d:
                    m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[:, last_mask, :, :]).cuda()
                elif type(m)==nn.Linear:
                    m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[:, last_mask]).cuda()
            if op_idx == self.prunable_idx[-1]:
                m = m_list[op_idx]
                if type(m) == nn.Linear:
                    m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[:, last_mask]).cuda()
            return 1., op.weight.size(1), preserve_idx, mask # TODO: should be a full index

        action = np.sum(mask) * 1. / len(mask)  # calculate the ratio


        prev_idx = self.prunable_idx[self.prunable_idx.index(op_idx)-1]
        for idx in range(prev_idx, op_idx):
            m = m_list[idx]
            if type(m) == nn.Conv2d:
                m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[:, last_mask, :, :]).cuda()
                m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[mask, :, :, :]).cuda()
                # m.bias.data = torch.from_numpy(m.bias.data.cpu().numpy()[mask]).cuda()
                if m.groups == m.in_channels:
                    m.groups = int(np.sum(mask))
            elif type(m) == nn.BatchNorm2d:
                # prGreen(m)
                m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[mask]).cuda()
                m.bias.data = torch.from_numpy(m.bias.data.cpu().numpy()[mask]).cuda()
                m.running_mean.data = torch.from_numpy(m.running_mean.data.cpu().numpy()[mask]).cuda()
                m.running_var.data = torch.from_numpy(m.running_var.data.cpu().numpy()[mask]).cuda()
            elif type(m)==nn.Linear:
                m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[:, last_mask]).cuda()
                m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[mask, :]).cuda()
            elif type(m)==LambdaLayer:
                m.planes = int(m.planes*action)

        if op_idx == self.prunable_idx[-1]:
            m = m_list[op_idx]
            if type(m) == nn.Linear:
                m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[:,mask]).cuda()    

        return action, d_prime, preserve_idx, mask


    def prune_kernel_mobilenet(self, op_idx, preserve_ratio, preserve_idx=None, last_mask=None):
        '''Return the real ratio'''
        m_list = list(self.model.modules())
        op = m_list[op_idx]
        assert (preserve_ratio <= 1.)


        def format_rank(x):
            rank = int(np.around(x))
            return max(rank, 1)

        n, c = op.weight.size(0), op.weight.size(1)
        d_prime = format_rank(c * preserve_ratio)
        d_prime = int(np.ceil(d_prime * 1. / self.channel_round) * self.channel_round)
        if d_prime > c:
            d_prime = int(np.floor(c * 1. / self.channel_round) * self.channel_round)

        if self.use_new_input:  # this is slow and may lead to overfitting
            self._regenerate_input_feature()
        X = self.layer_info_dict[op_idx]['input_feat']  # input after pruning of previous ops
        Y = self.layer_info_dict[op_idx]['output_feat']  # fixed output from original model
        weight = op.weight.data.cpu().numpy()
        # conv [C_out, C_in, ksize, ksize]
        # fc [C_out, C_in]
        op_type = 'Conv2D'
        if len(weight.shape) == 2:
            op_type = 'Linear'
            weight = weight[:, :, None, None]


        if preserve_idx is None:  # not provided, generate new
            importance = np.abs(weight).sum((0, 2, 3))
            sorted_idx = np.argsort(-importance)  # sum magnitude along C_in, sort descend
            preserve_idx = sorted_idx[:d_prime]  # to preserve index       

        assert len(preserve_idx) == d_prime
        mask = np.zeros(weight.shape[1], bool)
        mask[preserve_idx] = True                           
        
        if preserve_ratio == 1:  # do not prune
            mask[:] = True
            return 1., op.weight.size(1), preserve_idx, mask # TODO: should be a full index
            # n, c, h, w = op.weight.size()
            # mask = np.ones([c], dtype=bool)

        # reconstruct, X, Y <= [N, C]
        if weight.shape[2] == 1:  # 1x1 conv or fc
            masked_X = X[:, mask]
            from lib.utils import least_square_sklearn
            rec_weight = least_square_sklearn(X=masked_X, Y=Y)
            rec_weight = rec_weight.reshape(-1, 1, 1, d_prime)  # (C_out, K_h, K_w, C_in')
            rec_weight = np.transpose(rec_weight, (0, 3, 1, 2))  # (C_out, C_in', K_h, K_w)
        else: 
            pass

        if op_type == 'Linear':
            rec_weight = rec_weight.squeeze()
            assert len(rec_weight.shape) == 2
        # # now assign
        op.weight.data = torch.from_numpy(rec_weight).cuda()
        action = np.sum(mask) * 1. / len(mask)  # calculate the ratio

        prev_idx = self.prunable_idx[self.prunable_idx.index(op_idx)-1]
        for idx in range(prev_idx, op_idx):
            m = m_list[idx]
            if type(m) == nn.Conv2d:  # depthwise
                m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[mask, :, :, :]).cuda()
                if m.groups == m.in_channels:
                    m.groups = int(np.sum(mask))
            elif type(m) == nn.BatchNorm2d:
                m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[mask]).cuda()
                m.bias.data = torch.from_numpy(m.bias.data.cpu().numpy()[mask]).cuda()
                m.running_mean.data = torch.from_numpy(m.running_mean.data.cpu().numpy()[mask]).cuda()
                m.running_var.data = torch.from_numpy(m.running_var.data.cpu().numpy()[mask]).cuda()
            elif type(m) == nn.Linear:
                m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[mask, :]).cuda()
                m.bias.data = torch.from_numpy(m.bias.data.cpu().numpy()[mask]).cuda()

        return action, d_prime, preserve_idx, mask

    def _is_final_layer(self):
        return self.cur_ind == len(self.prunable_idx) - 1

    def _action_wall(self, action):
        assert len(self.strategy) == self.cur_ind

        action = float(action)
        action = np.clip(action, 0, 1)

        other_comp = 0
        this_comp = 0
        for i, idx in enumerate(self.prunable_idx):
            flop = self.layer_info_dict[idx]['flops']
            buffer_flop = self._get_buffer_flops(idx)

            if i == self.cur_ind - 1:  # TODO: add other member in the set
                this_comp += flop * self.strategy_dict[idx][0]
                # add buffer (but not influenced by ratio)
                other_comp += buffer_flop * self.strategy_dict[idx][0]
            elif i == self.cur_ind:
                this_comp += flop * self.strategy_dict[idx][1]
                # also add buffer here (influenced by ratio)
                this_comp += buffer_flop
            else:
                other_comp += flop * self.strategy_dict[idx][0] * self.strategy_dict[idx][1]
                # add buffer
                other_comp += buffer_flop * self.strategy_dict[idx][0]  # only consider input reduction

        self.expected_min_preserve = other_comp + this_comp * action
        max_preserve_ratio = (self.expected_preserve_computation - other_comp) * 1. / this_comp

        action = np.minimum(action, max_preserve_ratio)
        action = np.maximum(action, self.strategy_dict[self.prunable_idx[self.cur_ind]][0])  # impossible (should be)

        return action

    def _action_wall_para(self, action):
        assert len(self.strategy) == self.cur_ind

        action = float(action)
        action = np.clip(action, 0, 1)

        other_comp = 0
        this_comp = 0
        for i, idx in enumerate(self.prunable_idx):
            para = self.layer_info_dict[idx]['params']
            buffer_para = self._get_buffer_para(idx)

            if i == self.cur_ind - 1:  # TODO: add other member in the set
                this_comp += para * self.strategy_dict[idx][0]
                # add buffer (but not influenced by ratio)
                other_comp += buffer_para * self.strategy_dict[idx][0]
            elif i == self.cur_ind:
                this_comp += para * self.strategy_dict[idx][1]
                # also add buffer here (influenced by ratio)
                this_comp += buffer_para
            else:
                other_comp += para * self.strategy_dict[idx][0] * self.strategy_dict[idx][1]
                # add buffer
                other_comp += buffer_para * self.strategy_dict[idx][0]  # only consider input reduction

        self.expected_min_preserve = other_comp + this_comp * action
        max_preserve_ratio = (self.expected_preserve_computation - other_comp) * 1. / this_comp

        action = np.minimum(action, max_preserve_ratio)
        action = np.maximum(action, self.strategy_dict[self.prunable_idx[self.cur_ind]][0])  # impossible (should be)

        return action

    def _get_buffer_flops(self, idx):
        buffer_idx = self.buffer_dict[idx]
        buffer_flop = sum([self.layer_info_dict[_]['flops'] for _ in buffer_idx])
        return buffer_flop

    def _get_buffer_para(self, idx):
        buffer_idx = self.buffer_dict[idx]
        buffer_para = sum([self.layer_info_dict[_]['params'] for _ in buffer_idx])
        return buffer_para

    def _cur_flops(self):
        flops = 0
        for i, idx in enumerate(self.prunable_idx):
            c, n = self.strategy_dict[idx]  # input, output pruning ratio
            flops += self.layer_info_dict[idx]['flops'] * c * n
            # add buffer computation
            # flops += self._get_buffer_flops(idx) * c  # only related to input channel reduction
        return flops

    def _cur_para(self):
        para = 0
        for i, idx in enumerate(self.prunable_idx):
            c, n = self.strategy_dict[idx]  # input, output pruning ratio
            para += self.layer_info_dict[idx]['params'] * c * n
            # add buffer computation
            para += self._get_buffer_para(idx) * c  # only related to input channel reduction
        return para

    def _cur_para(self):
        para = 0
        for i, idx in enumerate(self.prunable_idx):
            c, n = self.strategy_dict[idx]  # input, output pruning ratio
            para += self.layer_info_dict[idx]['params'] * c * n
            # add buffer computation
            # flops += self._get_buffer_flops(idx) * c  # only related to input channel reduction
        return para

    def _cur_reduced(self):
        # return the reduced weight
        reduced = self.org_flops - self._cur_flops()
        return reduced

    def _cur_reduced_para(self):
        # return the reduced weight
        reduced = self.org_para - self._cur_para()
        return reduced

    def _init_data(self):
        # split the train set into train + val
        # for CIFAR, split 5k for val
        # for ImageNet, split 3k for val
        val_size = 5000 if 'cifar' in self.data_type else 3000
        self.train_loader, self.val_loader, n_class = get_split_dataset(self.data_type, self.batch_size,
                                                                        self.n_data_worker, val_size,
                                                                        data_root=self.data_root,
                                                                        use_real_val=self.use_real_val,
                                                                        shuffle=False)  # same sampling
        if self.use_real_val:  # use the real val set for eval, which is actually wrong
            print('*** USE REAL VALIDATION SET!')

    def _build_index(self):
        self.prunable_idx = []
        self.prunable_ops = []
        self.shared_idx = []
        self.shared_prunable_ops_index = []
        self.connected_idx = []
        self.layer_type_dict = {}
        self.strategy_dict = {}
        self.buffer_dict = {}
        this_buffer_list = []
        self.org_channels = []
        # build index and the min strategy dict
        for i, m in enumerate(self.model.modules()):
            # print("({}):{}".format(i,m))
            if type(m) in self.prunable_layer_types:
                if type(m) == nn.Conv2d and m.groups == m.in_channels:  # depth-wise conv, buffer
                    this_buffer_list.append(i)
                else:  # really prunable
                    self.prunable_idx.append(i)
                    self.prunable_ops.append(m)
                    self.layer_type_dict[i] = type(m)
                    self.buffer_dict[i] = this_buffer_list
                    this_buffer_list = []  # empty
                    self.org_channels.append(m.in_channels if type(m) == nn.Conv2d else m.in_features)

                    self.strategy_dict[i] = [self.lbound, self.lbound]
            elif type(m) == InvertedResidual:
                if m.use_res_connect:
                    # env.connected_idx.append(i+2)
                    # the "2" stand for skip the block(InvertedResidual),conv(Sequential) modules,\
                    # which is vary from your net architecture definition.
                    # env.connected_idx.append(i+len(list(m.modules()))+2)  # the "2" same as above comment.
                    for j, _m in enumerate(m.modules()):
                        if type(_m) in self.prunable_layer_types:
                            break
                    self.connected_idx.append(i+j)
                    self.connected_idx.append(i+len(list(m.modules()))+j)


        self.strategy_dict[self.prunable_idx[0]][0] = 1  # modify the input
        self.strategy_dict[self.prunable_idx[-1]][1] = 1  # modify the output
    
        self.connected_idx = sorted(list(set(self.connected_idx)))

        if self.args.model == 'resnet':
            self.connected_idx = [5, 11, 17, 23, 29, 35, 41, 47, 53, 60, 66, 72, 78, 84, 90, 96, 102, 108, 115, 121, 127, 133, 139, 145, 151, 157, 163, 165, 168]
        
        m_list = list(self.model.modules())
        if self.args.model == 'mobilenetv2' or self.args.model == 'resnet':  # TODO: to be tested! Share index for residual connection
            # self.shared_idx = [[1, 4], [11, 14], [24, 27], [43, 46]]
            # self.shared_idx = [[60, 66], [115, 121]]
            for c_idx in self.connected_idx:
                op_idx = self.prunable_idx.index(c_idx)
                self.shared_prunable_ops_index.append(op_idx)
            last_ch = -1
            share_group = None
            for c_idx in self.shared_prunable_ops_index:
                
                if type(m_list[self.prunable_idx[c_idx]])==nn.Linear:
                    input_ = self.prunable_ops[c_idx].in_features
                else:
                    input_ = self.prunable_ops[c_idx].in_channels

                if input_!= last_ch:  # new group
                    last_ch = self.prunable_ops[c_idx].in_channels
                    if share_group is not None:
                        self.shared_idx.append(share_group)
                    share_group = [c_idx]
                else:  # same group
                    share_group.append(c_idx)
            if share_group is not None:
                self.shared_idx.append(share_group)

            print('=> Conv layers to share channels: {}'.format(self.shared_idx))
            print("==>prunable_idx:{}".format(self.prunable_idx))
            print("==>connected_idx:{}".format(self.connected_idx))
            print("==>shared_prunable_ops_index:{}".format(self.shared_prunable_ops_index))

        self.min_strategy_dict = copy.deepcopy(self.strategy_dict)

        self.buffer_idx = []
        for k, v in self.buffer_dict.items():
            self.buffer_idx += v

        print('=> Prunable layer idx: {}'.format(self.prunable_idx))
        print('=> Buffer layer idx: {}'.format(self.buffer_idx))
        print('=> Initial min strategy dict: {}'.format(self.min_strategy_dict))

        # added for supporting residual connections during pruning
        self.visited = [False] * len(self.prunable_idx)
        self.index_buffer = {}

    def _extract_layer_information(self):
        m_list = list(self.model.modules())

        self.data_saver = []
        self.layer_info_dict = dict()
        self.wsize_list = []
        self.flops_list = []

        from lib.utils import measure_layer_for_pruning

        # extend the forward fn to record layer info
        def new_forward(m):
            def lambda_forward(x):
                m.input_feat = x.clone()
                measure_layer_for_pruning(m, x)
                y = m.old_forward(x)
                m.output_feat = y.clone()
                return y

            return lambda_forward

        def im2col(x, conv):
            x_unfold = f.unfold(x, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding)
            return x_unfold

        for idx in self.prunable_idx + self.buffer_idx:  # get all
            m = m_list[idx]
            m.old_forward = m.forward
            m.forward = new_forward(m)

        # now let the image flowm.weight.data
        print('=> Extracting information...')
        torch.cuda.empty_cache()
        with torch.no_grad():
            for i_b, (input, target) in enumerate(self.train_loader):  # use image from train set
                if i_b == self.n_calibration_batches:
                    break
                self.data_saver.append((input.clone(), target.clone()))
                input_var = torch.autograd.Variable(input).cuda()

                # inference and collect stats
                _ = self.model(input_var)

                if i_b == 0:  # first batch
                    for idx in self.prunable_idx + self.buffer_idx:
                        self.layer_info_dict[idx] = dict()
                        self.layer_info_dict[idx]['params'] = m_list[idx].params
                        self.layer_info_dict[idx]['flops'] = m_list[idx].flops
                        # print('({}):{}'.format(idx,m_list[idx].params))
                        self.wsize_list.append(m_list[idx].params)
                        self.flops_list.append(m_list[idx].flops)
                if self.args.model == 'mobilenet' or self.args.model == 'mobilenetv2':
                    for idx in self.prunable_idx:
                        f_in_np = m_list[idx].input_feat.data.cpu().numpy()
                        f_out_np = m_list[idx].output_feat.data.cpu().numpy()
                        if len(f_in_np.shape) == 4:  # conv
                            if self.prunable_idx.index(idx) == 0:  # first conv
                                f_in2save, f_out2save = None, None
                            # elif m_list[idx].weight.size(3) > 1:  # normal conv
                            #     # f_in2save, f_out2save = f_in_np, f_out_np
                            #     randx = np.random.randint(0, f_out_np.shape[2] - 0, self.n_points_per_layer)
                            #     randy = np.random.randint(0, f_out_np.shape[3] - 0, self.n_points_per_layer) 
                            #     # input: [N, C, H, W]
                            #     self.layer_info_dict[idx][(i_b, 'randx')] = randx.copy()
                            #     self.layer_info_dict[idx][(i_b, 'randy')] = randy.copy()
                                
                            #     #im2col
                            #     f_in_np = im2col(m_list[idx].input_feat.data.detach().cpu().clone(),m_list[idx])
                            #     f_in_np = f_in_np.numpy()
                            #     w_out = f_out_np.shape[2]
                            #     pts = randx * w_out + randy
                            #     self.layer_info_dict[idx][(i_b, 'pts')] = pts.copy()

                            #     f_in2save = f_in_np[:, :, pts].copy().transpose(0, 2, 1)\
                            #         .reshape(self.batch_size * self.n_points_per_layer, -1)

                            #     f_out2save = f_out_np[:, :, randx, randy].copy().transpose(0, 2, 1) \
                            #         .reshape(self.batch_size * self.n_points_per_layer, -1)

                            #     # f_in2save, f_out2save = f_in_np, f_out_npm.weight.data
                            elif m_list[idx].weight.size(3)==1:  # 1x1 conv
                                # assert f_out_np.shape[2] == f_in_np.shape[2]  # now support k=3
                                randx = np.random.randint(0, f_out_np.shape[2] - 0, self.n_points_per_layer)
                                randy = np.random.randint(0, f_out_np.shape[3] - 0, self.n_points_per_layer)
                                # input: [N, C, H, W]
                                self.layer_info_dict[idx][(i_b, 'randx')] = randx.copy()
                                self.layer_info_dict[idx][(i_b, 'randy')] = randy.copy()

                                f_in2save = f_in_np[:, :, randx, randy].copy().transpose(0, 2, 1)\
                                    .reshape(self.batch_size * self.n_points_per_layer, -1)

                                f_out2save = f_out_np[:, :, randx, randy].copy().transpose(0, 2, 1) \
                                    .reshape(self.batch_size * self.n_points_per_layer, -1)
                        else:
                            assert len(f_in_np.shape) == 2
                            f_in2save = f_in_np.copy()
                            f_out2save = f_out_np.copy()
                        if 'input_feat' not in self.layer_info_dict[idx]:
                            self.layer_info_dict[idx]['input_feat'] = f_in2save
                            self.layer_info_dict[idx]['output_feat'] = f_out2save
                        else:
                            self.layer_info_dict[idx]['input_feat'] = np.vstack(
                                (self.layer_info_dict[idx]['input_feat'], f_in2save))
                            self.layer_info_dict[idx]['output_feat'] = np.vstack(
                                (self.layer_info_dict[idx]['output_feat'], f_out2save))

    def _regenerate_input_feature(self):
        # only re-generate the input feature
        m_list = list(self.model.modules())

        # delete old features
        for k, v in self.layer_info_dict.items():
            if 'input_feat' in v:
                v.pop('input_feat')

        # now let the image flow
        print('=> Regenerate features...')

        with torch.no_grad():
            for i_b, (input, target) in enumerate(self.data_saver):
                input_var = torch.autograd.Variable(input).cuda()

                # inference and collect stats
                _ = self.model(input_var)

                for idx in self.prunable_idx:
                    f_in_np = m_list[idx].input_feat.data.cpu().numpy()
                    if len(f_in_np.shape) == 4:  # conv
                        if self.prunable_idx.index(idx) == 0:  # first conv
                            f_in2save = None
                        # elif m_list[idx].weight.size(3) > 1:
                        #     pts = self.layer_info_dict[idx][(i_b, 'pts')]
                        #     f_in2save = f_in_np[:, :, pts].copy().transpose(0, 2, 1)\
                        #         .reshape(self.batch_size * self.n_points_per_layer, -1)
                        elif m_list[idx].weight.size(3)==1:
                            randx = self.layer_info_dict[idx][(i_b, 'randx')]
                            randy = self.layer_info_dict[idx][(i_b, 'randy')]
                            f_in2save = f_in_np[:, :, randx, randy].copy().transpose(0, 2, 1)\
                                .reshape(self.batch_size * self.n_points_per_layer, -1)
                    else:  # fc
                        assert len(f_in_np.shape) == 2
                        f_in2save = f_in_np.copy()
                    if 'input_feat' not in self.layer_info_dict[idx]:
                        self.layer_info_dict[idx]['input_feat'] = f_in2save
                    else:
                        self.layer_info_dict[idx]['input_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['input_feat'], f_in2save))

    def _build_state_embedding(self):
        # build the static part of the state embedding
        layer_embedding = []
        module_list = list(self.model.modules())
        for i, ind in enumerate(self.prunable_idx):
            m = module_list[ind]
            this_state = []
            if type(m) == nn.Conv2d:
                # this_state.append(i)  # index
                this_state.append(0)  # layer type, 0 for conv
                this_state.append(m.in_channels)  # in channels
                this_state.append(m.out_channels)  # out channels
                this_state.append(m.stride[0])  # stride
                this_state.append(m.kernel_size[0])  # kernel size
                this_state.append(np.prod(m.weight.size()))  # weight size
                this_state.append(self.flops_list[i])
                this_state.append(self.wsize_list[i])
            elif type(m) == nn.Linear:
                # this_state.append(i)  # index
                this_state.append(1)  # layer type, 1 for fc
                this_state.append(m.in_features)  # in channels
                this_state.append(m.out_features)  # out channels
                this_state.append(0)  # stride
                this_state.append(1)  # kernel size
                this_state.append(np.prod(m.weight.size()))  # weight size
                this_state.append(self.flops_list[i])
                this_state.append(self.wsize_list[i])
            
            # this 3 features need to be changed later
            # this_state.append(0.)  # reducedand 
            # this_state.append(0.)  # rest
            # this_state.append(1.)  # a_{t-1}
            layer_embedding.append(np.array(this_state))
            
        # normalize the state
        layer_embedding = np.array(layer_embedding, dtype=np.float32)
        print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        assert len(layer_embedding.shape) == 2, layer_embedding.shape
        # for i in range(layer_embedding.shape[1]):
        #     fmin = min(layer_embedding[:, i])
        #     fmax = max(layer_embedding[:, i])
        #     if fmax - fmin > 0:
        #         layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

        self.layer_embedding = layer_embedding

    def _validate(self, val_loader, model, verbose=False):
        '''
        Validate the performance on validation set
        :param val_loader:
        :param model:
        :param verbose:
        :return:m.weight.data
        '''
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        criterion = nn.CrossEntropyLoss().cuda()
        # switch to evaluate mode
        model.eval()
        end = time.time()
        # gpu_tracker = MemTracker()

        i = 0
        total_time = 0
        # gpu_tracker.track()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)
                input_var = torch.autograd.Variable(input).cuda()
                target_var = torch.autograd.Variable(target).cuda()

                if i is 0:
                    memoryUse=1#modelsize(model,input.cuda())

                # compute output
                t1 = time.time()
                output = model(input_var)
                t2 = time.time()

                loss = criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                total_time += (t2-t1)

        timeUse = 1000*total_time/(i+1)
        if verbose:
            print('* Test loss: %.3f    top1: %.3f    top5: %.3f    time: %.5f      memory: %.3f' %
                  (losses.avg, top1.avg, top5.avg, timeUse, memoryUse))
        if self.acc_metric == 'acc1':
            return top1.avg,timeUse, memoryUse
        elif self.acc_metric == 'acc5':
            return top5.avg,timeUse, memoryUse
        else:
            raise NotImplementedError
