import time
import torch
import torch.nn as nn
from torch.nn import functional as f
from lib.utils import AverageMeter, accuracy, prGreen
from lib.data import get_split_dataset
from scipy.spatial import distance
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
        # assert self.preserve_ratio > self.lbound, 'Error! You can make achieve preserve_ratio smaller than lbound!'

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
        # self._build_state_embedding()

        # build reward
        self.reset()  # restore weight
        self.org_acc, self.org_time, self.org_memory = self._validate(self.val_loader, self.model)
        print('=> original acc: {:.3f}%'.format(self.org_acc))
        print('=> original time: {:.4f}ms'.format(self.org_time))
        print('=> original memory: {:.4f}MB'.format(self.org_memory))
        self.org_para = sum(self.wsize_list)
        print('=> original weight size: {:.4f} M param'.format(self.org_para * 1. / 1e6))
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
        elif self.model_type =='resnet56':
            if self.data_type == 'cifar10':
                from models.resnet import resnet56
                self.model = resnet56()
            elif self.data_type == 'cifar100':
                from models.resnet import resnet56
                self.model = resnet56()
            else:
                raise NotImplementedError
        elif self.model_type == 'mobilenetv2':
            if self.data_type == 'imagenet':
                from models.mobilenet_v2 import MobileNetV2
                self.model = MobileNetV2(n_class=1000, input_size=224)
            elif self.data_type == 'cifar10':
                from models.mobilenet_v2 import MobileNetV2
                self.model = MobileNetV2(n_class=10, input_size=32)
            elif self.data_type == 'cifar100':
                from models.mobilenet_v2 import MobileNetV2
                self.model = MobileNetV2(n_class=100, input_size=32)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        self.model.load_state_dict(self.checkpoint)
        self.model = self.model.cuda()


    def step(self, action):
        # Pseudo prune and get the corresponding statistics. The real pruning happens till the end of all pseudo pruning
        if self.visited[self.cur_ind]:
            action = self.strategy_dict[self.prunable_idx[self.cur_ind]][1]
            preserve_idx = self.index_buffer[self.cur_ind]
        else:
            if self.args.prune == 'para':
                action = self._action_wall_para(action)
            elif self.args.prune == 'flops':
                action = self._action_wall(action)  # percentage to preserve
            else:
                raise NotImplementedError
            preserve_idx = None

        action, d_prime, preserve_idx, mask = self.prune_kernel(self.prunable_idx[self.cur_ind], action, preserve_idx,last_mask=self.mask)

        self.mask = mask

        if not self.visited[self.cur_ind]:
            for group in self.shared_idx:
                if self.cur_ind in group:  # set the shared ones
                    for g_idx in group:
                        self.strategy_dict[self.prunable_idx[g_idx]][1] = action
                        self.strategy_dict[self.prunable_idx[g_idx + 1]][0] = action
                        self.visited[g_idx] = True
                        self.index_buffer[g_idx] = preserve_idx.copy()

        if self.export_model:  # export checkpoint
            print('# Pruning {}: ratio: {}, d_prime: {}'.format(self.cur_ind, action, d_prime))

        self.strategy.append(action)  # save action to strategy
        self.d_prime_list.append(d_prime)

        self.strategy_dict[self.prunable_idx[self.cur_ind]][1] = action
        if self.cur_ind > 0 and self.cur_ind+1 < len(self.prunable_idx):
            self.strategy_dict[self.prunable_idx[self.cur_ind + 1]][0] = action

        # all the actions are made
        if self._is_final_layer():

            assert len(self.strategy) == len(self.prunable_idx)
            current_flops = self._cur_flops()
            current_para = self._cur_para()
            compress_ratio = current_flops * 1. / self.org_flops
            para_ratio = current_para * 1. / self.org_para

            # if self.export_model:  # export state dict
            #     prGreen('compress: {:.4f}, para: {:.4f}'.format(compress_ratio, para_ratio))
            #     return None, None, None, None

            acc, val_time, val_memory = self._validate(self.val_loader, self.model)
            compress_ratio = current_flops * 1. / self.org_flops
            info_set = {'compress_ratio': compress_ratio, 'para_ratio': para_ratio, 'accuracy': acc, 'strategy': self.strategy.copy(), 'time': val_time,  'memory': val_memory}
            reward = self.reward(self, acc / 100, compress_ratio, val_time / self.org_time,
                                 val_memory / self.org_memory)


            if reward > self.best_reward:
                self.best_reward = reward
                self.best_strategy = self.strategy.copy()
                self.best_d_prime_list = self.d_prime_list.copy()
                prGreen(
                    'New best reward: {:.4f}, acc: {:.4f}, compress: {:.4f}, para: {:.4f}'.format(self.best_reward, acc,
                                                                                                  compress_ratio,
                                                                                                  para_ratio))
                prGreen('New best policy: {}'.format(self.best_strategy))
                prGreen('New best d primes: {}'.format(self.best_d_prime_list))
                torch.save(self.model.state_dict(), self.export_path)

            obs = self.layer_embedding[self.cur_ind, :].copy()  # actually the same as the last state
            done = True

            self._get_model()

            return obs, reward, done, info_set

        info_set = None
        reward = 0
        done = False
        self.visited[self.cur_ind] = True  # set to visited
        self.cur_ind += 1  # the index of next layer
        # build next state (in-place modify)
        if self.args.prune == 'para':
            self.layer_embedding[self.cur_ind][-1] = self.strategy[-1]  # last action
            self.layer_embedding[self.cur_ind][-2] = sum(
                self.wsize_list[self.cur_ind + 1:]) * 1. / self.org_para  # rest
            self.layer_embedding[self.cur_ind][-3] = self._cur_reduced_para() * 1. / self.org_para  # reduced
        elif self.args.prune == 'flops':
            self.layer_embedding[self.cur_ind][-1] = self.strategy[-1]  # last action
            self.layer_embedding[self.cur_ind][-2] = sum(
                self.flops_list[self.cur_ind + 1:]) * 1. / self.org_flops  # rest
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
        self.layer_embedding[:, -1] = 1.
        self.layer_embedding[:, -2] = 0.
        self.layer_embedding[:, -3] = 0.
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

    def prune_kernel(self, op_idx, preserve_ratio, lambdas, preserve_idx=None, last_mask=None):
        '''Return the real ratio'''
        m_list = list(self.model.modules())
        op = m_list[op_idx]
        assert (preserve_ratio <= 1.)

        if op_idx == self.prunable_idx[-1]:
            return 1, 1, None, None

        def format_rank(x):
            rank = int(np.around(x))
            return max(rank, 1)

        n, c = op.weight.size(0), op.weight.size(1)
        d_prime = format_rank(n * preserve_ratio)
        d_prime = int(np.ceil(d_prime * 1. / self.channel_round) * self.channel_round)
        if d_prime > n:
            d_prime = int(np.floor(n * 1. / self.channel_round) * self.channel_round)

        weight = op.weight.data.cpu().numpy()
        # conv [C_out, C_in, ksize, ksize]
        # fc [C_out, C_in]
        op_type = 'Conv2D'
        if len(weight.shape) == 2:
            op_type = 'Linear'
            weight = weight[:, :, None, None]

        if preserve_idx is None:
            weight_vec = op.weight.data.reshape(weight.shape[0], -1)
            # norm = torch.norm(weight_vec, 2, 1)
            # norm_np = norm.cpu().detach().numpy()
            # arg_max = np.argsort(norm_np)
            # preserve_idx = arg_max[::-1][:d_prime]

            #L2-GM
            weight_vec = weight_vec.cpu().detach().numpy()
            matrix = distance.cdist(weight_vec, weight_vec, 'euclidean')
            similar_sum = np.sum(np.abs(matrix), axis=0)
            preserve_idx = np.argpartition(similar_sum, -d_prime)[-d_prime:]


        assert len(preserve_idx) == d_prime
        mask = np.zeros(weight.shape[0], bool)
        mask[preserve_idx] = True
        action = np.sum(mask) * 1. / len(mask)  # calculate the ratio

        if preserve_ratio == 1:  # do not prune
            mask[:] = True
            return 1., op.weight.size(0), preserve_idx, mask  # TODO: should be a full index

        # prune output channel
        if op_type == 'Conv2D':
            op.weight.data = torch.from_numpy(op.weight.data.cpu().numpy()[mask, :, :, :]).cuda()
            if op.bias is not None:
                op.bias.data = torch.from_numpy(op.bias.data.cpu().numpy()[mask]).cuda()
        elif op_type == 'Linear':
            op.weight.data = torch.from_numpy(op.weight.data.cpu().numpy()[mask, :]).cuda()
            if op.bias is not None:
                op.bias.data = torch.from_numpy(op.bias.data.cpu().numpy()[mask]).cuda()

        # if self.args.model == 'resnet56':
        #     if op_idx == 35 or op_idx == 71 or op_idx == 123:
        #         next_idx = self.prunable_idx[self.prunable_idx.index(op_idx)] + 1
        #     elif op_idx == 26 or op_idx == 62 or op_idx == 114:
        #         next_idx = self.prunable_idx[self.prunable_idx.index(op_idx) + 1]
        #         connect_idx = self.connected_idx[self.connected_idx.index(op_idx) + 2]
        #         m = m_list[connect_idx]
        #         m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[:, mask, :, :]).cuda()
        #     else:
        #         next_idx = self.prunable_idx[self.prunable_idx.index(op_idx) + 1]
        # else:
        #     next_idx = self.prunable_idx[self.prunable_idx.index(op_idx) + 1]
        next_idx = self.prunable_idx[self.prunable_idx.index(op_idx) + 1]
        for idx in range(op_idx, next_idx):
            m = m_list[idx + 1]
            if type(m) == nn.Conv2d:
                if m.groups == m.in_channels:  # depthwise
                    m.groups = int(np.sum(mask))
                    m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[mask, :, :, :]).cuda()
                else:
                    m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[:, mask, :, :]).cuda()

            elif type(m) == nn.BatchNorm2d or type(m) == nn.BatchNorm1d:
                m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[mask]).cuda()
                m.bias.data = torch.from_numpy(m.bias.data.cpu().numpy()[mask]).cuda()
                m.running_mean.data = torch.from_numpy(m.running_mean.data.cpu().numpy()[mask]).cuda()
                m.running_var.data = torch.from_numpy(m.running_var.data.cpu().numpy()[mask]).cuda()

            elif type(m) == nn.Linear:
                m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[:, mask]).cuda()

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

            if i == self.cur_ind + 1:  # TODO: add other member in the set
                this_comp += flop * self.strategy_dict[idx][1]
                # add buffer (but not influenced by ratio)
                other_comp += buffer_flop * self.strategy_dict[idx][1]
            elif i == self.cur_ind:
                this_comp += flop * self.strategy_dict[idx][0]
                # also add buffer here (influenced by ratio)
                this_comp += buffer_flop
            else:
                other_comp += flop * self.strategy_dict[idx][0] * self.strategy_dict[idx][1]
                # add buffer
                other_comp += buffer_flop * self.strategy_dict[idx][1]  # only consider input reduction

        self.expected_min_preserve = other_comp + this_comp * action
        max_preserve_ratio = (self.expected_preserve_computation - other_comp) * 1. / this_comp

        action = np.minimum(action, max_preserve_ratio)
        action = np.maximum(action, self.strategy_dict[self.prunable_idx[self.cur_ind]][1])  # impossible (should be)

        return action

    def _get_buffer_para(self, idx):
        buffer_idx = self.buffer_dict[idx]
        buffer_para = sum([self.layer_info_dict[_]['params'] for _ in buffer_idx])
        return buffer_para

    def _cur_reduced_para(self):
        # return the reduced weight
        reduced = self.org_para - self._cur_para()
        return reduced

    def _cur_para(self):
        para = 0
        for i, idx in enumerate(self.prunable_idx):
            c, n = self.strategy_dict[idx]  # input, output pruning ratio
            para += self.layer_info_dict[idx]['params'] * c * n
            # add buffer computation
            para += self._get_buffer_para(idx) * c  # only related to input channel reduction
        return para

    def _action_wall_para(self, action):
        assert len(self.strategy) == self.cur_ind

        action = float(action)
        action = np.clip(action, 0, 1)

        other_comp = 0
        this_comp = 0
        for i, idx in enumerate(self.prunable_idx):
            para = self.layer_info_dict[idx]['params']
            buffer_para = self._get_buffer_para(idx)
            if i == self.cur_ind + 1:  # TODO: add other member in the set
                this_comp += para * self.strategy_dict[idx][1]
                # add buffer (but not influenced by ratio)
                other_comp += buffer_para * self.strategy_dict[idx][1]
            elif i == self.cur_ind:
                this_comp += para * self.strategy_dict[idx][0]
                # also add buffer here (influenced by ratio)
                this_comp += buffer_para
            else:
                other_comp += para * self.strategy_dict[idx][0] * self.strategy_dict[idx][1]
                # add buffer
                other_comp += buffer_para * self.strategy_dict[idx][1]  # only consider output reduction

        self.expected_min_preserve = other_comp + this_comp * action
        max_preserve_ratio = (self.expected_preserve_computation - other_comp) * 1. / this_comp

        action = np.minimum(action, max_preserve_ratio)
        action = np.maximum(action, self.strategy_dict[self.prunable_idx[self.cur_ind]][1])  # impossible (should be)

        return action

    def _get_buffer_flops(self, idx):
        buffer_idx = self.buffer_dict[idx]
        buffer_flop = sum([self.layer_info_dict[_]['flops'] for _ in buffer_idx])
        return buffer_flop

    def _cur_flops(self):
        flops = 0
        for i, idx in enumerate(self.prunable_idx):
            c, n = self.strategy_dict[idx]  # input, output pruning ratio
            flops += self.layer_info_dict[idx]['flops'] * c * n
            # add buffer computation
            flops += self._get_buffer_flops(idx) * c  # only related to input channel reduction
        return flops


    def _cur_reduced(self):
        # return the reduced weight
        reduced = self.org_flops - self._cur_flops()
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
            # print(i, m)
            if type(m) in self.prunable_layer_types:
                if type(m) == nn.Conv2d and m.groups == m.in_channels:  # depth-wise conv, buffer
                    this_buffer_list.append(i)
                else:  # really prunable
                    self.prunable_idx.append(i)
                    self.prunable_ops.append(m)
                    self.layer_type_dict[i] = type(m)
                    self.buffer_dict[i] = this_buffer_list
                    this_buffer_list = []  # empty
                    self.org_channels.append(m.out_channels if type(m) == nn.Conv2d else m.out_features)
                    self.strategy_dict[i] = [self.lbound, self.lbound]
            elif type(m) == InvertedResidual:
                if m.use_res_connect:
                    for j, _m in enumerate(m.modules()):
                        if type(_m) in self.prunable_layer_types:
                            break
                    self.connected_idx.append(i - j)
                    self.connected_idx.append(i + len(list(m.modules())) - j)

        self.strategy_dict[self.prunable_idx[0]][0] = 1  # modify the input
        self.strategy_dict[self.prunable_idx[-1]][1] = 1  # modify the output

        self.connected_idx = sorted(list(set(self.connected_idx)))

        if self.args.model == 'resnet56':
            self.connected_idx = [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 62, 68, 74, 80, 86, 92, 98, 104, 110, 117, 123,
                                  129, 135, 141, 147, 153, 159, 165]

        m_list = list(self.model.modules())
        if self.args.model == 'mobilenetv2' or self.args.model == 'resnet56':  # TODO: to be tested! Share index for residual connection
            for c_idx in self.connected_idx:
                op_idx = self.prunable_idx.index(c_idx)
                self.shared_prunable_ops_index.append(op_idx)
            last_ch = -1
            share_group = None
            for c_idx in self.shared_prunable_ops_index:
                if type(m_list[self.prunable_idx[c_idx]]) == nn.Linear:
                    output_ = self.prunable_ops[c_idx].out_features
                else:
                    output_ = self.prunable_ops[c_idx].out_channels

                if output_ != last_ch:  # new group
                    last_ch = self.prunable_ops[c_idx].out_channels
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
                        self.wsize_list.append(m_list[idx].params)
                        self.flops_list.append(m_list[idx].flops)

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
            this_state.append(0.)  # reducedand
            this_state.append(0.)  # rest
            this_state.append(1.)  # a_{t-1}
            layer_embedding.append(np.array(this_state))
            
        # normalize the state
        layer_embedding = np.array(layer_embedding, dtype=np.float32)
        print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        assert len(layer_embedding.shape) == 2, layer_embedding.shape
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

        self.layer_embedding = layer_embedding

    def modelsize(self, model):

        m_list = list(model.modules())
        num = 0
        type_size = 4

        def calc_feat(s):
            return np.prod(np.array(s))

        def new_forward(m):
            def lambda_forward(x):
                m.input_feat = x.clone()
                # measure_layer_for_pruning(m, x)
                y = m.old_forward(x)
                m.output_feat = y.clone()
                return y

            return lambda_forward

        for idx, m in enumerate(m_list):
            # m = m_list[idx]
            if type(m) in [torch.nn.modules.linear.Linear, torch.nn.modules.conv.Conv2d,
                           torch.nn.ReLU, torch.nn.BatchNorm2d]:
                m.old_forward = m.forward
                m.forward = new_forward(m)

        # now let the image flow
        torch.cuda.empty_cache()
        with torch.no_grad():
            b, c, h, w = 50, 3, 32, 32
            input_var = torch.randn(b, c, h, w).cuda()
            _ = model(input_var)

        for idx, m in enumerate(m_list):
            # print(idx, m)
            # m = m_list[idx]
            if type(m) in [torch.nn.modules.linear.Linear, torch.nn.modules.conv.Conv2d,
                           torch.nn.ReLU, torch.nn.BatchNorm2d]:
                if type(m) == torch.nn.ReLU:
                    if m.inplace:
                        continue
                num += calc_feat(m.output_feat.shape)

        return (num) * type_size / 1024 / 1024

    def _validate(self, val_loader, model, verbose=False):
        '''
        Validate the performance on validation set
        :param val_loader:
        :param model:
        :param verbose:
        :return:
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
                    memoryUse = modelsize(model,input_var)
                    # print(memoryUse)
                    # print(modelsize_new(model,input.cuda(), count_idx=self.prunable_idx + self.buffer_idx))

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
                total_time += (t2 - t1)
        # gpu_tracker.track()
        timeUse = 1000 * total_time / (i + 1)
        if verbose:
            print('* Test loss: %.3f    top1: %.3f    top5: %.3f    time: %.5f      memory: %.3f' %
                  (losses.avg, top1.avg, top5.avg, timeUse, memoryUse))
        if self.acc_metric == 'acc1':
            return top1.avg, timeUse, memoryUse
        elif self.acc_metric == 'acc5':
            return top5.avg, timeUse, memoryUse
        else:
            raise NotImplementedError
