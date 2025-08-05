# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import os
import numpy as np
import argparse
from copy import deepcopy
import torch
torch.backends.cudnn.deterministic = True

from env.channel_pruning_env_llm import ChannelPruningEnv
from lib.sac import SAC
from lib.utils import get_output_folder
from tensorboardX import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='AMC search script')

    parser.add_argument('--job', default='train', type=str, help='support option: train/export')
    parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')
    # env
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext2",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="wikitext-2-raw-v1",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument('--preserve_ratio', default=0.5, type=float, help='preserve ratio of the model')
    parser.add_argument('--lbound', default=0.2, type=float, help='minimum preserve ratio')
    parser.add_argument('--rbound', default=1., type=float, help='maximum preserve ratio')
    parser.add_argument('--reward', default='acc_reward', type=str, help='Setting the reward')
    parser.add_argument('--acc_metric', default='acc5', type=str, help='use acc1 or acc5')
    parser.add_argument('--use_real_val', dest='use_real_val', action='store_true')
    parser.add_argument('--ckpt_path', default=None, type=str, help='manual path of checkpoint')
    parser.add_argument('--prune', default='flops', type=str, help='prune method: flops or params')
    parser.add_argument('--recon', dest='recon', action='store_true', help='use new input feature')

    # parser.add_argument('--n_calibration_batches', default=60, type=int,
    #                     help='n_calibration_batches')
    # parser.add_argument('--n_points_per_layer', default=20, type=int,
    #                     help='method to prune (fg/cp for fine-grained and channel pruning)')
    parser.add_argument('--n_samples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--recon_sample', type=int, default=16, help='Number of reconstruction samples.')
    parser.add_argument('--channel_round', default=8, type=int, help='Round channel to multiple of channel_round')
    parser.add_argument('--norm', default=1, type=int, help='use l1-norm or l2-norm')
    parser.add_argument('--threshold', default=0.25, type=float, help='threshold of cosine sim')
    parser.add_argument('--lambdas', default=0.7, type=float, help='threshold of cosine sim')

    # agent
    parser.add_argument('--hidden1', default=300, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--lr_c', default=1e-3, type=float, help='learning rate for actor')
    parser.add_argument('--lr_a', default=1e-4, type=float, help='learning rate for actor')
    parser.add_argument('--warmup', default=100, type=int,
                        help='time without training but only filling the replay memory')

    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=100, type=int, help='memory size for each layer')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--auto-alpha', type=int, default=1)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)

    # noise (truncated normal distribution)
    parser.add_argument('--init_delta', default=0.5, type=float,
                        help='initial variance of truncated normal distribution')
    parser.add_argument('--delta_decay', default=0.95, type=float,
                        help='delta decay during exploration')
    # training
    parser.add_argument('--max_episode_length', default=1e9, type=int, help='')
    parser.add_argument('--output', default='./logs', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_episode', default=800, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of gpu to use')
    parser.add_argument('--n_worker', default=12, type=int, help='number of data loader worker')
    parser.add_argument('--data_bsize', default=50, type=int, help='number of data batch size')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # export
    parser.add_argument('--ratios', default=None, type=str, help='ratios for pruning')
    parser.add_argument('--channels', default=None, type=str, help='channels after pruning')
    parser.add_argument('--export_path', default=None, type=str, help='path for exporting models')
    parser.add_argument('--agent_path', default=None, type=str, help='path for off-line loading agent')
    parser.add_argument('--use_new_input', dest='use_new_input', action='store_true', help='use new input feature')
    parser.add_argument('--state_mode', default=1, type=int, choices=[0, 1], 
                        help='Agent state mode: 0=global pruning ratio, 1=feature extraction state')

    return parser.parse_args()


def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    model.seqlen = 2048
    return model


def train(num_episode, agent, env, output):
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    summary=None
    T = []
    while episode < num_episode:  # counting based on episode
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())

        # agent pick action ...
        if episode <= args.warmup:
            action = agent.random_action(args.lbound, args.rbound)
        else:
            action = agent.select_action(observation, args.lbound, args.rbound, episode)

        # env response with next_observation, reward, terminate_info
        next_observation, reward, done, info = env.step(np.float32(action))

        if episode % int(num_episode / 3) == 0:
            agent.save(output)

        T.append([reward, deepcopy(observation), deepcopy(next_observation), action, done])

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(next_observation)


        if done:  # end of episode
            print('#{}: episode_reward:{:.4f} ppl: {:.4f} ratio: {:.4f}'.format(episode, episode_reward,
                                                                                    info['ppl'],
                                                                                    info['compress_ratio']))
            text_writer.write(
                '#{}: episode_reward:{:.4f} ppl: {:.4f} ratio: {:.4f}'.format(episode, episode_reward,
                                                                                    info['ppl'],
                                                                                    info['compress_ratio']))
            final_reward = T[-1][0]
            for r_t, s_t, s_t1, a_t, done in T:
                agent.replay_buffer.push(s_t, a_t, r_t, s_t1, done)
                if episode > args.warmup:
                    summary = agent.update()

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []

            if summary is not None:
                for k, v in summary.items():
                    tfwriter.add_scalar(k, v, episode)

            tfwriter.add_scalar('reward/last', reward, episode)
            tfwriter.add_scalar('reward/best', env.best_reward, episode)
            tfwriter.add_scalar('info/ppl', info['ppl'], episode)
            tfwriter.add_scalar('info/compress_ratio', info['compress_ratio'], episode)
            tfwriter.add_text('info/best_policy', str(env.best_strategy), episode)

            # record the preserve rate for each layer
            for i, preserve_rate in enumerate(env.strategy):
                tfwriter.add_scalar('preserve_rate/{}'.format(i), preserve_rate, episode)

    text_writer.close()


def export_model(env, args):
    assert (args.preserve_ratio is not None) or (args.ratios is not None) , 'Please provide a valid ratio'
    assert args.export_path is not None, 'Please provide a valid export path'
    env.set_export_path(args.export_path)

    print('=> Original model channels: {}'.format(env.dim_list))
    if args.ratios:
        ratios = args.ratios.split(',')
        ratios = [float(r) for r in ratios]
    elif args.channels:
        channels = args.channels.split(',')
        channels = [int(r) for r in channels]
        ratios = [c2 / c1 for c2, c1 in zip(channels, env.dim_list)]
    else:
        ratios = torch.ones(env.num_hidden_layers*2).cpu()
        ratios = ratios * args.preserve_ratio

    print('=> Pruning with ratios: {}'.format(ratios))

    for r in ratios:
        env.step(np.float32(r))

    return


if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # === 根据 state_mode 参数决定是否使用新输入特征 ===
    # 在创建环境之前设置，确保环境创建时使用正确的参数
    if args.state_mode == 1:
        args.use_new_input = True
        print("=> State Mode 1: 启用特征提取状态")
    else:
        args.use_new_input = False
        print("=> State Mode 0: 使用全局剪枝率状态")

    env = ChannelPruningEnv(args.model, args.dataset_name,
                            preserve_ratio=1. if args.job == 'export' else args.preserve_ratio,
                            n_data_worker=args.n_worker, batch_size=args.data_bsize,
                            args=args, export_model=args.job == 'export', use_new_input=args.use_new_input)
    
    if args.job == 'train':
        env.set_export_path(args.export_path)
        # build folder and logs
        base_folder_name = '{}_{}_r{}_search'.format(args.model_name, args.dataset_name, args.preserve_ratio)
        if args.suffix is not None:
            base_folder_name = base_folder_name + '_' + args.suffix
        args.output = get_output_folder(args.output, base_folder_name)
        print('=> Saving logs to {}'.format(args.output))
        tfwriter = SummaryWriter(logdir=args.output)
        text_writer = open(os.path.join(args.output, 'log.txt'), 'w')
        print('=> Output path: {}...'.format(args.output))

        nb_states = len(env.layer_embedding)
        nb_actions = 1

        args.rmsize = args.rmsize * env.num_hidden_layers * 2  # for each layer
        print('** Actual replay buffer size: {}'.format(args.rmsize))

        if args.auto_alpha:
            target_entropy = -np.prod(1)
            log_alpha = torch.zeros(1, requires_grad=True, device='cuda')
            alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
            alpha = (target_entropy, log_alpha, alpha_optim)
        else:
            alpha = args.alpha

        agent = SAC(nb_actions, nb_states, args.hidden1, args.hidden2, args.rmsize, args.lr_a, 1, args.bsize,
                    args.gamma, args.tau, alpha, args.init_delta, args.delta_decay, args.warmup)
        train(args.train_episode, agent, env, args.output)

    elif args.job == 'export':
        export_model(env, args)
    else:
        raise RuntimeError('Undefined job {}'.format(args.job))
