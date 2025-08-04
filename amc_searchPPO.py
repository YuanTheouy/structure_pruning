# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import os
import numpy as np
import argparse
from copy import deepcopy
import torch
from torch import nn
torch.backends.cudnn.deterministic = True

from env.channel_pruning_env_llm_global import ChannelPruningEnv
from env.weight_pruning_env_llm_global import WeightPruningEnv
from lib.ppo.ppo.ppo_lstm import MLP, PPO, Actor, Critic, Gaussian
from lib.utils import get_output_folder
from tensorboardX import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaTokenizer
from feature_extractor import FeatureExtractor
from lib.data import get_loaders


def parse_args():
    parser = argparse.ArgumentParser(description='AMC search script')

    parser.add_argument('--job', default='train', type=str, help='support option: train/export')
    parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')
    # env
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--resume_path', type=str,default=None, help='resmue model')
    parser.add_argument('--start', type=int, help='start to recon')
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
    parser.add_argument('--structure', dest='structure', action='store_true', help='structure prune or unstructure')
    parser.add_argument("--sparsity_type", type=str, help='N:M')
    parser.add_argument('--recon', dest='recon', action='store_true', help='reconstruction or not')

    # parser.add_argument('--n_calibration_batches', default=60, type=int,
    #                     help='n_calibration_batches')
    # parser.add_argument('--n_points_per_layer', default=20, type=int,
    #                     help='method to prune (fg/cp for fine-grained and channel pruning)')
    parser.add_argument('--m', type=int, default=5, help='h para.')
    parser.add_argument('--n_samples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--recon_sample', type=int, default=16, help='Number of reconstruction samples.')
    parser.add_argument('--channel_round', default=1, type=int, help='Round channel to multiple of channel_round')
    parser.add_argument('--norm', default=1, type=int, help='use l1-norm or l2-norm')
    parser.add_argument('--threshold', default=0.25, type=float, help='threshold of cosine sim')
    parser.add_argument('--lambdas', default=0.7, type=float, help='threshold of cosine sim')

    # agent
    parser.add_argument('--hidden1', default=256, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=256, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--lr_c', default=1e-3, type=float, help='learning rate for actor')
    parser.add_argument('--lr_a', default=1e-4, type=float, help='learning rate for actor')
    parser.add_argument('--warmup', default=100, type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=1., type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=100, type=int, help='memory size for each layer')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--num_collect', default=15, type=int, help='')
    
    # PPO specific parameters
    parser.add_argument('--clip_param', default=0.2, type=float, help='PPO clip parameter')
    parser.add_argument('--entropy_coef', default=0.01, type=float, help='PPO entropy coefficient')
    parser.add_argument('--value_loss_coef', default=0.5, type=float, help='PPO value loss coefficient')
    parser.add_argument('--gamma', default=0.998, type=float, help='PPO discount factor')
    parser.add_argument('--lamda', default=0.95, type=float, help='PPO GAE lambda')
    parser.add_argument('--max_grad_norm', default=0.5, type=float, help='PPO max gradient norm')
    
    # GPU parameter
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU device ID to use')
    
    # Downstream task evaluation
    parser.add_argument('--enable_downstream', default='false', type=str, 
                        help='Enable downstream task evaluation (true/false)')

    parser.add_argument('--learning_epoch', default=10, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
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
    parser.add_argument('--metric', default='input', type=str, help='input or wanda')
    parser.add_argument('--feat', default='mean', type=str, help='mean or flat')

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

    while episode < num_episode:  # counting based on episode
        for i in range(agent.num_collects):
            # reset if it is the start of episode
            with torch.inference_mode():
                if observation is None:
                    observation = deepcopy(env.reset())
                    # observation = np.expand_dims(observation, 0) # State is now a matrix
                
                # If using new features, observation is a single feature vector, need to add batch dimension
                if not args.use_new_input:
                    observation = np.expand_dims(observation, 0)
                else:
                    # For new input features, ensure we have the correct batch dimension
                    if observation.ndim == 1:
                        observation = np.expand_dims(observation, 0)

                action = agent.act(observation)
                # action = np.expand_dims(action)
                # env response with next_observation, reward, terminate_info
                next_observation, reward, done, info = env.step(np.squeeze(action))
                
                if not args.use_new_input:
                    next_observation = np.expand_dims(next_observation, 0)
                else:
                    # For new input features, ensure we have the correct batch dimension
                    if next_observation.ndim == 1:
                        next_observation = np.expand_dims(next_observation, 0)

                # [optional] save intermideate model
                if episode % int(num_episode / 100) == 0:
                    agent.save_model(output)

                # update
                step += 1
                episode_steps += 1
                episode_reward += reward
                timeout = np.array([0], dtype=bool)
                observation = deepcopy(next_observation)

                agent.step(next_observation, reward, done, timeout)


            if done:  # end of episode
                print('#{}: episode_reward:{:.4f} ppl: {:.4f} ratio: {:.4f}, para: {:.4f}'.format(episode, episode_reward,
                                                                                        info['ppl'],
                                                                                        info['compress_ratio'],
                                                                                        info['para_ratio']))
                text_writer.write(
                    '#{}: episode_reward:{:.4f} ppl: {:.4f} ratio: {:.4f}, para: {:.4f}'.format(episode, episode_reward,
                                                                                        info['ppl'],
                                                                                        info['compress_ratio'],
                                                                                        info['para_ratio']))
                if reward > env.best_reward:
                    agent.save_model(output)

                # reset
                observation = None
                episode_steps = 0
                episode_reward = 0.
                episode += 1

                if summary is not None:
                    for k, v in summary.items():
                        tfwriter.add_scalar(k, v, episode)

                tfwriter.add_scalar('reward/last', reward, episode)
                tfwriter.add_scalar('reward/best', env.best_reward, episode)
                tfwriter.add_scalar('info/ppl', info['ppl'], episode)
                tfwriter.add_scalar('info/compress_ratio', info['compress_ratio'], episode)
                tfwriter.add_scalar('info/para_ratio', info['para_ratio'], episode)
                tfwriter.add_text('info/best_policy', str(env.best_strategy), episode)

                # record the preserve rate for each layer
                for i, preserve_rate in enumerate(env.action):
                    tfwriter.add_scalar('preserve_rate/{}'.format(i), preserve_rate, episode)

                text_writer.write('best reward: {}\n'.format(env.best_reward))
                text_writer.write('best policy: {}\n'.format(env.best_strategy))

        summary = agent.update()

    text_writer.close()


def export_model(env, args):
    assert (args.preserve_ratio is not None) or (args.ratios is not None), 'Please provide a valid ratio'
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
        if args.structure:
            ratios = np.ones(env.num_hidden_layers * 2)
        else:
            ratios = np.ones(env.num_hidden_layers * 6)
        ratios = list(ratios * args.preserve_ratio)

    print('=> Pruning with ratios: {}'.format(ratios))

    env.step(ratios)

    return

def get_agent(nb_states, nb_actions):

    net1 = MLP([args.hidden1, args.hidden1], nn.ReLU, nb_states, nb_actions)
    # net1 = RMlp('lstm', 1, 256, [args.hidden1, args.hidden2], nn.ReLU, nb_states, nb_actions)
    explorer = Gaussian(nb_actions, 1.0)
    actor = Actor(net1, explorer)
    net2 = MLP([args.hidden2, args.hidden2], nn.ReLU, nb_states, 1)
    # net2 = RMlp('lstm', 1, 256, [args.hidden1, args.hidden2], nn.ReLU, nb_states, 1)
    critic = Critic(net2)
    ppo = PPO(actor, critic, 1, args.num_collect, args.learning_epoch, 1,
              clip_param=args.clip_param,
              entropy_coef=args.entropy_coef,
              value_loss_coef=args.value_loss_coef,
              gamma=args.gamma,
              lamda=args.lamda,
              learning_rate=args.lr_a,
              max_grad_norm=args.max_grad_norm)
    return ppo


if __name__ == "__main__":
    args = parse_args()
    
    # 处理下游任务评估开关
    if hasattr(args, 'enable_downstream'):
        if args.enable_downstream.lower() in ['true', '1', 'yes', 'on']:
            args.enable_downstream = True
        else:
            args.enable_downstream = False
    else:
        args.enable_downstream = True  # 默认开启

    # 智能GPU设置 - 支持多GPU环境和自动分配
    if hasattr(args, 'gpu_id') and args.gpu_id is not None and torch.cuda.is_available():
        # 如果指定了GPU ID，设置可见设备但让PyTorch自动处理分配
        if args.gpu_id < torch.cuda.device_count():
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
            print(f'=> Setting CUDA_VISIBLE_DEVICES to GPU {args.gpu_id}')
            print(f'=> PyTorch will handle device allocation automatically')
        else:
            print(f'=> Warning: GPU {args.gpu_id} not available (only {torch.cuda.device_count()} GPUs found)')
            print(f'=> Using automatic GPU allocation instead')
    else:
        print('=> Using automatic multi-GPU allocation (if available)')

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if args.structure:
        env = ChannelPruningEnv(args.model, args.dataset_name,
                                preserve_ratio=1. if args.job == 'export' else args.preserve_ratio,
                                n_data_worker=args.n_worker, batch_size=args.data_bsize,
                                args=args, export_model=args.job == 'export', use_new_input=args.use_new_input)
    else:
        prune_n, prune_m = 0, 0
        if args.sparsity_type is not None:
            # assert args.preserve_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
            prune_n, prune_m = map(int, args.sparsity_type.split(":"))
            args.recon = True
        env = WeightPruningEnv(args.model, args.dataset_name,
                                preserve_ratio=1. if args.job == 'export' else args.preserve_ratio, batch_size=args.data_bsize,
                                args=args, prune_n=prune_n, prune_m=prune_m, export_model=args.job == 'export', use_new_input=args.use_new_input)

    
    if args.job == 'train':
        # === 集成 FeatureExtractor ===
        if args.use_new_input:
            print("=> Using new input features, starting FeatureExtractor...")
            # 1. 从环境中获取数据加载器和可剪枝模块列表
            # 注意：环境在初始化时已经准备好了数据加载器
            # 我们需要重新获取一个数据加载器，因为环境中的加载器可能不适合特征提取
            # 使用正确的参数调用 get_loaders
            train_loader, val_loader = get_loaders(
                name=args.dataset_name,
                nsamples=args.n_samples,
                seed=args.seed,
                seqlen=env.model.seqlen,
                tokenizer=env.tokenizer
            )
            
            prunable_modules = env.prunable_module_names
            
            # 2. 创建并运行 FeatureExtractor（极致内存优化）
            print("=> Initializing FeatureExtractor...")
            # 极大减少样本数以避免显存不足
            max_samples_for_features = min(8, args.n_samples // 8)  # 使用1/8的样本或最多8个
            print(f"=> 极致内存优化：使用 {max_samples_for_features} 个样本进行特征提取")
            
            feature_extractor = FeatureExtractor(
                model=env.model,
                dataloader=train_loader,  # 使用训练集进行特征提取
                prunable_module_names=prunable_modules,
                max_samples=max_samples_for_features
            )
            
            print("=> Extracting features...")
            static_state_tensor = feature_extractor.extract()
            
            # 3. 将提取的特征设置到环境中
            print("=> Setting static state in the environment...")
            env.set_static_state(static_state_tensor)
            print("=> Feature extraction and setup complete.")
        # ==========================

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

        if args.use_new_input:
            nb_states = 8  # FeatureExtractor generates 8 features per module
        else:
            nb_states = 1
        
        if args.structure:
            nb_actions = env.num_hidden_layers * 2  # head and ffn
        else:
            nb_actions = env.num_hidden_layers * 7  # k, v, q, out_proj, fc1 , fc2

        agent = get_agent(nb_states, nb_actions)
        if args.agent_path is not None:
            sd = torch.load(args.agent_path)
            agent.load_state_dict(sd)
        train(args.train_episode, agent, env, args.output)

    elif args.job == 'export':
        export_model(env, args)
    else:
        raise RuntimeError('Undefined job {}'.format(args.job))
