# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import os
import sys
import numpy as np
import argparse
from copy import deepcopy
import torch
from torch import nn
torch.backends.cudnn.deterministic = True

from env.channel_pruning_env_llm_global import ChannelPruningEnv
from env.weight_pruning_env_llm_global import WeightPruningEnv
from lib.ppo.ppo.ppo_lstm import MLP, PPO, Actor, Critic, Gaussian, GumbelActor
from lib.utils import get_output_folder
from tensorboardX import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaTokenizer
from feature_extractor import FeatureOrchestrator  # 使用新的模块化特征编排器
from feature_configs import get_config_by_name, PREDEFINED_CONFIGS  # 导入配置管理
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
    parser.add_argument('--reward', default='reward_ppl', type=str, help='Setting the reward')
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
    parser.add_argument('--state_mode', default=1, type=int, choices=[0, 1], 
                        help='Agent state mode: 0=global pruning ratio, 1=feature extraction state')
    parser.add_argument('--feature_config', default='default', type=str, 
                        choices=list(PREDEFINED_CONFIGS.keys()),
                        help='Name of the feature configuration to use from feature_configs.py')
    parser.add_argument('--metric', default='input', type=str, help='input or wanda')
    parser.add_argument('--feat', default='mean', type=str, help='mean or flat')

    # >>> ADD THE FOLLOWING CODE BLOCK
    # Gumbel-Softmax specific parameters
    parser.add_argument('--use_gumbel_softmax', action='store_true', 
                        help='Use Gumbel-Softmax for action selection.')
    parser.add_argument('--num_action_bins', default=4, type=int, 
                        help='Number of discrete action bins for each module.')
    parser.add_argument('--gumbel_tau_initial', default=2.0, type=float, 
                        help='Initial temperature for Gumbel-Softmax.')
    parser.add_argument('--gumbel_tau_final', default=0.2, type=float, 
                        help='Final temperature for Gumbel-Softmax.')
    parser.add_argument('--gumbel_anneal_episodes', default=500, type=int, 
                        help='Number of episodes to anneal Gumbel temperature.')
    # >>> END OF CODE BLOCK

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
    global tfwriter, text_writer
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    summary=None

    # Setup Gumbel-Softmax temperature annealing schedule if enabled
    if args.use_gumbel_softmax:
        tau_schedule = np.linspace(args.gumbel_tau_initial, args.gumbel_tau_final, args.gumbel_anneal_episodes)
        print(f"Gumbel temperature annealing enabled: from {args.gumbel_tau_initial} to {args.gumbel_tau_final} over {args.gumbel_anneal_episodes} episodes.")

    while episode < num_episode:  # counting based on episode
        
        # Update Gumbel temperature at the start of each episode
        if args.use_gumbel_softmax:
            if episode < args.gumbel_anneal_episodes:
                current_tau = tau_schedule[episode]
            else:
                current_tau = args.gumbel_tau_final
            
            # Set the new temperature in the actor
            agent.actor.set_tau(current_tau)
            
            # Log the temperature periodically
            if episode % 20 == 0: # Log every 20 episodes
                tfwriter.add_scalar('hparams/gumbel_tau', current_tau, episode)

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
                save_interval = max(1, int(num_episode / 100))  # Avoid division by zero
                if episode % save_interval == 0:
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

    # === 为 export 模式设置静态状态 (如果使用新输入特征) ===
    if args.use_new_input:
        print("=> Export mode: Setting up feature-based state...")
        
        # 获取特征配置
        from feature_configs import get_config_by_name
        from feature_extractor import FeatureOrchestrator
        from lib.data import get_loaders
        import torch
        import os
        
        master_config = get_config_by_name('comprehensive') 
        exp_config = get_config_by_name(args.feature_config)
        print(f"=> Using feature configuration: '{args.feature_config}'")

        # 准备数据加载器和模块列表
        train_loader, _ = get_loaders(
            name=args.dataset_name, nsamples=args.n_samples,
            seed=args.seed, seqlen=env.model.seqlen, tokenizer=env.tokenizer
        )
        prunable_modules = env.prunable_module_names

        # 计算/加载特征张量
        print("=> Initializing orchestrator for export mode...")
        orchestrator = FeatureOrchestrator(
            model=env.model, dataloader=train_loader,
            prunable_module_names=prunable_modules,
            feature_config=master_config,
            max_samples=min(64, args.n_samples),
            cache_dir="./feature_cache"  # 使用默认缓存目录
        )
        master_features_tensor = orchestrator.extract()

        # 根据实验配置筛选特征
        all_feature_names = [f.name for f in orchestrator.active_module_features]
        selected_indices = [
            i for i, name in enumerate(all_feature_names) 
            if exp_config.get(name, False)
        ]
        selected_features_tensor = master_features_tensor[:, selected_indices]
        
        # 组装最终的全局状态向量
        state_features_flat = selected_features_tensor.flatten()
        preserve_ratio_tensor = torch.tensor([env.preserve_ratio], dtype=torch.float32)
        final_state_vector = torch.cat((state_features_flat, preserve_ratio_tensor))
        
        # 设置静态状态到环境中
        env.set_static_state(final_state_vector.numpy())
        print("=> Feature-based state setup complete for export mode.")

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

def get_agent(nb_states, nb_actions, args):
    """
    根据状态和动作维度，构建并返回一个PPO智能体。
    网络结构已增强，以适应高维状态输入。
    """
    actor_hidden_layers = [args.hidden1, args.hidden1, args.hidden1 // 2]
    critic_hidden_layers = [args.hidden2, args.hidden2, args.hidden2 // 2]
    
    # >>> START OF MODIFICATION <<<
    if args.use_gumbel_softmax:
        # If using Gumbel-Softmax, the Actor's MLP outputs logits for each bin of each action.
        actor_output_dim = nb_actions * args.num_action_bins
        print(f"Building Gumbel-Softmax Actor. Input: {nb_states}, Output Logits: {actor_output_dim} ({nb_actions} actions x {args.num_action_bins} bins)")
    else:
        # Original logic
        actor_output_dim = nb_actions
        print(f"Building standard Gaussian Actor. Input: {nb_states}, Output Actions: {actor_output_dim}")

    net1 = MLP(actor_hidden_layers, nn.ReLU, nb_states, actor_output_dim)
    
    if args.use_gumbel_softmax:
        # Create the action bins tensor. It will be moved to the correct device by the GumbelActor's buffer mechanism.
        action_bins = torch.linspace(args.lbound, args.rbound, args.num_action_bins, dtype=torch.float32)
        
        # Instantiate the real GumbelActor, replacing the placeholder.
        actor = GumbelActor(net1, nb_actions, args.num_action_bins, action_bins)
    else:
        # This is the original logic for backward compatibility.
        explorer = Gaussian(nb_actions, 1.0)
        actor = Actor(net1, explorer)
    # >>> END OF MODIFICATION <<<
    
    print(f"Building Critic network with input_dim={nb_states}, hidden_layers={critic_hidden_layers}")
    net2 = MLP(critic_hidden_layers, nn.ReLU, nb_states, 1)
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

    # 强制GPU绑定 - 确保每个进程严格使用指定GPU
    # 严格GPU绑定 - 简单直接的方案
    if hasattr(args, 'gpu_id') and args.gpu_id is not None and torch.cuda.is_available():
        # 强制设置CUDA_VISIBLE_DEVICES，确保进程只能看到指定GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        print(f'=> [STRICT GPU BINDING] CUDA_VISIBLE_DEVICES={args.gpu_id}')
        print(f'=> Process will only see GPU {args.gpu_id}')
        
        # 由于只有一个GPU可见，PyTorch设备ID就是0
        if torch.cuda.device_count() > 0:
            torch.cuda.set_device(0)
            print(f'=> PyTorch device set to cuda:0 (physical GPU {args.gpu_id})')
        else:
            print(f'=> ERROR: No CUDA devices visible after setting CUDA_VISIBLE_DEVICES={args.gpu_id}')
            sys.exit(1)
    elif torch.cuda.is_available():
        print('=> No GPU ID specified, using default GPU allocation')
    else:
        print('=> CUDA not available, using CPU')

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
        # === 将原有的 if args.use_new_input: 块替换为以下内容 ===
        if args.use_new_input:
            print("=> State Mode 1: Using feature-based state.")
            
            # --- 1. 定义并获取特征配置 ---
            # 主配置，用于计算并缓存所有可用特征
            master_config = get_config_by_name('comprehensive') 
            # 本次实验要使用的配置
            exp_config = get_config_by_name(args.feature_config)
            print(f"=> Using feature configuration: '{args.feature_config}'")

            # --- 2. 准备数据加载器和模块列表 (逻辑不变) ---
            train_loader, _ = get_loaders(
                name=args.dataset_name, nsamples=args.n_samples,
                seed=args.seed, seqlen=env.model.seqlen, tokenizer=env.tokenizer
            )
            prunable_modules = env.prunable_module_names

            # --- 3. 计算/加载 "母版" 特征张量 ---
            # 使用"全面配置"来确保一次性计算并缓存所有特征
            print("=> Initializing orchestrator with 'comprehensive' config to get all features...")
            orchestrator = FeatureOrchestrator(
                model=env.model, dataloader=train_loader,
                prunable_module_names=prunable_modules,
                feature_config=master_config,
                max_samples=min(64, args.n_samples),
                cache_dir=os.path.join(args.output, "feature_cache")
            )
            master_features_tensor = orchestrator.extract() # Shape: [num_modules, num_all_features]

            # --- 4. 根据实验配置筛选特征 (核心步骤) ---
            print(f"=> Selecting features based on '{args.feature_config}' config...")
            
            # 获取所有可用特征的名称顺序
            all_feature_names = [f.name for f in orchestrator.active_module_features]
            
            # 找出本次实验需要保留的特征的索引
            selected_indices = [
                i for i, name in enumerate(all_feature_names) 
                if exp_config.get(name, False)
            ]
            
            # 从母版张量中筛选出本次实验所需的特征列
            selected_features_tensor = master_features_tensor[:, selected_indices]
            
            print(f"   Selected {len(selected_indices)} features out of {len(all_feature_names)}.")
            print(f"   Final module features shape: {selected_features_tensor.shape}")

            # --- 5. 组装最终的全局状态向量 ---
            # 将筛选后的模块特征扁平化
            state_features_flat = selected_features_tensor.flatten()
            
            # 获取全局保留率
            preserve_ratio_tensor = torch.tensor([env.preserve_ratio], dtype=torch.float32)

            # 拼接成最终的、完整的状态向量
            final_state_vector = torch.cat((state_features_flat, preserve_ratio_tensor))
            
            # --- 6. 将最终状态设置到环境中 ---
            print("=> Setting final assembled static state in the environment...")
            env.set_static_state(final_state_vector.numpy())
            print("=> Feature extraction, selection, and setup complete.")
        # =======================================================

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

        # 1. 从环境中动态获取正确的状态维度
        nb_states = env.state_dim
        print(f"=> Correct state dimension from environment: {nb_states}")

        # 2. 获取动作维度 (逻辑不变)
        if args.structure:
            nb_actions = env.num_hidden_layers * 2  # head and ffn
        else:
            nb_actions = env.num_hidden_layers * 7  # k, v, q, out_proj, fc1 , fc2

        print(f"=> Action dimension: {nb_actions}")

        # 3. 将正确的参数传递给 agent，并额外传入 args
        agent = get_agent(nb_states, nb_actions, args)
        if args.agent_path is not None:
            sd = torch.load(args.agent_path)
            agent.load_state_dict(sd)
        train(args.train_episode, agent, env, args.output)

    elif args.job == 'export':
        export_model(env, args)
    else:
        raise RuntimeError('Undefined job {}'.format(args.job))
