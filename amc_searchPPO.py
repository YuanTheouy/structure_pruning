# -*- coding: utf-8 -*-
# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import os
import sys
import csv
import json
import math
import time
import numpy as np
import argparse
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
torch.backends.cudnn.deterministic = True

from env.channel_pruning_env_llm_global import ChannelPruningEnv
from env.weight_pruning_env_llm_global import WeightPruningEnv
from lib.ppo.ppo.ppo_lstm import MLP, PPO, Actor, Critic, Gaussian, GumbelActor
from lib.utils import get_output_folder, prGreen, prRed
from lib.ew_candidates import CandidateRecorder, dump_json, load_json, read_jsonl, safe_model_name
from lib.ew_projector import compute_policy_budget, make_module_costs, project_score_to_policy
from tensorboardX import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaTokenizer
from feature_extractor import FeatureOrchestrator  # Use new modularized feature orchestrator
from feature_configs import get_config_by_name, PREDEFINED_CONFIGS  # Import configuration management
from lib.data import get_loaders

# ================== MEMORY PROFILING HELPERS ==================
def get_tensor_memory(component):
    """
    Calculate the GPU memory usage of all CUDA tensors in a PyTorch module or optimizer (unit: MiB).
    """
    total_mem = 0
    if isinstance(component, torch.nn.Module):
        # Calculate memory usage of model parameters
        for param in component.parameters():
            if param.is_cuda:
                total_mem += param.element_size() * param.nelement()
    elif isinstance(component, torch.optim.Optimizer):
        # Calculate optimizer state memory usage (e.g., Adam's momentum and variance)
        for state in component.state.values():
            for v in state.values():
                if torch.is_tensor(v) and v.is_cuda:
                     total_mem += v.element_size() * v.nelement()
    # Calculate gradient memory usage
    if isinstance(component, torch.nn.Module):
        for param in component.parameters():
            if param.grad is not None and param.grad.is_cuda:
                total_mem += param.grad.element_size() * param.grad.nelement()

    return total_mem / (1024 ** 2)

def report_vram_usage(point_in_code: str, device=0):
    """
    Report current and peak VRAM usage at specific points in the code.
    """
    print(f"--- VRAM Report at: {point_in_code} ---")
    # torch.cuda.memory_allocated(): Current memory occupied by Tensors
    # torch.cuda.max_memory_allocated(): Peak memory occupied by Tensors since program start or last reset
    # torch.cuda.memory_reserved(): Total memory pool size requested by PyTorch from CUDA driver
    # torch.cuda.max_memory_reserved(): Peak memory pool size requested by PyTorch
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    print(f"  - VRAM Allocated: {allocated:.2f} MiB")
    print(f"  - VRAM Peak Allocated: {max_allocated:.2f} MiB")
    print(f"  - VRAM Reserved: {reserved:.2f} MiB")
    print(f"  - VRAM Peak Reserved: {max_reserved:.2f} MiB")
    print("-" * 50)
# =============================================================


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
    
    # Gradual pruning parameters    
    parser.add_argument('--use_gradual_pruning', action='store_true',
                        help='Enable gradual pruning with a cubic schedule.')
    parser.add_argument('--gradual_final_sparsity', default=0.5, type=float,
                        help='The final target sparsity (1 - preserve_ratio) for the gradual schedule.')
    parser.add_argument('--gradual_initial_sparsity', default=0.0, type=float,
                        help='The initial sparsity to start the gradual schedule from. Default is 0.0.')
    parser.add_argument('--gradual_pruning_start_episode', default=0, type=int,
                        help='The episode number to start the gradual pruning process.')
    parser.add_argument('--gradual_pruning_end_episode', default=600, type=int,
                        help='The episode number to end the gradual pruning process (e.g., 80% of total episodes).')

    # GPU parameter
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU device ID to use')
    
    # Downstream task evaluation
    parser.add_argument('--enable_downstream', default='false', type=str, 
                        help='Enable downstream task evaluation (true/false)')
    parser.add_argument('--delayed_downstream_eval', action='store_true',
                        help='Delay downstream evaluation until after pruning+reconstruction is complete (for export mode)')

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
    parser.add_argument('--state_mode', default=0, type=int, choices=[0, 1], 
                        help='Agent state mode: 0=global pruning ratio (default), 1=feature extraction state')
    parser.add_argument('--feature_config', default='default', type=str, 
                        choices=list(PREDEFINED_CONFIGS.keys()),
                        help='Name of the feature configuration to use from feature_configs.py')
    parser.add_argument('--metric', default='input', type=str, help='input or wanda')
    parser.add_argument('--feat', default='mean', type=str, help='mean or flat')
    parser.add_argument('--resume_from_checkpoint', default=None, type=str, help='Path to a training checkpoint to resume from.')

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
    # MODIFICATION 1: Add new arguments
    # Remove old reward subset parameters since we're simplifying
    # parser.add_argument('--reward_subset_size_small', type=float, default=0.05, 
    #                     help='Percentage of validation set for fast evaluation')
    # parser.add_argument('--reward_subset_size_large', type=float, default=0.2,
    #                     help='Percentage of validation set for accurate evaluation')
    # parser.add_argument('--use_staged_eval', action='store_true',
    #                     help='Enable staged evaluation (small batch then large batch)')
    
    # Dataset progressive growth parameters
    parser.add_argument('--use_dataset_growth', action='store_true',
                        help='Enable dataset progressive growth with a cubic schedule.')
    parser.add_argument('--dataset_initial_ratio', default=1.0, type=float,
                        help='Initial ratio of the dataset to use (0.0-1.0).')
    parser.add_argument('--dataset_final_ratio', default=1.0, type=float,
                        help='Final ratio of the dataset to use (0.0-1.0).')
    parser.add_argument('--dataset_growth_start_episode', default=0, type=int,
                        help='Episode number to start the dataset growth process.')
    parser.add_argument('--dataset_growth_end_episode', default=200, type=int,
                        help='Episode number to end the dataset growth process.')

    # Early-Warning Guided Candidate Reranking
    parser.add_argument('--save_candidates', action='store_true',
                        help='Save endpoint candidates for Early-Warning probing without changing PPO training.')
    parser.add_argument('--candidate_save_mode', default='topk',
                        choices=['topk', 'periodic', 'topk_and_periodic'],
                        help='Candidate saving policy.')
    parser.add_argument('--candidate_top_k', '--top_k', dest='candidate_top_k', default=50, type=int,
                        help='Number of endpoint-best candidates to keep in candidates.jsonl. Use <=0 to keep all.')
    parser.add_argument('--save_every', default=None, type=int,
                        help='Save one candidate every N endpoint evaluations in periodic modes.')
    parser.add_argument('--candidate_dir', default=None, type=str,
                        help='Directory for candidates, probe results, rerank outputs, and final policy metadata.')
    parser.add_argument('--candidate_output_dir', default=None, type=str,
                        help='Alias for --candidate_dir.')
    parser.add_argument('--run_id', default=None, type=str,
                        help='Run id used under results/{model}/{sparsity}/{run_id}/.')
    parser.add_argument('--candidates_path', default=None, type=str,
                        help='Path to candidates.jsonl for probe/rerank/compile jobs.')
    parser.add_argument('--ew_delta', default=0.05, type=float,
                        help='Neighboring-budget delta for Early-Warning probe.')
    parser.add_argument('--probe_sparsity', default=None, type=float,
                        help='Override diagnostic center sparsity. Defaults to each candidate current_sparsity.')
    parser.add_argument('--probe_output', default=None, type=str,
                        help='CSV path for Early-Warning probe results.')
    parser.add_argument('--probe_jsonl_output', default=None, type=str,
                        help='JSONL path for Early-Warning probe results.')
    parser.add_argument('--num_shards', default=1, type=int,
                        help='Number of candidate shards for probe jobs.')
    parser.add_argument('--shard_id', default=0, type=int,
                        help='Current shard id for probe jobs.')
    parser.add_argument('--rerank_mode', default='curvature', choices=['endpoint', 'slope', 'curvature'],
                        help='Candidate reranking mode.')
    parser.add_argument('--lambda_ew', default=1.0, type=float,
                        help='Penalty weight for curvature reranking.')
    parser.add_argument('--lambda_slope', default=1.0, type=float,
                        help='Penalty weight for slope reranking.')
    parser.add_argument('--curvature_tau', default=0.0, type=float,
                        help='Curvature threshold tau for max(0, curvature - tau).')
    parser.add_argument('--rerank_output', default=None, type=str,
                        help='CSV path for rerank results.')
    parser.add_argument('--best_candidate_path', default=None, type=str,
                        help='JSON path for selected best candidate.')
    parser.add_argument('--final_sparsity', default=None, type=float,
                        help='Final sparsity for compiling the selected candidate. Defaults to 1 - preserve_ratio.')
    parser.add_argument('--final_policy_path', default=None, type=str,
                        help='JSON path for the final projected policy.')
    
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


def default_candidate_dir(args):
    target_sparsity = getattr(args, "final_sparsity", None)
    if target_sparsity is None:
        target_sparsity = 1.0 - float(getattr(args, "preserve_ratio", 0.7))
    run_id = getattr(args, "run_id", None)
    if not run_id:
        seed = getattr(args, "seed", None)
        run_id = f"seed{seed if seed is not None else 'none'}"
    return os.path.join(
        "results",
        safe_model_name(args.model_name),
        f"sparsity_{float(target_sparsity):.2f}",
        safe_model_name(run_id),
        "candidates",
    )


def resolve_candidate_dir(args):
    if getattr(args, "candidate_output_dir", None):
        return args.candidate_output_dir
    if args.candidate_dir:
        return args.candidate_dir
    return default_candidate_dir(args)


def resolve_candidates_path(args):
    if args.candidates_path:
        return args.candidates_path
    return os.path.join(resolve_candidate_dir(args), "candidates.jsonl")


def resolve_probe_output(args):
    if args.probe_output:
        return args.probe_output
    return os.path.join(resolve_candidate_dir(args), "probe_results.csv")


def resolve_probe_jsonl_output(args):
    if args.probe_jsonl_output:
        return args.probe_jsonl_output
    return os.path.join(resolve_candidate_dir(args), "probe_results.jsonl")


def resolve_rerank_output(args):
    if args.rerank_output:
        return args.rerank_output
    return os.path.join(resolve_candidate_dir(args), "rerank_results.csv")


def resolve_best_candidate_path(args):
    if args.best_candidate_path:
        return args.best_candidate_path
    return os.path.join(resolve_candidate_dir(args), "best_candidate.json")


def resolve_selected_candidates_path(args):
    return os.path.join(resolve_candidate_dir(args), "selected_candidates.json")


def build_env_module_costs(env):
    if env.args.prune == "para":
        return make_module_costs(
            env.param_list,
            norm_costs=getattr(env, "norm_para", None),
            dim_list=env.dim_list,
            channel_round=env.channel_round,
            cost_type="para",
        )
    return make_module_costs(
        env.flops_list,
        dim_list=env.dim_list,
        channel_round=env.channel_round,
        cost_type="flops",
    )


def project_candidate_score(env, score_vector, sparsity):
    return project_score_to_policy(
        score_vector,
        target_sparsity=sparsity,
        module_costs=build_env_module_costs(env),
        p_min=env.lbound,
        p_max=env.rbound,
    )


def project_candidate_score_with_metadata(env, score_vector, sparsity):
    return project_score_to_policy(
        score_vector,
        target_sparsity=sparsity,
        module_costs=build_env_module_costs(env),
        p_min=env.lbound,
        p_max=env.rbound,
        return_metadata=True,
    )


def load_candidate_score(candidate):
    if candidate.get("score_path"):
        return torch.load(candidate["score_path"], map_location="cpu").detach().cpu().numpy()
    if candidate.get("score_vector") is not None:
        return np.asarray(candidate["score_vector"], dtype=np.float32)
    raise ValueError(f"Candidate {candidate.get('candidate_id')} does not include score_path or score_vector")


def evaluate_projected_policy(env, policy, checkpoint_path=None):
    old_export_model = env.export_model
    old_export_path = getattr(env, "export_path", None)
    try:
        env.reset()
        env.export_model = True
        if checkpoint_path is not None:
            env.set_export_path(checkpoint_path)
        obs, reward, done, info = env.step(list(policy))
        if checkpoint_path is not None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(env.model.state_dict(), checkpoint_path)
        return {
            "obs": obs,
            "reward": reward,
            "done": done,
            "info": info,
            "ppl": info.get("ppl"),
            "policy": info.get("strategy", policy),
        }
    finally:
        env.export_model = old_export_model
        if old_export_path is not None:
            env.set_export_path(old_export_path)


def write_policy_json(path, candidate_id, policy, sparsity, extra=None):
    payload = {
        "candidate_id": candidate_id,
        "sparsity": float(sparsity),
        "policy": policy,
    }
    if extra:
        payload.update(extra)
    dump_json(path, payload)


def run_probe(env, args):
    candidates_path = resolve_candidates_path(args)
    candidates = read_jsonl(candidates_path)
    if not candidates:
        raise RuntimeError(f"No candidates found in {candidates_path}")
    candidates = candidates[: args.candidate_top_k] if args.candidate_top_k and args.candidate_top_k > 0 else candidates
    num_shards = max(1, int(getattr(args, "num_shards", 1)))
    shard_id = int(getattr(args, "shard_id", 0))
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
    candidates = [candidate for idx, candidate in enumerate(candidates) if idx % num_shards == shard_id]

    output_path = resolve_probe_output(args)
    jsonl_path = resolve_probe_jsonl_output(args)
    probe_dir = Path(output_path).parent
    policy_dir = probe_dir / "probe_policies"
    policy_dir.mkdir(parents=True, exist_ok=True)

    completed = set()
    if os.path.exists(output_path):
        for row in read_csv_rows(output_path):
            completed.add(row["candidate_id"])

    rows = []
    delta = float(args.ew_delta)
    for index, candidate in enumerate(candidates):
        candidate_id = candidate.get("candidate_id", f"candidate_{index:05d}")
        if candidate_id in completed:
            print(f"=> Skipping already probed candidate {candidate_id}")
            continue
        center_sparsity = (
            float(args.probe_sparsity)
            if args.probe_sparsity is not None
            else float(candidate.get("current_sparsity", 1.0 - args.preserve_ratio))
        )
        score_vector = load_candidate_score(candidate)

        probe_values = {}
        for tag, sparsity in (
            ("minus", center_sparsity - delta),
            ("zero", center_sparsity),
            ("plus", center_sparsity + delta),
        ):
            sparsity = float(np.clip(sparsity, 0.0, 1.0))
            projection = project_candidate_score_with_metadata(env, score_vector, sparsity)
            policy = projection["policy"]
            policy_path = policy_dir / f"{candidate_id}_{tag}.json"
            write_policy_json(policy_path, candidate_id, policy, sparsity, {"probe_point": tag, "projection": projection})

            start_time = time.time()
            eval_result = evaluate_projected_policy(env, policy)
            elapsed = time.time() - start_time
            ppl = float(eval_result["ppl"])
            logppl = math.log(ppl) if ppl and ppl > 0 else float("inf")

            probe_values[tag] = {
                "sparsity": sparsity,
                "policy_path": str(policy_path),
                "ppl": ppl,
                "logppl": logppl,
                "reward": float(eval_result["reward"]),
                "eval_seconds": elapsed,
                "actual_sparsity": projection["actual_sparsity"],
                "budget_error": projection["budget_error"],
                "relative_budget_error": projection["relative_budget_error"],
            }

        log_minus = probe_values["minus"]["logppl"]
        log_zero = probe_values["zero"]["logppl"]
        log_plus = probe_values["plus"]["logppl"]
        slope = (log_plus - log_zero) / delta
        curvature = (log_plus - 2.0 * log_zero + log_minus) / (delta ** 2)
        local_probe_degradation = log_plus - log_zero

        row = {
            "candidate_id": candidate_id,
            "score_path": candidate.get("score_path", ""),
            "candidate_policy_path": candidate.get("policy_path", ""),
            "model_name": candidate.get("model_name", args.model_name),
            "seed": candidate.get("seed", args.seed),
            "step": candidate.get("step", ""),
            "center_sparsity": center_sparsity,
            "delta": delta,
            "sparsity_minus": probe_values["minus"]["sparsity"],
            "sparsity_zero": probe_values["zero"]["sparsity"],
            "sparsity_plus": probe_values["plus"]["sparsity"],
            "actual_sparsity_minus": probe_values["minus"]["actual_sparsity"],
            "actual_sparsity_zero": probe_values["zero"]["actual_sparsity"],
            "actual_sparsity_plus": probe_values["plus"]["actual_sparsity"],
            "budget_error_minus": probe_values["minus"]["budget_error"],
            "budget_error_zero": probe_values["zero"]["budget_error"],
            "budget_error_plus": probe_values["plus"]["budget_error"],
            "relative_budget_error_minus": probe_values["minus"]["relative_budget_error"],
            "relative_budget_error_zero": probe_values["zero"]["relative_budget_error"],
            "relative_budget_error_plus": probe_values["plus"]["relative_budget_error"],
            "ppl_minus": probe_values["minus"]["ppl"],
            "ppl_zero": probe_values["zero"]["ppl"],
            "ppl_plus": probe_values["plus"]["ppl"],
            "ppl_0": probe_values["zero"]["ppl"],
            "logppl_minus": log_minus,
            "logppl_zero": log_zero,
            "logppl_plus": log_plus,
            "ell_minus": log_minus,
            "ell_0": log_zero,
            "ell_plus": log_plus,
            "slope": slope,
            "curvature": curvature,
            "local_probe_degradation": local_probe_degradation,
            "probe_plus_minus_zero_degradation": local_probe_degradation,
            "policy_minus_path": probe_values["minus"]["policy_path"],
            "policy_zero_path": probe_values["zero"]["policy_path"],
            "policy_plus_path": probe_values["plus"]["policy_path"],
            "eval_seconds_minus": probe_values["minus"]["eval_seconds"],
            "eval_seconds_zero": probe_values["zero"]["eval_seconds"],
            "eval_seconds_plus": probe_values["plus"]["eval_seconds"],
            "eval_seconds_total": (
                probe_values["minus"]["eval_seconds"]
                + probe_values["zero"]["eval_seconds"]
                + probe_values["plus"]["eval_seconds"]
            ),
            "eval_num_samples": getattr(args, "n_samples", ""),
            "seq_len": getattr(env.model, "seqlen", ""),
            "gpu_id": getattr(args, "gpu_id", ""),
            "num_shards": num_shards,
            "shard_id": shard_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        rows.append(row)
        Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"=> Probed {candidate_id}: logppl0={log_zero:.4f}, curvature={curvature:.4f}")

    if not rows:
        print(f"=> No new candidates to probe for shard {shard_id}/{num_shards}")
        return output_path

    existing_rows = []
    if os.path.exists(output_path):
        existing_rows = read_csv_rows(output_path)
    all_rows = existing_rows + rows
    fieldnames = []
    for row in all_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"=> Early-Warning probe results saved to {output_path}")
    return output_path


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def float_field(row, key, default=float("nan")):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def run_rerank(args):
    probe_path = resolve_probe_output(args)
    rows = read_csv_rows(probe_path)
    if not rows:
        raise RuntimeError(f"No probe rows found in {probe_path}")

    candidates_by_id = {}
    candidates_path = resolve_candidates_path(args)
    if os.path.exists(candidates_path):
        candidates_by_id = {row["candidate_id"]: row for row in read_jsonl(candidates_path)}

    scored_rows = []
    for row in rows:
        endpoint_score = float_field(row, "logppl_zero")
        slope_score = endpoint_score + args.lambda_ew * float_field(row, "slope")
        curvature_penalty = max(0.0, float_field(row, "curvature") - args.curvature_tau)
        curvature_score = endpoint_score + args.lambda_ew * curvature_penalty
        selected_score = {
            "endpoint": endpoint_score,
            "slope": slope_score,
            "curvature": curvature_score,
        }[args.rerank_mode]

        out = dict(row)
        out.update({
            "endpoint_score": endpoint_score,
            "slope_score": slope_score,
            "curvature_score": curvature_score,
            "warning_penalty": curvature_penalty,
            "selected_mode": args.rerank_mode,
            "selected_score": selected_score,
            "lambda_ew": args.lambda_ew,
            "curvature_tau": args.curvature_tau,
        })
        scored_rows.append(out)

    scored_rows.sort(key=lambda row: float(row["selected_score"]))
    for rank, row in enumerate(scored_rows, start=1):
        row["rank"] = rank

    output_path = resolve_rerank_output(args)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scored_rows)

    best = dict(scored_rows[0])
    original_candidate = candidates_by_id.get(best["candidate_id"], {})
    best_payload = {
        "selected_mode": args.rerank_mode,
        "selected_score": float(best["selected_score"]),
        "lambda_slope": args.lambda_slope,
        "lambda_ew": args.lambda_ew,
        "curvature_tau": args.curvature_tau,
        "probe_row": best,
        "candidate": original_candidate,
    }
    best_path = resolve_best_candidate_path(args)
    dump_json(best_path, best_payload)

    selected_payload = {}
    for mode, score_key in (
        ("endpoint", "endpoint_score"),
        ("slope", "slope_score"),
        ("curvature", "curvature_score"),
    ):
        mode_best = min(scored_rows, key=lambda item: float(item[score_key]))
        selected_payload[f"{mode}_best"] = {
            "mode": mode,
            "score": float(mode_best[score_key]),
            "probe_row": mode_best,
            "candidate": candidates_by_id.get(mode_best["candidate_id"], {}),
        }
    selected_path = resolve_selected_candidates_path(args)
    dump_json(selected_path, selected_payload)

    print(f"=> Rerank results saved to {output_path}")
    print(f"=> Best candidate saved to {best_path}: {best['candidate_id']} ({args.rerank_mode})")
    print(f"=> Selected candidates saved to {selected_path}")
    print("=> Top 10 candidates:")
    for row in scored_rows[:10]:
        print(f"   rank={row['rank']} id={row['candidate_id']} score={float(row['selected_score']):.6f}")
    return best_path


def run_compile_best(env, args):
    best_path = resolve_best_candidate_path(args)
    best_payload = load_json(best_path)
    candidate = best_payload.get("candidate") or {}
    probe_row = best_payload.get("probe_row") or {}
    candidate_id = candidate.get("candidate_id") or probe_row.get("candidate_id")

    if not candidate.get("score_path") and probe_row.get("score_path"):
        candidate["score_path"] = probe_row["score_path"]
    score_vector = load_candidate_score(candidate)

    final_sparsity = float(args.final_sparsity) if args.final_sparsity is not None else 1.0 - args.preserve_ratio
    final_projection = project_candidate_score_with_metadata(env, score_vector, final_sparsity)
    final_policy = final_projection["policy"]
    final_policy_path = args.final_policy_path or os.path.join(resolve_candidate_dir(args), "final_policy.json")
    write_policy_json(
        final_policy_path,
        candidate_id,
        final_policy,
        final_sparsity,
        {"source_best_candidate": best_path, "selected_mode": best_payload.get("selected_mode")},
    )

    export_path = args.export_path or os.path.join(resolve_candidate_dir(args), "final_static_checkpoint.pth.tar")
    print(f"=> Compiling candidate {candidate_id} at final sparsity {final_sparsity:.4f}")
    eval_result = evaluate_projected_policy(env, final_policy, checkpoint_path=export_path)

    metadata_path = os.path.splitext(export_path)[0] + ".json"
    dump_json(metadata_path, {
        "candidate_id": candidate_id,
        "final_sparsity": final_sparsity,
        "final_policy_path": final_policy_path,
        "checkpoint_path": export_path,
        "ppl": eval_result["ppl"],
        "reward": eval_result["reward"],
        "actual_sparsity": final_projection["actual_sparsity"],
        "budget_error": final_projection["budget_error"],
        "relative_budget_error": final_projection["relative_budget_error"],
        "selected_mode": best_payload.get("selected_mode"),
        "best_candidate_path": best_path,
    })

    print(f"=> Final static checkpoint saved to {export_path}")
    print(f"=> Final metadata saved to {metadata_path}")
    return export_path


def train(num_episode, agent, env, output, candidate_recorder=None):
    global tfwriter, text_writer
    # ====================  ▼▼▼ 添加这个代码块 ▼▼▼ ====================
    start_episode = 0
    if args.resume_from_checkpoint and os.path.isfile(args.resume_from_checkpoint):
        print(f"=> Loading checkpoint '{args.resume_from_checkpoint}'")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location='cpu')
        
        agent.load_state_dict(checkpoint['agent_state_dict'])
        start_episode = checkpoint['episode']
        env.best_reward = checkpoint.get('best_reward', -1e9)
        env.best_strategy = checkpoint.get('best_strategy', None)
        
        print(f"=> Loaded checkpoint. Resuming training from episode {start_episode}")
    # ======================================================================
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    summary = None
    
    # --- 初始化Epoch级逻辑所需的状态变量 ---
    last_update_episode = 0

    # 渐进式剪枝参数设置
    if args.use_gradual_pruning:
        print("=> Gradual Pruning Schedule Enabled.")
        print(f"   Target Sparsity: {args.gradual_final_sparsity}")
        print(f"   Schedule: Episode {args.gradual_pruning_start_episode} to {args.gradual_pruning_end_episode}")
        
        t0 = args.gradual_pruning_start_episode
        tf = args.gradual_pruning_end_episode
        pruning_duration = tf - t0
        if pruning_duration <= 0:
            raise ValueError("gradual_pruning_end_episode must be greater than start_episode.")
    
    # 数据集渐进增长参数设置
    if args.use_dataset_growth:
        print("=> Dataset Progressive Growth Schedule Enabled (Staircase Mode).")
        print(f"   Initial Dataset Ratio: {args.dataset_initial_ratio} ({args.dataset_initial_ratio*100:.1f}% of validation set)")
        print(f"   Final Dataset Ratio: {args.dataset_final_ratio} ({args.dataset_final_ratio*100:.1f}% of validation set)")
        print(f"   Schedule: Episode {args.dataset_growth_start_episode} to {args.dataset_growth_end_episode}")
        
        dt0 = args.dataset_growth_start_episode
        dtf = args.dataset_growth_end_episode
        dataset_growth_duration = dtf - dt0
        if dataset_growth_duration <= 0:
            raise ValueError("dataset_growth_end_episode must be greater than start_episode.")
    else:
        print("=> Using full validation set for evaluation.")
    
    # Gumbel-Softmax温度退火设置
    if args.use_gumbel_softmax:
        tau_schedule = np.linspace(args.gumbel_tau_initial, args.gumbel_tau_final, args.gumbel_anneal_episodes)
        print(f"Gumbel temperature annealing enabled: from {args.gumbel_tau_initial} to {args.gumbel_tau_final} over {args.gumbel_anneal_episodes} episodes.")

    current_preserve_ratio = args.preserve_ratio
    
    # --- [主训练循环开始] ---
    while episode < num_episode:
        # ====================  ▼▼▼ 添加这三行 ▼▼▼ ====================
        if episode < start_episode:
            episode += 1
            continue
        # ======================================================================
        # --- [Epoch级逻辑] - 在每个更新周期(e.g., 15 episodes)开始前执行 ---
        # 这个代码块确保了评估集的大小和内容只在周期的边界发生改变。
        if episode - last_update_episode >= agent.num_collects or episode == 0:
            if episode > 0:
                print(f"--- Epoch End (Episode {episode}) ---")
            
            # --- [修改后的核心逻辑] ---
            # 1. 首先处理数据集尺寸的阶梯式增长
            if args.use_dataset_growth:
                dataset_progress = np.clip((episode - dt0) / dataset_growth_duration, 0.0, 1.0)
                initial_ratio = args.dataset_initial_ratio
                final_ratio = args.dataset_final_ratio
                current_dataset_ratio = initial_ratio + (final_ratio - initial_ratio) * dataset_progress**3
                current_dataset_ratio = max(initial_ratio, min(current_dataset_ratio, final_ratio))
                
                # 更新环境中的数据集比例 (这将自动触发重采样)
                if hasattr(env, 'update_dataset_ratio'):
                    env.update_dataset_ratio(current_dataset_ratio)
                
                print(f"=> New Epoch Start: Current dataset ratio updated to {current_dataset_ratio:.3f} ({current_dataset_ratio*100:.1f}%)")
                tfwriter.add_scalar('hparams/dataset_ratio', current_dataset_ratio, episode)

            # 2. 如果不使用增长模式，也要在每个周期开始时重采样，以防止过拟合
            else:
                 print(f"=> New Epoch Start: Resampling reward evaluation set...")
                 if hasattr(env, 'resample_reward_eval_set'):
                    env.resample_reward_eval_set()
            # --- [修改结束] ---

            print("---------------------------------")
            last_update_episode = episode
        
        # --- [Episode级数据收集循环] ---
        # 在这个循环中，数据集的大小和内容将保持不变
        for i in range(agent.num_collects):
            with torch.inference_mode():
                if observation is None:
                    if args.use_gradual_pruning and episode < args.gradual_pruning_start_episode:
                        initial_preserve_ratio = 1.0 - args.gradual_initial_sparsity
                        env.update_target_ratio(initial_preserve_ratio)
                    
                    observation = deepcopy(env.reset())

                if args.use_gradual_pruning:
                    current_progress = np.clip((episode - t0) / pruning_duration, 0.0, 1.0)
                    preserve_i = 1.0 - args.gradual_initial_sparsity
                    preserve_f = args.preserve_ratio
                    current_preserve_ratio = preserve_f + (preserve_i - preserve_f) * (1 - current_progress)**3
                    env.update_target_ratio(current_preserve_ratio)
                    current_sparsity = 1.0 - current_preserve_ratio
                
                # Gumbel温度更新逻辑
                if args.use_gumbel_softmax:
                    current_tau = args.gumbel_tau_final
                    if episode < args.gumbel_anneal_episodes:
                        current_tau = tau_schedule[episode]
                    agent.actor.set_tau(current_tau)
                    if episode % 20 == 0:
                        tfwriter.add_scalar('hparams/gumbel_tau', current_tau, episode)

                # State形状处理逻辑
                if not args.use_new_input:
                    if isinstance(observation, (int, float)):
                        observation = np.array([observation], dtype=np.float32)
                    elif observation.ndim == 0:
                        observation = np.expand_dims(observation, 0)
                    if observation.ndim == 1:
                        observation = np.expand_dims(observation, 0)
                else:
                    if observation.ndim == 1:
                        observation = np.expand_dims(observation, 0)
                    elif observation.ndim == 0:
                        observation = np.array([[observation]], dtype=np.float32)
                
                if episode == 0 and i == 0: # 只在最开始打印一次
                    print(f"=> [调试] 状态形状: {observation.shape}, 状态内容: {observation}")
                
                # Agent核心交互
                action = agent.act(observation)
                raw_action = np.squeeze(action)
                next_observation, reward, done, info = env.step(raw_action)
                
                # 对 next_observation 进行同样的状态形状处理
                if not args.use_new_input:
                    if isinstance(next_observation, (int, float)):
                        next_observation = np.array([next_observation], dtype=np.float32)
                    elif hasattr(next_observation, 'ndim') and next_observation.ndim == 0:
                        next_observation = np.expand_dims(next_observation, 0)
                    if hasattr(next_observation, 'ndim') and next_observation.ndim == 1:
                        next_observation = np.expand_dims(next_observation, 0)
                else:
                    if hasattr(next_observation, 'ndim'):
                        if next_observation.ndim == 1:
                            next_observation = np.expand_dims(next_observation, 0)
                        elif next_observation.ndim == 0:
                            next_observation = np.array([[next_observation]], dtype=np.float32)
                
                agent.step(next_observation, reward, done, np.array([0], dtype=bool))
                observation = deepcopy(next_observation)
                
            if done:
                episode_reward += reward
                log_message_template = ("#{episode}: reward={reward:.4f}, ppl={ppl:.4f}, compress_ratio={compress:.4f}, para_ratio={para:.4f}, expect_preserve_ratio={expect_preserve:.4f}\nPolicy: {policy}")
                log_content = log_message_template.format(episode=episode, reward=episode_reward, ppl=info.get('ppl', float('nan')), compress=info['compress_ratio'], para=info['para_ratio'], expect_preserve=current_preserve_ratio, policy=env.action)
                
                is_best = reward > env.best_reward
                is_nan = np.isnan(info.get('ppl', float('nan')))

                if is_nan: prRed(log_content)
                elif is_best: prGreen(log_content)
                else: print(log_content)
                
                text_writer.write(log_content + '\n')

                if candidate_recorder is not None:
                    target_sparsity = 1.0 - current_preserve_ratio
                    budget = compute_policy_budget(
                        info.get("strategy", env.action),
                        target_sparsity,
                        build_env_module_costs(env),
                    )
                    candidate_id = "{}_seed{}_step{:06d}_ep{:06d}".format(
                        safe_model_name(args.model_name),
                        args.seed if args.seed is not None else "none",
                        step,
                        episode,
                    )
                    candidate_recorder.record(
                        candidate_id=candidate_id,
                        score_vector=raw_action,
                        projected_policy=info.get("strategy", env.action),
                        current_sparsity=target_sparsity,
                        target_sparsity=target_sparsity,
                        actual_sparsity=budget["actual_sparsity"],
                        budget_error=budget["budget_error"],
                        relative_budget_error=budget["relative_budget_error"],
                        endpoint_ppl=info.get("ppl", float("nan")),
                        endpoint_reward=reward,
                        step=step,
                        seed=args.seed,
                        model_name=args.model_name,
                        config=vars(args),
                        extra={
                            "episode": episode,
                            "compress_ratio": info.get("compress_ratio"),
                            "para_ratio": info.get("para_ratio"),
                            "output_dir": output,
                        },
                    )
                
                if is_best:
                    env.best_reward = reward
                    env.best_strategy = env.action.copy()
                    env.best_d_prime_list = env.d_prime_list.copy()
                    # ==================== ▼▼▼ 替换/修改成这样 ▼▼▼ ====================
                    checkpoint_path = os.path.join(output, 'checkpoint_best.pth.tar')
                    torch.save({
                        'episode': episode + 1,
                        'agent_state_dict': agent.state_dict(),
                        'best_reward': env.best_reward,
                        'best_strategy': env.best_strategy,
                    }, checkpoint_path)
                    
                    # (可选) 如果你还想保留原来的 actor.pt, critic.pt 文件，可以保留这行
                    agent.save_model(output)
                    prGreen(f"    -> New best reward found and comprehensive checkpoint saved.")
                    # ======================================================================
                
                observation = None
                episode_steps = 0
                episode_reward = 0.
                episode += 1
                step += 1

                if summary is not None:
                    for k, v in summary.items(): tfwriter.add_scalar(k, v, episode)

                tfwriter.add_scalar('reward/last', reward, episode)
                tfwriter.add_scalar('reward/best', env.best_reward, episode)
                tfwriter.add_scalar('info/ppl', info['ppl'], episode)
                tfwriter.add_scalar('info/compress_ratio', info['compress_ratio'], episode)
                tfwriter.add_scalar('info/para_ratio', info['para_ratio'], episode)
                tfwriter.add_text('info/best_policy', str(env.best_strategy), episode)

                for i, preserve_rate in enumerate(env.action):
                    tfwriter.add_scalar('preserve_rate/{}'.format(i), preserve_rate, episode)

                text_writer.write('best reward: {}\n'.format(env.best_reward))
                text_writer.write('best policy: {}\n'.format(env.best_strategy))

        # --- 在数据收集完毕后，进行PPO更新 ---
        summary = agent.update()

    # --- 训练结束后 ---
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
    
    # 如果启用了延迟下游任务评估，在剪枝+重构完成后进行评估
    if getattr(args, 'delayed_downstream_eval', False) and getattr(args, 'enable_downstream', 'false').lower() == 'true':
        print("\n=> Performing delayed downstream task evaluation on pruned+reconstructed model...")
        try:
            success = env.test_model(env.model)
            if success:
                print("=> Post-pruning downstream task evaluation completed successfully")
            else:
                print("=> Post-pruning downstream task evaluation completed with warnings")
        except Exception as e:
            print(f"=> WARNING: Post-pruning downstream evaluation failed: {str(e)}")

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
    # GPU可用性检查与日志记录
    if torch.cuda.is_available():
        # Python代码不再修改CUDA_VISIBLE_DEVICES，它由启动脚本全权负责
        # 我们只打印出当前进程能看到的GPU数量
        print(f"=> PyTorch detected {torch.cuda.device_count()} available GPU(s).")
        print(f"=> This process can see GPUs with IDs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All')}")
        # Hugging Face device_map='auto' 会自动处理多GPU的分配
    else:
        print('=> CUDA not available, using CPU')
        
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # # 零点：在任何东西加载到GPU之前
    # torch.cuda.empty_cache()
    # report_vram_usage("Initial State")
    
    # === 根据 state_mode 参数决定是否使用新输入特征 ===
    # 在创建环境之前设置，确保环境创建时使用正确的参数
    if args.state_mode == 1:
        args.use_new_input = True
        print("=> State Mode 1: 启用特征提取状态")
    else:
        args.use_new_input = False
        print("=> State Mode 0: 使用全局剪枝率状态")

    if args.job == 'rerank':
        run_rerank(args)
        sys.exit(0)

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

    
    # # 节点1：LLM模型加载后
    # report_vram_usage("After LLM (env) Initialization")
    # llm_mem = get_tensor_memory(env.model)
    # print(f">>> Breakdown: Llama 7B model parameters occupy ~{llm_mem:.2f} MiB")
    
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

        candidate_recorder = None
        if args.save_candidates:
            candidate_dir = resolve_candidate_dir(args)
            candidate_recorder = CandidateRecorder(
                candidate_dir,
                top_k=args.candidate_top_k,
                save_mode=args.candidate_save_mode,
                save_every=args.save_every,
                run_config=vars(args),
            )
            print(f"=> Saving Early-Warning candidates to {candidate_dir}")

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
            
        # # 节点2：RL Agent创建后
        # report_vram_usage("After RL Agent Initialization")
        # actor_mem = get_tensor_memory(agent.actor)
        # critic_mem = get_tensor_memory(agent.critic)
        # print(f">>> Breakdown: RL Agent parameters (Actor: {actor_mem:.2f} MiB, Critic: {critic_mem:.2f} MiB)")
    
        train(args.train_episode, agent, env, args.output, candidate_recorder=candidate_recorder)

    elif args.job == 'export':
        export_model(env, args)
    elif args.job == 'probe':
        run_probe(env, args)
    elif args.job == 'compile':
        run_compile_best(env, args)
    elif args.job == 'ew_pipeline':
        run_probe(env, args)
        run_rerank(args)
        run_compile_best(env, args)
    else:
        raise RuntimeError('Undefined job {}'.format(args.job))
