# Import necessary modules
import time
import torch
import collections
import torch.nn as nn
import random
import itertools
import sys
# Import get_loaders function from data module within the same directory
from .data import get_loaders 
# from lib.lm_eval_local_backup.evaluator import evaluate, make_table
# from lib.lm_eval_local_backup.tasks import get_task_dict, ALL_TASKS
# from lib.lm_eval_local_backup.utils import pattern_match
# from lib.lm_eval_local_backup.models import get_model

# def get_loader_benchmark(dataset, tokenizer):
#     dataloader = []
#     seqlen=2048

#     dataloader_bench = []
#     task_list = []
#     task_list.append(dataset)
#     task_dict = get_task_dict(task_list)

#     task = task_dict[dataset]
#     task_doc_func = task.training_docs
#     doc = task_doc_func()

#     for i in doc:
#         dataloader.append(task.doc_to_text(i))
    
#     trainenc = tokenizer(" ".join(dataloader), return_tensors='pt')
#     for _ in range(len(dataloader)):
#         i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
#         j = i + seqlen
#         inp = trainenc.input_ids[:, i:j]
#         tar = inp.clone()
#         tar[:, :-1] = -100
#         dataloader_bench.append((inp, tar))

#     return dataloader
#     # return doc, dataloader_bench

# def eval_acc(model, task):
#     lm = get_model("hf-causal")(
#         pretrained=model,
#         batch_size = 4 ,
#     )
#     no_cache = True

#     rnd = random.Random()
#     rnd.seed(42)


#     task_names = task
#     task_names = pattern_match(task_names.split(","), ALL_TASKS)
#     task_dict = get_task_dict(task_names)
#     description_dict = {}
#     results = evaluate(
#         lm=lm,
#         task_dict=task_dict,
#         num_fewshot=0,
#         limit=None,
#         bootstrap_iters=100000,
#         description_dict=description_dict,
#         decontamination_ngrams_path=None,
#         write_out=False,
#         output_base_path=None,
#     )

#     # task = task_dict[task]
#     # description = (
#     #         description_dict[task_names]
#     #         if description_dict and task_names in description_dict
#     #         else ""
#     #     )

#     # limit=None
#     # task_doc_func = task.validation_docs
#     # task_docs = list(task_doc_func())
#     # requests = collections.defaultdict(list)
#     # requests_origin = collections.defaultdict(list)
#     # docs = {}
#     # for doc_id, doc in enumerate(itertools.islice(task_docs, 0, limit)):
#     #     docs = doc
#     #     ctx = task.fewshot_context(doc=doc, num_fewshot=0, rnd=rnd, description=description)
#     #     reqs = task.construct_requests(doc, ctx)
#     #     if not isinstance(reqs, (list, tuple)):
#     #         reqs = [reqs]
#     #     for i, req in enumerate(reqs):
#     #         requests[req.request_type].append(req)
#     #         requests_origin[req.request_type].append((i, task_names, doc, doc_id))
    
#     # for reqtype, reqs in requests.items():
#     #     resps = getattr(lm, reqtype)([req.args for req in reqs])
#     #     resps = [
#     #         x if req.index is None else x[req.index] for x, req in zip(resps, reqs)
#     #     ]
#     #     for resp, (i, task_name, doc, doc_id) in zip(resps, requests_origin[reqtype]):
#     #         process_res_queue[(task_name, doc_id)].append((i, resp))
    
#     # vals = collections.defaultdict(list)
#     # for (task_name, doc_id), requests in process_res_queue.items():
#     #     requests.sort(key=lambda x: x[0])
#     #     requests = [x[1] for x in requests]
#     #     doc = docs
#     #     metrics = task.process_results(doc, reqs)
#     #     for metric, value in metrics.items():
#     #         vals[(task_name, metric)].append(value)

#     # for (task_name, metric), items in vals.items():
#     #     task = task_dict[task_name]
#     #     real_metric = metric  # key when looking up the metric with task.aggregation
#     #     if metric.endswith(decontaminate_suffix):
#     #         real_metric = metric.replace(
#     #             decontaminate_suffix, ""
#     #         )  # decontaminated still uses the same metric
#     #     results[task_name][metric] = task.aggregation()[real_metric](items)

#     return results["results"][task]['acc']

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
# MODIFIED: This function can now accept an override dataset
def eval_ppl(model, tokenizer, device=None, dataset_override=None):
    """
    评估模型的PPL。如果提供了dataset_override，则使用它，否则加载默认的wikitext2测试集。
    """
    dataset = "wikitext2"
    
    if device is None:
        device = next(model.parameters()).device

    # This logic correctly passes the dataset (either override or default) to the core function
    if dataset_override is not None:
        # print(f"INFO: eval_ppl is using the provided override dataset.")
        testenc = dataset_override # 直接使用传入的数据集
    else:
        print("INFO: eval_ppl is loading the default full wikitext2 test set.")
        _, testloader = get_loaders(
            dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
        )
        testenc = testloader # 在您的代码中，testloader就是testenc张量
    
    # 调用核心函数并返回浮点数结果
    return eval_ppl_wikitext(model, testenc, 1, device).item()

# MODIFIED: This function now correctly handles both a tensor and a list of samples
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    """
    PPL评估的核心逻辑
    """
    # --- [关键修正] ---
    # 我们从环境中收到的 'testenc' 现在是一个样本列表 (a LIST of samples)。
    # 原始代码期望一个单一的张量 (a TENSOR)。
    # 我们必须在执行任何操作之前，将这个列表转换回它所期望的单一长张量。
    
    if isinstance(testenc, list):
        if not testenc: # 处理列表为空的边缘情况
            print("WARNING: eval_ppl_wikitext received an empty list for evaluation.")
            return torch.tensor(float('inf'))
        # 列表中的每个 'item' 是一个元组 (input_tensor, target_tensor)
        # 我们将所有的 input_tensor 沿着维度1拼接起来，形成一个单一的长序列
        input_ids_list = [item[0] for item in testenc]
        testenc_tensor = torch.cat(input_ids_list, dim=1)
    else: 
        # 为原始行为保留的回退路径，此时 testenc 是一个封装了张量的对象
        testenc_tensor = testenc.input_ids
    
    # 确保最终的张量在正确的设备上
    testenc_tensor = testenc_tensor.to(device)
    # --- [修正结束] ---

    # 现在，所有后续代码都必须使用 'testenc_tensor'
    nsamples = testenc_tensor.numel() // model.seqlen
    if nsamples == 0:
        # print("WARNING: Not enough tokens in the evaluation set for even one sample.")
        return torch.tensor(float('inf'))
        
    nlls = []

    with torch.no_grad():
        for i in range(0, nsamples, bs):
            j = min(i + bs, nsamples)
            # 使用我们处理过的 testenc_tensor
            inputs = testenc_tensor[:, (i * model.seqlen):(j * model.seqlen)]
            inputs = inputs.reshape(j - i, model.seqlen)

            lm_logits = model(inputs).logits
            
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            neg_log_likelihood = loss.float() * model.seqlen * (j - i)
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    torch.cuda.empty_cache()

    return ppl