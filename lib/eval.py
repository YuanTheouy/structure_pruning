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
def eval_ppl(model, tokenizer, device=None):
    # Set dataset
    dataset = "wikitext2"
    
    # 智能设备选择 - 使用模型实际所在的设备
    if device is None:
        device = next(model.parameters()).device
        # print(f"=> PPL evaluation using model device: {device}")

    # Print status
    # print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    # print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        # if i % 50 == 0:
            # print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)


        # print ("nlls",nlls)
        sys.stdout.flush()

    
    # print ('begin calcualte ppl')
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()