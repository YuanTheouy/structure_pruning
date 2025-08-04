python amc_searchBenchmark.py \
    --job=benchmark \
    --model=/public/experiments/wjt/llama/llama1-7b  \
    --model_name=llama1-7b \
    --dataset_name=arc_challenge\
    --preserve_ratio=0.8\
    --use_real_val\
    --prune=para \
    --recon\
    --structure \
    --lbound=0.3 \
    --rbound=1 \
    --m=3\
    --n_samples=64\
    --recon_sample=32\
    --use_real_val \
    --acc_metric=acc1 \
    --reward=reward_ppl \
    --seed=2024\
    --start=63\
    --resume_path=./checkpoints/llama1-7b_arc_challenge_resume.pth.tar\
    --export_path=./checkpoints/llama1-7b_arc_challenge_export.pth.tar
 
 