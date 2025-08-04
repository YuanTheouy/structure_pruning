python amc_searchPPO.py \
    --job=train \
    --model=/home/theo/data/yx_repository/01_Models/opt-1.3b \
    --model_name=opt-1.3b \
    --preserve_ratio=0.7 \
    --prune=para \
    --lbound=0.2 \
    --rbound=1.0 \
    --structure \
    --n_samples=64\
    --num_collect=15\
    --learning_epoch=10\
    --use_real_val \
    --acc_metric=acc1 \
    --reward=reward_ppl \
    --train_episode=3000 \
    --seed=2025 \
    --export_path=./checkpoints/opt-1.3b_piqa_export.pth.tar
    