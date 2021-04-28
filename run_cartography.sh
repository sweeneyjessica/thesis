#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis-env

python -m cartography.classification.run_glue \
    -c configs/mnli.jsonnet \
    --do_train \
    --do_eval \
    -o roberta-output/

python -m cartography.selection.train_dy_filtering \
    --filter \
    --task_name MNLI \
    --model_dir roberta-output/training_dynamics/ \
    --metric confidence \
    --data_dir data/ 

