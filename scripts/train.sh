#!/bin/bash
# This script trains the model with the best performing arguments.
# Please specify your own run name below.
RUN_NAME=your_run_name

python -Wignore -m metric_wsd.run \
    --run-name $RUN_NAME \
    --mode train \
    --gpus 1 \
    --batch_size 5 \
    --max_epochs 200 \
    --lr 1e-5 \
    --model-type cbert-proto \
    --episodic \
    --dist dot \
    --ks_support_kq_query 5 50 5 \
    --max_inference_supports 30
