#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

TASK_NAME="artifact_function"
for FOLD in 0 1 2 3 4; do
DATA_DIR="data/$TASK_NAME/cross_val/$FOLD"
OUTPUT_DIR="models/model_""$TASK_NAME/$FOLD"

python main.py \
--model_type bert \
--model_name_or_path bert-base-uncased \
--task_name $TASK_NAME \
--encode_type fndef \
--do_train \
--do_eval \
--do_predict \
--data_dir $DATA_DIR \
--output_dir $OUTPUT_DIR \
--learning_rate 2e-5 \
--num_train_epochs 10 \
--max_seq_length 200 \
--per_gpu_eval_batch_size 1 \
--per_gpu_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--overwrite_output \
--overwrite_cache

done
