#!/bin/bash

#python -m llava.eval.model_vqa_loader \
#    --model-path liuhaotian/llava-v1.5-7b \
#    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#    --image-folder ./playground/data/eval/textvqa/train_images \
#    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b.jsonl \
#    --temperature 0 \
#    --conv-mode vicuna_v1
#
#python -m llava.eval.eval_textvqa \
#    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b.jsonl

python -m llava.eval.model_bw_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file /users/fjc3/sharedscratch/datasets/llava/source_dev.json \
    --image-folder /users/fjc3/sharedscratch/generated_data/576p \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

#python -m llava.eval.eval_textvqa \
#    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b.jsonl
