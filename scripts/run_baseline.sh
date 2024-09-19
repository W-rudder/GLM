#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

port=25680
dataset='arxiv'
test_dataset='arxiv'
length=700
prefix='graphsage_1000tp_5token_512_neg0_arxiv_linear_1_3400'


suffix='arxiv_baseline1'
seed=1
bash ./scripts/test_baseline.sh $port $dataset $test_dataset $length $prefix $suffix $seed 

suffix='arxiv_baseline2'
seed=2
bash ./scripts/test_baseline.sh $port $dataset $test_dataset $length $prefix $suffix $seed 
