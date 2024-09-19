#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

port=25681
dataset='computer'
test_dataset='computer'
length=500
prefix='graphsage_0tp_5token_512_neg0_computer_2_3400_more_epoch'


suffix='computer_baseline1'
seed=1
bash ./scripts/test_baseline.sh $port $dataset $test_dataset $length $prefix $suffix $seed
