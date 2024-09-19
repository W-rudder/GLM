#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

neck=512
grad_steps=1
batch_size=2

dataset='arxiv'

seed=0
prefix='graphsage_1000tp_5token_512_neg0_arxiv_linear_1_3400'
# bash ./scripts/train_cross.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix
# bash ./scripts/train_paper_LP.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix
bash ./scripts/test_children.sh $dataset $neck $prefix

bash ./scripts/test_history.sh $dataset $neck $prefix

bash ./scripts/test_computer.sh $dataset $neck $prefix

bash ./scripts/test_photo.sh $dataset $neck $prefix

bash ./scripts/test_sports.sh $dataset $neck $prefix

seed=1
prefix='graphsage_1000tp_5token_512_neg0_arxiv_linear_2_3400'
# bash ./scripts/train_cross.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix
# bash ./scripts/train_paper_LP.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix
bash ./scripts/test_children.sh $dataset $neck $prefix

bash ./scripts/test_history.sh $dataset $neck $prefix

bash ./scripts/test_computer.sh $dataset $neck $prefix

bash ./scripts/test_photo.sh $dataset $neck $prefix

bash ./scripts/test_sports.sh $dataset $neck $prefix