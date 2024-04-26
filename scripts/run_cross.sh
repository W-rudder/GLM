#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

dataset='paper_LP'
neck=512
grad_steps=1
batch_size=2
prefix='graphsage_5tp_5token_paper_lp'
# pretrain_gnn='GraphSAGE_5_arxiv_no_sim_undir.pth'
pretrain_gnn='Grace_arxiv_undir.pth'

# bash ./scripts/train_cross.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix
bash ./scripts/train_paper_LP.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix

bash ./scripts/test_arxiv.sh $dataset $neck $prefix

bash ./scripts/test_pubmed.sh $dataset $neck $prefix

bash ./scripts/test_cora.sh $dataset $neck $prefix

grad_steps=2
batch_size=4
prefix='graphsage_5tp_5token_paper_lp_bs'

# bash ./scripts/train_cross.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix
bash ./scripts/train_paper_LP.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix

bash ./scripts/test_arxiv.sh $dataset $neck $prefix

bash ./scripts/test_pubmed.sh $dataset $neck $prefix

bash ./scripts/test_cora.sh $dataset $neck $prefix