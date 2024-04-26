#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

neck=512
grad_steps=1
batch_size=2


dataset='arxiv'

# seed=0
# prefix='graphsage_0tp_5token_512_neg0_arxiv_linear_1_3400'
# pretrain_gnn='Grace_arxiv_512_neg0_undir_run_1_3400.pth'

# bash ./scripts/train_arxiv.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed

# bash ./scripts/test_arxiv.sh $dataset $neck $prefix

# bash ./scripts/test_pubmed.sh $dataset $neck $prefix

# bash ./scripts/test_cora.sh $dataset $neck $prefix

seed=1
prefix='graphsage_0tp_5token_512_neg0_arxiv_linear_2_3400'
pretrain_gnn='Grace_arxiv_512_neg0_undir_run_2_3400.pth'

bash ./scripts/train_arxiv.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed

bash ./scripts/test_arxiv.sh $dataset $neck $prefix

bash ./scripts/test_pubmed.sh $dataset $neck $prefix

bash ./scripts/test_cora.sh $dataset $neck $prefix

seed=2
prefix='graphsage_0tp_5token_512_neg0_arxiv_linear_3_3400'
pretrain_gnn='Grace_arxiv_512_neg0_undir_run_3_3400.pth'

bash ./scripts/train_arxiv.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed

bash ./scripts/test_arxiv.sh $dataset $neck $prefix

bash ./scripts/test_pubmed.sh $dataset $neck $prefix

bash ./scripts/test_cora.sh $dataset $neck $prefix

seed=0
prefix='graphsage_1000tp_5token_512_neg0_arxiv_linear_1_3400'
pretrain_gnn='GraphSAGE_arxiv_512_neg0_PCA_1000_tp_undir_run_1_3400.pth'

bash ./scripts/train_arxiv.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed

bash ./scripts/test_arxiv.sh $dataset $neck $prefix

bash ./scripts/test_pubmed.sh $dataset $neck $prefix

bash ./scripts/test_cora.sh $dataset $neck $prefix

seed=1
prefix='graphsage_1000tp_5token_512_neg0_arxiv_linear_2_3400'
pretrain_gnn='GraphSAGE_arxiv_512_neg0_PCA_1000_tp_undir_run_2_3400.pth'

bash ./scripts/train_arxiv.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed

bash ./scripts/test_arxiv.sh $dataset $neck $prefix

bash ./scripts/test_pubmed.sh $dataset $neck $prefix

bash ./scripts/test_cora.sh $dataset $neck $prefix

seed=2
prefix='graphsage_1000tp_5token_512_neg0_arxiv_linear_3_3400'
pretrain_gnn='GraphSAGE_arxiv_512_neg0_PCA_1000_tp_undir_run_3_3400.pth'

bash ./scripts/train_arxiv.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed

bash ./scripts/test_arxiv.sh $dataset $neck $prefix

bash ./scripts/test_pubmed.sh $dataset $neck $prefix

bash ./scripts/test_cora.sh $dataset $neck $prefix