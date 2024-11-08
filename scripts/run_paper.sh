#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

neck=512
grad_steps=1
batch_size=2
dataset='arxiv'


llm='/home/wangduo/zhr/model/vicuna-7b-v1.5'
conv='sage'
gnn_type='GraphSAGE'
num_token=5
embed='bert'
llm_type='vicuna'


seed=0
prefix='graphsage_1000_5token_512_neg0_new_arxiv'
pretrain_gnn='GraphSAGE_arxiv_512_neg0_PCA_1000_tp_undir_run_1_3400.pth'

bash ./scripts/train_arxiv.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_arxiv.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_pubmed.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_cora.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

# seed=1
# prefix='graphsage_ae_5token_512_neg0_arxiv_linear_2_epoch_100'
# pretrain_gnn='GraphSAGE_arxiv_512_neg0_PCAAotuencoder_1000_run_2_epoch_100.pth'

# bash ./scripts/train_arxiv.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed $num_token $embed $llm $conv $llm_type $gnn_type

# bash ./scripts/test_arxiv.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

# bash ./scripts/test_pubmed.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

# bash ./scripts/test_cora.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type