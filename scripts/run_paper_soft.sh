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
num_token=5
gnn_type='SoftPrompt'
embed='bert'
llm_type='vicuna'
pretrain_gnn='GraphSAGE_arxiv_512_neg0_PCA_1000_tp_undir_run_1_3400.pth'

prefix='soft_prompt_1'
seed=0

bash ./scripts/train_arxiv.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_arxiv.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_pubmed.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_cora.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type


prefix='soft_prompt_2'
seed=1

bash ./scripts/train_arxiv.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_arxiv.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_pubmed.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_cora.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

prefix='soft_prompt_3'
seed=2

bash ./scripts/train_arxiv.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_arxiv.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_pubmed.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_cora.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

prefix='soft_prompt_4'
seed=3

bash ./scripts/train_arxiv.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_arxiv.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_pubmed.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_cora.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

prefix='soft_prompt_5'
seed=4

bash ./scripts/train_arxiv.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_arxiv.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_pubmed.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_cora.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type