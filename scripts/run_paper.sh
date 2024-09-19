#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

neck=512
grad_steps=1
batch_size=2
dataset='arxiv'


llm='/home/zuographgroup/zhr/model/Llama-2-7b-chat-hf'
conv='sage'
seed=0
num_token=5
embed='bert'
llm_type='llama2'
prefix='graphsage_1000tp_5token_512_neg0_arxiv_linear_1_3400_llama'
pretrain_gnn='GraphSAGE_arxiv_512_neg0_PCA_1000_tp_undir_run_1_3400_llama.pth'

bash ./scripts/train_arxiv.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed $num_token $embed $llm $conv $llm_type

bash ./scripts/test_arxiv.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type

bash ./scripts/test_pubmed.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type

bash ./scripts/test_cora.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type


llm='/home/zuographgroup/zhr/model/vicuna-13b-v1.5'
conv='sage'
seed=0
num_token=20
embed='bert'
llm_type='vicuna'
prefix='graphsage_1000tp_20token_512_neg0_arxiv_linear_1_3400_13b_1'
pretrain_gnn='GraphSAGE_arxiv_512_neg0_PCA_1000_tp_undir_run_1_3400_13b.pth'

bash ./scripts/train_arxiv.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed $num_token $embed $llm $conv $llm_type

bash ./scripts/test_arxiv.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type

bash ./scripts/test_pubmed.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type

bash ./scripts/test_cora.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type