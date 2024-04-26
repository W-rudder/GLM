#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

neck=512
grad_steps=1
batch_size=2

dataset='arxiv'
prefix='graphsage_0tp_5token_linear'

bash ./scripts/test_children.sh $dataset $neck $prefix

bash ./scripts/test_history.sh $dataset $neck $prefix

bash ./scripts/test_computer.sh $dataset $neck $prefix

bash ./scripts/test_photo.sh $dataset $neck $prefix

bash ./scripts/test_sports.sh $dataset $neck $prefix

dataset='computer'
prefix='graphsage_0tp_5token_computer_linear'

bash ./scripts/test_arxiv.sh $dataset $neck $prefix

bash ./scripts/test_pubmed.sh $dataset $neck $prefix

bash ./scripts/test_cora.sh $dataset $neck $prefix

dataset='book_children'
prefix='graphsage_0tp_5token_children_linear'

bash ./scripts/test_arxiv.sh $dataset $neck $prefix

bash ./scripts/test_pubmed.sh $dataset $neck $prefix

bash ./scripts/test_cora.sh $dataset $neck $prefix