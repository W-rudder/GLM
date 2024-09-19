#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

neck=512
grad_steps=1
batch_size=2

dataset='computer'
llm='/home/wangduo/zhr/model/vicuna-7b-v1.5'
conv='sage'
num_token=5
gnn_type='SoftPrompt'
embed='bert'
llm_type='vicuna'
pretrain_gnn='GraphSAGE_computer_512_neg0_PCA_1000_tp_undir_run_2_3400_more_epoch.pth'

seed=0
prefix='soft_promt_1'

# bash ./scripts/train_computer.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_children.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_history.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

# bash ./scripts/test_computer.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

# bash ./scripts/test_photo.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

# bash ./scripts/test_sports.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

seed=1
prefix='soft_promt_2'

# bash ./scripts/train_computer.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_children.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_history.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

# bash ./scripts/test_computer.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

# bash ./scripts/test_photo.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

# bash ./scripts/test_sports.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

seed=2
prefix='soft_promt_3'

# bash ./scripts/train_computer.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_children.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_history.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

# bash ./scripts/test_computer.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

# bash ./scripts/test_photo.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

# bash ./scripts/test_sports.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

seed=3
prefix='soft_promt_4'

# bash ./scripts/train_computer.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_children.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_history.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

# bash ./scripts/test_computer.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

# bash ./scripts/test_photo.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

# bash ./scripts/test_sports.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

seed=4
prefix='soft_promt_5'

# bash ./scripts/train_computer.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_children.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

bash ./scripts/test_history.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

# bash ./scripts/test_computer.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

# bash ./scripts/test_photo.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type

# bash ./scripts/test_sports.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type $gnn_type