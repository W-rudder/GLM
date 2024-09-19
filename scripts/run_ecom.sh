#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

neck=512
grad_steps=1
batch_size=2

dataset='computer'
# unfinished test
# prefix='graphsage_1000tp_5token_512_neg0_computer_linear'
# pretrain_gnn='GraphSAGE_computer_512_neg0_PCA_1000_tp_undir.pth'
# prefix='graphsage_5tp_5token_256_neg1_computer_linear'
# pretrain_gnn='GraphSAGE_computer_256_neg1_kmean_5_tp_undir.pth'
# prefix='graphsage_0tp_5token_512_neg_computer_linear'
# pretrain_gnn='Grace_computer_undir_512_neg.pth'
# seed=2
# num_token=5
# prefix='graphsage_1000tp_5token_512_neg0_computer_2_3400_more_epoch_seed2'
# pretrain_gnn='GraphSAGE_computer_512_neg0_PCA_1000_tp_undir_run_2_3400_more_epoch.pth'

# bash ./scripts/train_computer.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed $num_token

# bash ./scripts/test_children.sh $dataset $neck $prefix $num_token

# bash ./scripts/test_history.sh $dataset $neck $prefix $num_token

# bash ./scripts/test_computer.sh $dataset $neck $prefix $num_token

# bash ./scripts/test_photo.sh $dataset $neck $prefix $num_token

# bash ./scripts/test_sports.sh $dataset $neck $prefix $num_token

# seed=3
# num_token=5
# prefix='graphsage_1000tp_5token_512_neg0_computer_2_3400_more_epoch_seed3'
# pretrain_gnn='GraphSAGE_computer_512_neg0_PCA_1000_tp_undir_run_2_3400_more_epoch.pth'

# bash ./scripts/train_computer.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed $num_token

# bash ./scripts/test_children.sh $dataset $neck $prefix $num_token

# bash ./scripts/test_history.sh $dataset $neck $prefix $num_token

# bash ./scripts/test_computer.sh $dataset $neck $prefix $num_token

# bash ./scripts/test_photo.sh $dataset $neck $prefix $num_token

# bash ./scripts/test_sports.sh $dataset $neck $prefix $num_token

seed=4
num_token=5
prefix='graphsage_1000tp_5token_512_neg0_computer_2_3400_more_epoch_seed4'
pretrain_gnn='GraphSAGE_computer_512_neg0_PCA_1000_tp_undir_run_2_3400_more_epoch.pth'

bash ./scripts/train_computer.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed $num_token

bash ./scripts/test_children.sh $dataset $neck $prefix $num_token

bash ./scripts/test_history.sh $dataset $neck $prefix $num_token

bash ./scripts/test_computer.sh $dataset $neck $prefix $num_token

bash ./scripts/test_photo.sh $dataset $neck $prefix $num_token

bash ./scripts/test_sports.sh $dataset $neck $prefix $num_token