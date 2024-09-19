#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

# prefixes=("graphsage_1000tp_5token_512_neg0_arxiv_linear_1_3400")
# test_datasets=("arxiv_LP" "pubmed_LP" "cora_LP")
# dataset='arxiv'
# text_length=750

# # 遍历所有prefix和测试数据集
# for prefix in "${prefixes[@]}"; do
#     for test_dataset in "${test_datasets[@]}"; do
#         echo "Testing with prefix $prefix on dataset $test_dataset"
#         bash ./scripts/test_cross_task.sh $dataset $test_dataset $text_length $prefix
#     done
# done


prefixes=("graphsage_1000tp_5token_512_neg0_computer_2_3400_more_epoch")
test_datasets=("children_LP" "history_LP" "computer_LP" "photo_LP" "sports_LP")
dataset='computer'
text_length=1500

# 遍历所有prefix和测试数据集
for prefix in "${prefixes[@]}"; do
    for test_dataset in "${test_datasets[@]}"; do
        echo "Testing with prefix $prefix on dataset $test_dataset"
        bash ./scripts/test_cross_task.sh $dataset $test_dataset $text_length $prefix
    done
done