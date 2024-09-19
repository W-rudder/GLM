#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline


accelerate launch \
    --config_file accelerate_config/my_config_0.yaml \
    --main_process_port 25679 \
    train_glm.py \
        --freeze_llama \
        --inference \
        --zero_shot \
        --best_epoch 0 \
        --dataset 'computer' \
        --test_dataset photo_LP \
        --neck 512 \
        --att_d_model 2048 \
        --gnn_output 4096 \
	    --grad_steps 1 \
        --batch_size 4 \
        --num_token 5 \
        --clip_grad_norm 1.0 \
        --backbone '/home/zuographgroup/zhr/model/vicuna-7b-v1.5' \
        --epoch 1 \
	    --weight_decay 0.1 \
        --max_text_length 1500 \
        --gen_max_length 64 \
	    --lr 0.001 \
        --prefix 'graphsage_1000tp_5token_512_neg0_computer_2_3400_more_epoch' \
        --suffix 'LP_baseline' \
        --ablation