#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline


accelerate launch \
    --config_file accelerate_config/my_config_0.yaml \
    --main_process_port 25678 \
    train_glm.py \
        --freeze_llama \
        --inference \
        --zero_shot \
        --best_epoch 0 \
        --gnn_type 'SoftPrompt' \
        --dataset 'computer' \
        --test_dataset $1 \
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
        --prefix 'soft_prompt_1' \
        --suffix 'LP_soft_prompt_1' \
        --embed_type 'bert' \
        --conv_type 'sage' \
        --llm_type 'vicuna'