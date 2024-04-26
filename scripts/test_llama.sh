#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline


accelerate launch \
    --config_file accelerate_config/my_config_0.yaml \
    train_glm.py \
        --freeze_llama \
        --inference \
        --llm_type opt \
        --zero_shot \
        --graph_unsup \
        --best_epoch 0 \
        --dataset chembl \
        --test_dataset muv \
        --att_d_model 2048 \
        --gnn_output 4096 \
	    --grad_steps 1 \
        --batch_size 1 \
        --clip_grad_norm 1.0 \
        --backbone '/home/zuographgroup/zhr/model/galactica-6.7b' \
        --epoch 1 \
	    --weight_decay 0.1 \
        --max_text_length 800 \
        --gen_max_length 64 \
	    --lr 0.001 \
        --prefix 'graphsage_5tp_100token_opt'