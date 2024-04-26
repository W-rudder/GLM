#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline


accelerate launch \
    --config_file accelerate_config/my_config_0.yaml \
    train_glm.py \
        --freeze_llama \
        --graph_unsup \
        --llm_type opt \
        --dataset chembl \
        --att_d_model 2048 \
        --gnn_output 4096 \
	    --grad_steps 2 \
        --batch_size 4 \
        --clip_grad_norm 1.0 \
        --backbone '/home/zuographgroup/zhr/model/galactica-6.7b' \
        --epoch 1 \
	    --weight_decay 0. \
        --max_text_length 860 \
        --gen_max_length 64 \
	    --lr 0.002 \
        --prefix 'graphsage_5tp_100token_opt'