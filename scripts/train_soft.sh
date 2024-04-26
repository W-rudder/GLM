#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline


accelerate launch \
    --config_file accelerate_config/my_config_0.yaml \
    --main_process_port 25678 \
    train_glm.py \
        --freeze_llama \
        --dataset arxiv \
        --pretrain_gnn 'GraphSAGE_arxiv_512_neg0_PCA_1000_tp_undir_run_3_test.pth' \
        --gnn_type 'SoftPrompt' \
        --att_d_model 2048 \
        --gnn_output 4096 \
        --neck 512 \
	    --grad_steps 1 \
        --batch_size 2 \
        --num_token 5 \
        --clip_grad_norm 1.0 \
        --backbone '/home/zuographgroup/zhr/model/vicuna-7b-v1.5' \
        --epoch 1 \
	    --weight_decay 0. \
        --max_text_length 700 \
        --gen_max_length 64 \
	    --lr 0.002 \
        --prefix 'soft_prompt' \
        --seed 42