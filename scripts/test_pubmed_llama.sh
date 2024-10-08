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
        --llm_type 'llama2' \
        --dataset 'arxiv' \
        --test_dataset pubmed \
        --neck 512 \
        --att_d_model 2048 \
        --gnn_output 4096 \
	    --grad_steps 1 \
        --batch_size 1 \
        --num_token 5 \
        --clip_grad_norm 1.0 \
        --backbone '/home/zuographgroup/zhr/model/Llama-2-7b-chat-hf' \
        --epoch 1 \
	    --weight_decay 0.1 \
        --max_text_length 700 \
        --gen_max_length 64 \
	    --lr 0.001 \
        --prefix 'graphsage_0tp_5token_512_neg0_arxiv_linear_2_3400' \
        --suffix 'llama2_baseline' \
        --ablation