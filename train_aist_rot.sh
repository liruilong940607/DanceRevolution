#! /bin/bash

# export CUDA_VISIBLE_DEVICES=3

model_dir=60FPS_aist_rot_v2

python train.py --train_dir data_aist_rot/train \
                --output_dir ${model_dir} \
                --batch_size 32 \
                --lr 0.0001 \
                --dropout 0.05 \
                --frame_dim 438 \
                --encoder_hidden_size 1024 \
                --pose_dim 219 \
                --decoder_hidden_size 512 \
                --seq_len 420 \
                --max_seq_len 4500 \
                --num_heads 8 \
                --num_layers 3 \
                --window_size 100 \
                --fixed_step 10 \
                --alpha 0.04 \
                --save_per_epochs 20 \
                --aist \
                --rotmat 
