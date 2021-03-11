#! /bin/bash

# Generate pose sequences of json format and visualize them
python test.py --input_dir data_aist_rot/train \
               --model 60FPS_aist_rot_v2/epoch_80.pt \
               --json_dir 60FPS_aist_rot_outputs_v2/ \
               --batch_size 1 \
               --aist \
               --rotmat