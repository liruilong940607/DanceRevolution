#! /bin/bash

# Generate pose sequences of json format and visualize them
python test.py --input_dir data_aist_rot/test_only_music \
               --model 60FPS_aist_rot/epoch_1.pt \
               --json_dir 60FPS_aist_rot_outputs/ \
               --batch_size 1 \
               --aist \
               --rotmat