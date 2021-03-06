#! /bin/bash

# Generate pose sequences of json format and visualize them
python test.py --input_dir data_aist/test_only_music \
               --model 60FPS_aist/epoch_1200.pt \
               --json_dir 60FPS_aist_outputs/ \
               --batch_size 1 \
               --aist