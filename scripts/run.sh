#! /bin/bash

# Create new tmux session named "training"
tmux new-session -d -s training

# Create window for SegResNet and run it (regular training)
tmux send-keys -t training:0 "python scripts/train.py --arch_name SegResNet --max_epochs 200 --check_val_every_n_epoch 5 --gpu_number 0 || echo 'SegResNet training failed!'" C-m

# Create window for SegResNet residual training
tmux new-window -t training:1
tmux send-keys -t training:1 "python scripts/residual_train.py --arch_name SegResNet --max_epochs 200 --check_val_every_n_epoch 5 --gpu_number 0 || echo 'SegResNet residual training failed!'" C-m

# Create new window for UNETR and run it
tmux new-window -t training:2
tmux send-keys -t training:2 "python scripts/train.py --arch_name UNETR --max_epochs 200 --check_val_every_n_epoch 5 --gpu_number 2 || echo 'UNETR training failed!'" C-m

# Create new window for SwinUNETR regular training
tmux new-window -t training:3
tmux send-keys -t training:3 "python scripts/train.py --arch_name SwinUNETR --max_epochs 200 --check_val_every_n_epoch 5 --gpu_number 1 || echo 'SwinUNETR training failed!'" C-m

# Create new window for SwinUNETR residual training
tmux new-window -t training:4
tmux send-keys -t training:4 "python scripts/residual_train.py --arch_name SwinUNETR --max_epochs 200 --check_val_every_n_epoch 5 --gpu_number 2 || echo 'SwinUNETR residual training failed!'" C-m

# Attach to the tmux session
tmux attach -t training
