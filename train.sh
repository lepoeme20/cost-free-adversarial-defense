#!/bin/bash

# To save adversaria images

# === Set parameters ===
dataset='cifar10' #cifar10 | cifar100
adv_training='false'
lr=0.1
epochs=300
batch_size=256
gpu_id=(0)
# versions=('v0' 'v1' 'v2' 'v3' 'v4')
v='v2'
network='resnet110'
data_path='/media/lepoeme20/Data/benchmark'

echo ""
echo "**START Training**"
echo "-------------------------------------------------------------------------"
echo "Dataset: $dataset"
echo "Adversarial Training: '$adv_training'"
echo "Train with lr $lr_pre for $epochs epochs"
echo "-------------------------------------------------------------------------"
echo ""

python main.py --dataset $dataset --lr $lr --epochs $epochs --data-root-path $data_path --batch-size $batch_size --v $v --device-ids ${gpu_id[@]} --network $network --data-root $data_path --proposed 
