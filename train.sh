#!/bin/bash

# To save adversaria images

# === Set parameters ===
dataset='cifar10' #cifar10 | cifar100
adv_training='false'
epochs=300
gpu_id=(0)
# versions=('v0' 'v1' 'v2' 'v3' 'v4')
v='v2'
data_path='/media/lepoeme20/Data/benchmark'
inter_p_list=(0 1 2)
intra_p_list=(0 1 2)

for inter_p in ${inter_p_list[@]}
do
    for intra_p in ${intra_p_list[@]}
    do
echo ""
echo "**START Training | $inter_p | $intra_p**"
echo "-------------------------------------------------------------------------"
echo "Dataset: $dataset"
echo "Adversarial Training: '$adv_training'"
echo "Train with lr $lr_pre for $epochs epochs"
echo "-------------------------------------------------------------------------"
echo ""

python main.py --dataset $dataset --epochs $epochs --data-root-path $data_path --v $v --device-ids ${gpu_id[@]} --data-root $data_path --proposed --intra-p $intra_p --inter-p $inter_p

done
done

