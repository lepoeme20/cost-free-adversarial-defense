#!/bin/bash

# To save adversaria images

# === Set parameters ===
dataset='svhn' #cifar10 | cifar100 | svhn
adv_training='false'
phase_tuple=('restricted'  'intra') # ce | inter | restricted | intra
model='110' # renet | 18 | 34 | 110: str
restrict_dist=6
gpu_id=(0 1 2 3) # (0 1 2 3)
data_path='/repo/data/'
lr=0.1
lr_intra=0.001 # CIFAR10: 0.001
batch_size=512


for phase in ${phase_tuple[@]}
do
    if [ $phase = 'ce' ]
    then
        ce_epoch=200
        epochs=0
    elif [ $phase = 'restricted' ]
    then
        ce_epoch=50
        epochs=100
    else
        ce_epoch=0
        epochs=1000
    fi
    echo "**START Training**"
    echo "-------------------------------------------------------------------------"
    echo "Dataset: $dataset"
    echo "Test model: $phase (backbone: $model)"
    echo "Adversarial Training: '$adv_training'"
    echo "-------------------------------------------------------------------------"
    echo ""
python main.py --batch-size $batch_size --lr $lr --lr-intra $lr_intra \
    --dataset $dataset --epochs $epochs --ce-epoch $ce_epoch --data-root-path $data_path \
    --device-ids ${gpu_id[@]} --phase $phase --restrict-dist $restrict_dist --model $model
    # --resume --resume-model 'intra_663_model_110'
done

