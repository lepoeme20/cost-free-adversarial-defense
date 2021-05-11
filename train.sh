#!/bin/bash

# To save adversaria images

# === Set parameters ===
dataset='cifar100' #cifar10 | cifar100
adv_training='false'
phase_tuple=('intra') # ce | inter | restricted | intra
model='110' # renet | 18 | 34 | 110: str
restrict_dist=6
gpu_id=(0 2 3) # (0 1 2 3)
data_path='/repo/data/'
lr=0.1
lr_intra=0.0001 # CIFAR10: 0.001
batch_size=512

echo "**START Training | $inter_p | $intra_p**"
echo "-------------------------------------------------------------------------"
echo "Dataset: $dataset"
echo "Adversarial Training: '$adv_training'"
echo "Train with lr $lr_pre for $epochs epochs"
echo "-------------------------------------------------------------------------"
echo ""

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
        ce_epoch=100
        epochs=1000
    fi
python main.py --batch-size $batch_size --lr $lr --lr-intra $lr_intra \
    --dataset $dataset --epochs $epochs --ce-epoch $ce_epoch --data-root-path $data_path \
    --device-ids ${gpu_id[@]} --phase $phase --restrict-dist $restrict_dist --model $model \
    --resume --resume-model 'intra_663_model_110'
done

