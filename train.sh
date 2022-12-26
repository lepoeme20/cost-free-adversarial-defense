#!/bin/bash
# === Set parameters ===
dataset='cifar10' #cifar10 | cifar100 | svhn
phase_tuple=('radial' 'shrinkage') # ce | inter | restricted | intra
model='110' # renet | 18 | 34 | 110: str
gpu_id=(0 1 2 3) 
data_path='/repo/data/'
lr=0.1
lr_shrinkage=0.0005 # CIFAR10: 0.0001
batch_size=512


for phase in ${phase_tuple[@]}
do
    if [ $phase = 'ce' ]
    then
        ce_epoch=50
        epochs=0
    elif [ $phase = 'restricted' ]
    then
        ce_epoch=50
        epochs=100
    else
        ce_epoch=0
        epochs=100
    fi
    echo "**START Training**"
    echo "-------------------------------------------------------------------------"
    echo "Dataset: $dataset"
    echo "Test model: $phase (backbone: $model)"
    echo "-------------------------------------------------------------------------"
    echo ""
    start=$(date +%s.%N)
    python main.py --batch-size $batch_size --lr $lr --lr-shrinkage $lr_shrinkage \
    --dataset $dataset --epochs $epochs --ce-epoch $ce_epoch --data-root-path $data_path \
    --device-ids ${gpu_id[@]} --phase $phase --model $model  
    echo "-------------------------------------------------------------------------"
    dur=$(echo "$(date +%s.%N) - $start" | bc)
    execution_time=`printf "%.2f seconds" $dur`
    echo "$phase execution Time: $execution_time"
    echo "-------------------------------------------------------------------------"
    echo ""
done

