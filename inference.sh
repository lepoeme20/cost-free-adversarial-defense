#!/bin/bash

# === Set parameters ===
dataset='cifar100' #cifar10 | cifar100
attack_methods=('Clean' 'FGSM'  'BIM'  'PGD') # 'Clean' 'FGSM' 'BIM' 'PGD'
epsilon=(0.03)
models=('intra') # ce | inter | restricted | intra
backbone='110'
data_path='/repo/data'
tst_batch_size=1024
device_ids=(0 2 3)

for e in ${epsilon[@]}
do
    for model in ${models[@]}
    do
        for attack_method in ${attack_methods[@]}
        do
        echo ""
        echo "**START INFERENCE**"
        echo "-------------------------------------------------"
        if [ "$attack_method" != "Clean" ]; then
            echo "Performance on '$attack_method' attack with '$e' epsilon"
        else
            echo "Performance on 'clean' dataset:"
        fi
        echo "Dataset: $dataset"
        echo "Test model: $model (backbone: $backbone)"
        echo "-------------------------------------------------"
        echo ""

        python inference.py --attack-name $attack_method \
            --test-model $model --dataset $dataset --eps $e \
            --data-root-path $data_path \
            --device-ids ${device_ids[@]} \
            --model $backbone \
            --test-batch-size $tst_batch_size # --adv-train
        done
    done
done
