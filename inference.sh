#!/bin/bash

# To save adversaria images

# === Set parameters ===
dataset='cifar100' #cifar10 | cifar100
attack_methods=('Clean' 'FGSM'  'BIM'  'PGD') # 'Clean' 'FGSM' 'BIM' 'PGD'
epsilon=0.03
models=('intra_202') # ce | inter | restricted | intra
backbone='110'
adv_training='false'
data_path='/repo/data'
tst_batch_size=256
device_ids=(0 1 2 3)

# for v in ${version[@]}
    for model in ${models[@]}
    do
        for attack_method in ${attack_methods[@]}
        do
        echo ""
        echo "**START INFERENCE Trn mini-batch size $b Model**"
        echo "-------------------------------------------------"
        if [ "$attack_method" != "Clean" ]; then
            echo "Performance on '$attack_method' attack with '$epsilon' epsilon"
        else
            echo "Performance on 'clean' dataset:"
        fi
        echo "Dataset: $dataset"
        echo "Test model: $model"
        echo "Adversarial Training: '$adv_training'"
        echo "-------------------------------------------------"
        echo ""

        python inference.py --attack-name $attack_method \
            --test-model $model --dataset $dataset --eps $epsilon \
            --data-root-path $data_path \
            --device-ids ${device_ids[@]} \
            --model $backbone \
            --test-batch-size $tst_batch_size # --adv-train
        done
    done
