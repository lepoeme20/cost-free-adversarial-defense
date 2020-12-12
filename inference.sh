#!/bin/bash

# To save adversaria images

# === Set parameters ===
dataset='cifar10' #cifar10 | cifar100
attack_methods=('Clean'  'FGSM') # 'Clean' 'FGSM' 'BIM' 'PGD'
epsilon=0.03
models=('pretrained_model' 'proposed_model') #'pretrained_model' 'proposed_model'
adv_training='false'
lr=0.1
data_path='/media/lepoeme20/Data/benchmark'
trn_batch_size=(256)
tst_batch_size=256
# version=('v0' 'v1' 'v2' 'v3' 'v4')
version='v2'
device_ids=(0)
network='resnet110'

# for v in ${version[@]}
for b in ${trn_batch_size[@]}
do
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
        --data-root-path $data_path --batch-size $b \
        --lr $lr --v $version --device-ids ${device_ids[@]} \
        --test-batch-size $tst_batch_size --network $network
        done
    done
done
