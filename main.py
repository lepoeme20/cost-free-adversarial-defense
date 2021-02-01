"""Main module
2020.08.18
"""
import os
import torch
import random
import config
import numpy as np
from ce_loss import Trainer as ce_trainer
from inter_model_trainer import Trainer as inter_trainer
from restricted_model_trainer import Trainer as restricted_trainer
from intra_model_trainer import Trainer as intra_trainer

def main():
    args = config.get_config()
    print(args)
    if args.device != 'cpu':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    # set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    if args.phase == 'ce':
        print("Standard model will be trained")
        trainer = ce_trainer(args)
    elif args.phase == 'inter':
        print("Inter loss model will be trained")
        trainer = inter_trainer(args)
    elif args.phase == 'restricted':
        print("Restricted loss model will be trained")
        trainer = restricted_trainer(args)
    else:
        print("Intra model will be trained")
        trainer = intra_trainer(args)
    trainer.training(args)

if __name__ == "__main__":
    main()
