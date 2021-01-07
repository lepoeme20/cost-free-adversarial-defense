"""Main module
2020.08.18
"""
import os
import torch
import random
import config
import numpy as np
from trainer import Trainer
from proposed_model_trainer import ProposedTrainer

def main():
    args = config.get_config()
    print(args)
    if args.device != 'cpu':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    if not args.proposed:
        trainer = Trainer(args)
    else:
        print("Proposed model will be trained")
        trainer = ProposedTrainer(args)
    trainer.training(args)

if __name__ == "__main__":
    main()
