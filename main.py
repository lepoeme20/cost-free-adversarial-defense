"""Main module
2020.08.18
"""
import os
import torch
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
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.device_ids)

    seed_dict = {
        'v0': 0,
        'v1': 100,
        'v2': 200,
        'v3': 300,
        'v4': 400
    }
    torch.manual_seed(seed_dict[args.v])
    np.random.seed(seed_dict[args.v])
    if not args.proposed:
        trainer = Trainer(args)
    else:
        print("Proposed model will be trained")
        trainer = ProposedTrainer(args)
        # fine_tuning(args)
    trainer.training(args)

if __name__ == "__main__":
    main()
