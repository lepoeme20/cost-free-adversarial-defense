"""Set configuration for the model
"""
import argparse
import multiprocessing
import torch
import torch.distributed as dist


def str2float(s):
    if '/' in s:
        s1, s2 = s.split('/')
        s = float(s1)/float(s2)
    return float(s)


def parser_setting(parser):
    """Set arguments
    """
    base_args = parser.add_argument_group('base arguments')
    base_args.add_argument(
        '--local_rank', type=int, default=-1, metavar='N', help='Local process rank.'
    )
    base_args.add_argument(
        '--save-path', type=str, default='./bestmodel',
        help='save path for best model'
    )
    base_args.add_argument(
        '--workers', type=int, default=multiprocessing.cpu_count()-1, metavar='N',
        help='dataloader threads'
    )
    base_args.add_argument(
        '--padding', type=int, default=4, help='base padding size'
        )
    base_args.add_argument(
        '--img-size', type=int, default=32, help='cropped image size'
        )
    base_args.add_argument(
        '--dataset', type=str, default='cifar100', choices=['mnist', 'fmnist', 'cifar10', 'cifar100'],
        help='Dataset name'
        )
    base_args.add_argument(
        "--data-root-path", type=str, default='/media/lepoeme20/Data/basics', help='data path'
        )
    base_args.add_argument(
        "--n_cpu", type=int, default=multiprocessing.cpu_count(),
        help="number of cpu threads to use during batch generation"
        )
    base_args.add_argument(
        "--device-ids", type=int, nargs='*', help="device id"
    )

    trn_args = parser.add_argument_group('training hyper params')
    trn_args.add_argument(
        '--adv-train', action='store_true', default=False,
        help = 'if adversarial training'
    )
    trn_args.add_argument(
        '--proposed', action='store_true', default=False,
        help = 'train with proposed loss'
    )
    trn_args.add_argument(
        '--phase', type=str, choices=['ce', 'inter', 'intra', 'restricted']
    )
    trn_args.add_argument(
        '--epochs', type=int, default=300, metavar='N',
        help='number of epochs to train (default: auto)'
        )
    trn_args.add_argument(
        '--ce-epoch', type=int, default=50
    )
    trn_args.add_argument(
        '--batch-size', type=int, default=256,
        help='input batch size for training (default: auto)'
        )
    trn_args.add_argument(
        '--test-batch-size', type=int, default=256,
        help='input batch size for testing (default: auto)'
        )
    trn_args.add_argument(
        '--seed', type=int, default=22, help='Seed for reproductibility'
    )
    trn_args.add_argument(
        '--restrict-dist', type=float, help='Distance for restricting loss'
    )

    opt_args = parser.add_argument_group('optimizer params')
    opt_args.add_argument(
        '--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: auto)'
        )
    opt_args.add_argument(
        '--lr-proposed', type=float, default=0.5, help='proposed center lr'
    )
    opt_args.add_argument(
        '--b1', type=float, default=0.5, help='momentum (default: 0.9)'
        )
    opt_args.add_argument(
        '--b2', type=float, default=0.99, help='momentum (default: 0.9)'
        )
    oth_args = parser.add_argument_group('others')
    oth_args.add_argument(
        "--sample-interval", type=int, default=1000, help="interval between image samples"
        )
    oth_args.add_argument(
        "--dev-interval", type=int, default=500, help="interval between image samples"
        )

    attack_args = parser.add_argument_group('Attack')
    # Deeply Supervised Discriminative Learning for Adversarial Defense (baseline)의
    # setting을 최대한 따를 것
    attack_args.add_argument(
        '--attack-name', type=str, default='FGSM', choices=['Clean', 'FGSM', 'BIM', 'CW', 'PGD']
    )
    attack_args.add_argument(
        '--test-model', type=str, default='pretrained_model'
    )
    attack_args.add_argument(
        '--eps', type=float, default=8/255, help="For bound eta"
    )
    # arguments for PGD
    attack_args.add_argument(
        '--pgd-iters', type=int, default=10, help="# of iteration for PGD attack"
    )
    attack_args.add_argument(
        '--pgd-alpha', type=float, help="Magnitude of perturbation"
    )
    attack_args.add_argument(
        '--pgd-random-start', action='store_true', default=False,
        help="If ture, initialize perturbation using eps"
    )
    # arguments for C&W
    attack_args.add_argument(
        '--cw-c', type=str2float, default=0.1, help="loss scaler"
    )
    attack_args.add_argument(
        '--cw-kappa', type=float, default=0, help="minimum value on clamping"
    )
    attack_args.add_argument(
        '--cw-iters', type=int, default=1000, help="# of iteration for CW grdient descent"
    )
    attack_args.add_argument(
        '--cw-lr', type=float, default=0.01, help="learning rate for CW attack"
    )
    attack_args.add_argument(
        '--cw-binary-search-steps', type=int, default=10, help="# of iteration for CW optimization"
    )
    attack_args.add_argument(
        '--cw-targeted', action='store_true', default=False, help="d"
    )
    # arguments for i-FGSM
    attack_args.add_argument(
        '--bim-step', type=int, default=10, help="Iteration for iterative FGSM"
    )

    ablation_args = parser.add_argument_group('Ablation')
    ablation_args.add_argument(
        '--lambda-intra', type=float, default=1., help="Intra loss weight"
    )
    ablation_args.add_argument(
        '--lambda-inter', type=float, default=1., help="Inter loss weight"
    )

    return parser

def get_config():
    parser = argparse.ArgumentParser(description="PyTorch Defense by distance-based model")
    default_parser = parser_setting(parser)
    args, _ = default_parser.parse_known_args()

    # args.is_master = args.local_rank == 0
    # args.device = torch.device("cuda:{}".format(args.local_rank))
    args.device = torch.device(f'cuda:{args.device_ids[0]}' if torch.cuda.is_available else 'cpu')
    # torch.distributed.init_process_group(backend="nccl")
    # torch.cuda.set_device(args.local_rank)

    # input channels
    if 'mnist' in args.dataset:
        args.rgb = 1
        # args.img_size = 28
    else:
        args.rgb = 3
        # args.img_size = 32

    # number of input classes
    # CelebA: Female/Male
    # Cifar100: A hundred classes
    # The rest: Ten classes
    args.num_class = 100 if args.dataset == 'cifar100' else 10

    return args

