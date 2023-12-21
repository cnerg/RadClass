import argparse
import os
import subprocess
import glob

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import lightning.pytorch as pl

from ...scripts.configs import get_datasets
from ..PyTorch.critic import LinearCritic
from ..PyTorch.lightModel import LitSimCLR
from ...scripts.evaluate import save_checkpoint, encode_train_set, train_clf, test
# from models import *
from ...scripts.scheduler import CosineAnnealingWithLinearRampLR
from ..PyTorch.ann import LinearNN, ConvNN

from pytorch_metric_learning.losses import SelfSupervisedLoss, NTXentLoss
from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

import numpy as np
import joblib

import logging

# needed for lightning's distributed package
# os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
# torch.distributed.init_process_group("gloo")

'''
Author: Jordan Stomps

Largely adapted from a PyTorch conversion of SimCLR by Adam Foster.
More information found here: https://github.com/ae-foster/pytorch-simclr

MIT License

Copyright (c) 2023 Jordan Stomps

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

'''Train an encoder using Contrastive Learning.'''


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch'
                                                 'Contrastive Learning.')
    parser.add_argument('--base-lr', default=0.25, type=float,
                        help='base learning rate, rescaled by batch_size/256')
    parser.add_argument("--momentum", default=0.9, type=float,
                        help='SGD momentum')
    parser.add_argument('--resume', '-r', type=str, default=None,
                        help='resume from checkpoint with this filename')
    parser.add_argument('--dataset', '-d', type=str, default='minos',
                        help='dataset keyword',
                        choices=['minos', 'minos-ssml', 'minos-transfer-ssml',
                                 'minos-curated', 'minos-2019',
                                 'minos-2019-binary'])
    parser.add_argument('--dfpath', '-p', type=str,
                        help='filepath for dataset')
    parser.add_argument('--valfpath', '-v', type=str,
                        help='filepath for validation dataset')
    parser.add_argument('--testfpath', '-t', type=str,
                        help='filepath for test dataset')
    parser.add_argument('--bfpath', '-f', type=str,
                        help='filepath for background library augmentations')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='InfoNCE temperature')
    parser.add_argument("--batch-size", type=int, default=512,
                        help='Training batch size')
    parser.add_argument("--num-epochs", type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument("--cosine-anneal", action='store_true',
                        help="Use cosine annealing on the learning rate")
    parser.add_argument("--normalization", action='store_true',
                        help='Use normalization instead of'
                             'standardization in pre-processing.')
    parser.add_argument("--accounting", action='store_true',
                        help='Remove estimated background before'
                             'returning spectra in training.')
    parser.add_argument("--convolution", action="store_true",
                        help="Create a CNN rather than FCNN.")
    parser.add_argument("--arch", type=str, default='minos',
                        help='Encoder architecture',
                        choices=['minos', 'minos-ssml', 'minos-transfer-ssml',
                                 'minos-curated', 'minos-2019',
                                 'minos-2019-binary'])
    parser.add_argument("--num-workers", type=int, default=2,
                        help='Number of threads for data loaders')
    parser.add_argument("--test-freq", type=int, default=10,
                        help='Frequency to fit a clf with L-BFGS for testing'
                             'Not appropriate for large datasets.'
                             'Set 0 to avoid classifier only training here.')
    parser.add_argument("--filename", type=str, default='ckpt',
                        help='Output file name')
    parser.add_argument('--in-dim', '-i', type=int,
                        help='number of input image dimensions')
    parser.add_argument('--mid', '-m', type=int, nargs='+',
                        help='hidden layer size')
    parser.add_argument('--n-layers', '-n', type=int,
                        help='number of hidden layers')
    parser.add_argument('--n-classes', '-c', type=int, default=7,
                        help='number of classes/labels in projection head')
    parser.add_argument('--alpha', '-a', type=float, default=1.,
                        help='weight for semi-supervised contrastive loss')
    parser.add_argument('--beta1', type=float, default=0.8,
                        help='first beta used by AdamW optimizer')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='second beta used by AdamW optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-6,
                        help='weight decay hyperparameter for AdamW optimizer')
    parser.add_argument('--augs', '-u', type=str, nargs='+', default=None,
                        help='list of augmentations to be applied in SSL')

    args = parser.parse_args()
    return args


def main():
    torch.set_printoptions(profile='full')
    logging.basicConfig(filename='debug.log',
                        filemode='a',
                        level=logging.INFO)
    args = parse_arguments()
    if args.batch_size <= 1024:
        args.lr = args.base_lr * (np.sqrt(args.batch_size) / 256)
    else:
        args.lr = args.base_lr * (args.batch_size / 256)

    args.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    args.git_diff = subprocess.check_output(['git', 'diff'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # for use with a GPU
    if device == 'cuda':
        torch.set_float32_matmul_precision('medium')
    print(f'device used={device}')

    # set seed(s) for reproducibility
    torch.manual_seed(20230316)
    np.random.seed(20230316)

    print('==> Preparing data..')
    print('min-max normalization? ', args.normalization)
    num_classes = args.n_classes
    trainset, valset, testset, ssmlset = get_datasets(args.dataset,
                                                      args.dfpath,
                                                      args.bfpath,
                                                      args.valfpath,
                                                      args.testfpath,
                                                      args.normalization,
                                                      args.accounting,
                                                      args.augs)
    print(f'ssml dataset={ssmlset}')

    pin_memory = True if device == 'cuda' else False
    print(f'pin_memory={pin_memory}')

    if ssmlset is not None:
        full_trainset = torch.utils.data.ConcatDataset([trainset, ssmlset])
    else:
        full_trainset = trainset
    trainloader = torch.utils.data.DataLoader(full_trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=pin_memory)
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            # num_workers=args.num_workers,
                                            num_workers=0,
                                            pin_memory=pin_memory)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             #  num_workers=args.num_workers,
                                             num_workers=0,
                                             pin_memory=pin_memory)

    # Model
    print('==> Building model..')
    ##############################################################
    # Encoder
    ##############################################################
    if args.arch in ['minos', 'minos-ssml', 'minos-transfer-ssml',
                     'minos-curated', 'minos-2019', 'minos-2019-binary']:
        if args.convolution:
            print('-> running a convolutional NN')
            net = ConvNN(dim=args.in_dim, mid=args.mid, kernel=3,
                         n_layers=args.n_layers, dropout_rate=0.1,
                         n_epochs=args.num_epochs, out_bias=True,
                         n_classes=None)
        elif not args.convolution:
            print('-> running a fully-connected NN')
            net = LinearNN(dim=args.in_dim, mid=args.mid,
                           n_layers=args.n_layers, dropout_rate=1.,
                           n_epochs=args.num_epochs, mid_bias=True,
                           out_bias=True, n_classes=None)
    else:
        raise ValueError("Bad architecture specification")
    net = net.to(device)
    clf = nn.Linear(net.representation_dim, args.n_classes)
    print(f'net dimensions={net.representation_dim}')

    ##############################################################
    # Critic
    ##############################################################
    # projection head to reduce dimensionality for contrastive loss
    proj_head = LinearCritic(latent_dim=net.representation_dim).to(device)
    # classifier for better decision boundaries
    # latent_clf = nn.Linear(proj_head.projection_dim, num_classes).to(device)
    # NTXentLoss on its own requires labels (all unique)
    critic = NTXentLoss(temperature=0.07, reducer=reducers.DoNothingReducer())
    sub_batch_size = 64

    if device == 'cuda':
        repr_dim = net.representation_dim
        net = torch.nn.DataParallel(net)
        net.representation_dim = repr_dim
        cudnn.benchmark = True

    # if args.resume:
    #     # Load checkpoint.
    #     print('==> Resuming from checkpoint..')
    #     assert os.path.isdir('checkpoint'), \
    #         'Error: no checkpoint directory found!'
    #     resume_from = os.path.join('./checkpoint', args.resume)
    #     checkpoint = torch.load(resume_from)
    #     net.load_state_dict(checkpoint['net'])
    #     critic.load_state_dict(checkpoint['critic'])

    # make checkpoint directory
    ckpt_path = './checkpoint/'+args.filename+'/'
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    # if args.resume:
    #     # the last version run
    #     last_ver = glob.glob(ckpt_path+'lightning_logs/version_*/')[-1]
    #     ckpt = ckpt_path + last_ver + glob.glob(last_ver+'checkpoints/*.ckpt')[-1]
    # else:
    #     ckpt = None

    # save statistical data
    joblib.dump(trainset.mean, ckpt_path+args.filename+'-train_means.joblib')
    joblib.dump(trainset.std, ckpt_path+args.filename+'-train_stds.joblib')

    lightning_model = LitSimCLR(clf, net, proj_head, critic, args.batch_size,
                                sub_batch_size, args.lr, args.momentum,
                                args.cosine_anneal, args.num_epochs,
                                args.alpha, num_classes, args.test_freq,
                                testloader, args.convolution, (args.beta1, args.beta2), args.weight_decay)
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=ckpt_path)
    trainer = pl.Trainer(max_epochs=args.num_epochs,
                         default_root_dir=ckpt_path,
                         check_val_every_n_epoch=args.test_freq,
                         profiler='simple', limit_train_batches=0.002,
                         logger=tb_logger, num_sanity_val_steps=0)
    trainer.fit(model=lightning_model, train_dataloaders=trainloader,
                val_dataloaders=valloader, ckpt_path=args.resume)
    trainer.test(model=lightning_model, dataloaders=testloader)


if __name__ == "__main__":
    main()
