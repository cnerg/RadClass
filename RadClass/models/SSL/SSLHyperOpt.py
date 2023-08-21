import argparse
import os
import subprocess
import glob
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import lightning.pytorch as pl
# from torchlars import LARS

# import sys
# import os
# sys.path.append(os.getcwd()+'/scripts/')
# sys.path.append(os.getcwd()+'/models/PyTorch/')
# sys.path.append(os.getcwd()+'/models/SSL/')

from ...scripts.utils import run_hyperopt
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

from ray import put, tune
from ray.air import session
# hyperopt
from hyperopt.pyll.base import scope
from hyperopt import hp
from hyperopt import STATUS_OK

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
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='filename to checkpoint for resuming raytune')
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
    parser.add_argument("--njobs", type=int, default=5,
                        help='Number of raytune parallel jobs')
    parser.add_argument("--max-evals", type=int, default=50,
                        help='Number of raytune iterations')
    parser.add_argument("--batches", type=float, default=0.75,
                        help='Maximum number or percent of batches per epoch.')
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
    parser.add_argument('--augs', '-u', type=str, nargs='+', default=None,
                        help='list of augmentations to be applied in SSL')

    args = parser.parse_args()
    return args


def architecture(config):
    if config['convolution']:
        return np.array([np.random.choice([8, 16, 32, 64, 128]) for i in range(config['n_layers'])])
    else:
        return np.array([np.random.choice([512, 1024, 2048, 4096]) for i in range(config['n_layers'])])


def fresh_start(params, data, testset):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    # for use with a GPU
    # if device == 'cuda':
    #     torch.set_float32_matmul_precision('medium')
    # print(f'device used={device}')
    pin_memory = True if device == 'cuda' else False
    print(f'pin_memory={pin_memory}')

    params['mid'] = architecture(params)

    # Model
    print('==> Building model..')
    ##############################################################
    # Encoder
    ##############################################################
    if params['convolution']:
        print('-> running a convolutional NN')
        net = ConvNN(dim=params['in_dim'], mid=params['mid'], kernel=3,
                     n_layers=params['n_layers'], dropout_rate=0.1,
                     n_epochs=params['num_epochs'], out_bias=True,
                     n_classes=None)
    elif not params['convolution']:
        print('-> running a fully-connected NN')
        net = LinearNN(dim=params['in_dim'], mid=params['mid'],
                       n_layers=params['n_layers'], dropout_rate=1.,
                       n_epochs=params['num_epochs'], mid_bias=True,
                       out_bias=True, n_classes=None)
    net = net.to(device)
    clf = nn.Linear(net.representation_dim, params['num_classes'])
    print(f'net dimensions={net.representation_dim}')

    ##############################################################
    # Critic
    ##############################################################
    # projection head to reduce dimensionality for contrastive loss
    proj_head = LinearCritic(latent_dim=net.representation_dim).to(device)
    # classifier for better decision boundaries
    # latent_clf = nn.Linear(proj_head.projection_dim, num_classes).to(device)
    # NTXentLoss on its own requires labels (all unique)
    critic = NTXentLoss(temperature=params['temperature'],
                        reducer=reducers.DoNothingReducer())
    sub_batch_size = 64

    # if device == 'cuda':
    #     repr_dim = net.representation_dim
    #     net = torch.nn.DataParallel(net)
    #     net.representation_dim = repr_dim
    #     cudnn.benchmark = True

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
    # ckpt_path = './checkpoint/'+args.filename+'/'
    # if not os.path.isdir(ckpt_path):
    #     os.mkdir(ckpt_path)

    # if args.resume:
    #     # the last version run
    #     last_ver = glob.glob(ckpt_path+'lightning_logs/version_*/')[-1]
    #     ckpt = ckpt_path + last_ver + glob.glob(last_ver+'checkpoints/*.ckpt')[-1]
    # else:
    #     ckpt = None

    # save statistical data
    # joblib.dump(trainset.mean, ckpt_path+args.filename+'-train_means.joblib')
    # joblib.dump(trainset.std, ckpt_path+args.filename+'-train_stds.joblib')

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=len(testset),
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=data.pin_memory)

    if params['batch_size'] <= 1024:
        lr = params['lr'] * (np.sqrt(params['batch_size']) / 256)
    else:
        lr = params['lr'] * (params['batch_size'] / 256)

    lightning_model = LitSimCLR(clf, net, proj_head, critic,
                                params['batch_size'],
                                sub_batch_size, lr, params['momentum'],
                                params['cosine_anneal'], params['num_epochs'],
                                params['alpha'], params['num_classes'],
                                params['test_freq'], testloader,
                                params['convolution'],
                                (params['beta1'], params['beta2']),
                                params['weight_decay'])
    # tb_logger = pl.loggers.TensorBoardLogger(save_dir=ckpt_path)
    trainer = pl.Trainer(max_epochs=params['num_epochs'],
                         #  default_root_dir=ckpt_path,
                         check_val_every_n_epoch=params['test_freq'],
                         #  profiler='simple',
                         limit_train_batches=params['batches'],
                         num_sanity_val_steps=0,
                         enable_checkpointing=False,
                         accelerator='cpu')
    trainer.fit(model=lightning_model, datamodule=data)
                # val_dataloaders=valloader)  # , ckpt_path=args.resume)
    loss = trainer.callback_metrics['train_loss']
    trainer.test(model=lightning_model,
                 datamodule=data)
                #  dataloaders=testloader)
    accuracy = trainer.callback_metrics['test_bacc']

    # loss function minimizes misclassification
    # by maximizing metrics
    results = {
        # 'score': acc+(self.alpha*rec)+(self.beta*prec),
        # 'loss': lightning_model.log['train_loss'][-1],
        'loss': loss.item(),
        'model': lightning_model,
        'status': STATUS_OK,
        'params': params,
        'accuracy': accuracy.item(),
        # 'precision': prec,
        # 'recall': rec
    }

    session.report(results)
    return results


class RadDataModule(pl.LightningDataModule):
    def __init__(self, trainset, valset, testset, batch_size=512,
                 num_workers=0, pin_memory=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.trainset = trainset
        self.valset = valset
        self.testset = testset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valset,
                                           # only one batch for validation
                                           batch_size=len(self.valset),
                                           shuffle=False,
                                           num_workers=0,
                                           pin_memory=self.pin_memory)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.testset,
                                           # only one batch for testing
                                           batch_size=len(self.testset),
                                           shuffle=False,
                                           num_workers=0,
                                           pin_memory=self.pin_memory)


def main():
    torch.set_printoptions(profile='full')
    # eval('setattr(torch.backends.cudnn, "benchmark", True)')
    logging.basicConfig(filename='debug.log',
                        filemode='a',
                        level=logging.INFO)
    args = parse_arguments()

    # args.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    # args.git_diff = subprocess.check_output(['git', 'diff'])

    # set seed(s) for reproducibility
    torch.manual_seed(20230316)
    np.random.seed(20230316)

    print('==> Preparing data..')
    # print('min-max normalization? ', args.normalization)
    trainset, valset, testset, ssmlset = get_datasets(args.dataset,
                                                      args.dfpath,
                                                      args.bfpath,
                                                      args.valfpath,
                                                      args.testfpath,
                                                      args.normalization,
                                                      args.accounting,
                                                      args.augs)
    print(f'ssml dataset={ssmlset}')

    if ssmlset is not None:
        full_trainset = torch.utils.data.ConcatDataset([trainset, ssmlset])
    else:
        full_trainset = trainset

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    # for use with a GPU
    # if device == 'cuda':
    #     torch.set_float32_matmul_precision('medium')
    # print(f'device used={device}')
    pin_memory = True if device == 'cuda' else False
    print(f'pin_memory={pin_memory}')

    dataset = RadDataModule(full_trainset, valset, testset, args.batch_size,
                            args.num_workers, pin_memory)

    space = {
        'lr': hp.uniform('lr', 1e-5, 0.5),
        'n_layers': scope.int(hp.uniformint('n_layers', 1, 7)),
        'convolution': hp.choice('convolution', [0, 1]),
        # 'mid': tune.sample_from(architecture),
        'temperature': hp.uniform('temperature', 0.1, 0.9),
        'momentum': hp.uniform('momentum', 0.5, 0.99),
        'beta1': hp.uniform('beta1', 0.7, 0.99),
        'beta2': hp.uniform('beta2', 0.8, 0.999),
        'weight_decay': hp.uniform('weight_decay', 1e-7, 1e-2),
        'batch_size': tune.choice([128, 256, 512, 1024, 2048, 4096]),#, 8192]),
        # 'batch_size': args.batch_size,
        'batches': args.batches,
        'cosine_anneal': True,
        'alpha': 1.,
        'num_classes': 2,
        'num_epochs': args.num_epochs,
        'test_freq': args.test_freq,
        'in_dim': 1000
    }

    # if args.checkpoint is not None:
    #     checkpoint = joblib.load(args.checkpoint)
    #     space['start_from_checkpoint']: put(checkpoint)

    best, worst, trials = run_hyperopt(space, fresh_start, dataset, testset,
                                       max_evals=args.max_evals,
                                       num_workers=args.num_workers,
                                       njobs=args.njobs,
                                       verbose=True)
    joblib.dump(best, 'best_model.joblib')
    joblib.dump(trials, 'trials.joblib')


if __name__ == "__main__":
    main()
