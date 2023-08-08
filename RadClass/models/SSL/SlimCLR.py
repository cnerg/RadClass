import argparse
import os
import subprocess

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
# from torchlars import LARS
from tqdm import tqdm

# import sys
# import os
# sys.path.append(os.getcwd()+'/scripts/')
# sys.path.append(os.getcwd()+'/models/PyTorch/')
# sys.path.append(os.getcwd()+'/models/SSL/')

from ...scripts.configs import get_datasets
from ..PyTorch.critic import LinearCritic
from ...scripts.evaluate import save_checkpoint, encode_train_set, train_clf, test
# from models import *
from ...scripts.scheduler import CosineAnnealingWithLinearRampLR
from ..PyTorch.ann import LinearNN

from pytorch_metric_learning.losses import SelfSupervisedLoss, NTXentLoss
from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

import numpy as np
import joblib

import logging

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
    parser.add_argument('--resume', '-r', type=str, default='',
                        help='resume from checkpoint with this filename')
    parser.add_argument('--dataset', '-d', type=str, default='minos',
                        help='dataset keyword',
                        choices=['minos', 'minos-curated', 'minos-2019',
                                 'minos-2019-binary', 'cifar10', 'cifar100',
                                 'stl10', 'imagenet'])
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
    parser.add_argument("--arch", type=str, default='minos',
                        help='Encoder architecture',
                        choices=['minos', 'minos-curated', 'minos-2019',
                                 'minos-2019-binary', 'resnet18',
                                 'resnet34', 'resnet50'])
    parser.add_argument("--num-workers", type=int, default=2,
                        help='Number of threads for data loaders')
    parser.add_argument("--test-freq", type=int, default=10,
                        help='Frequency to fit a clf with L-BFGS for testing.'
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

    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(filename='debug.log',
                        filemode='a',
                        level=logging.INFO)
    args = parse_arguments()
    args.lr = args.base_lr * (args.batch_size / 256)

    args.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    args.git_diff = subprocess.check_output(['git', 'diff'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    clf = None

    # set seed(s) for reproducibility
    torch.manual_seed(20230316)
    np.random.seed(20230316)

    print('==> Preparing data..')
    num_classes = args.n_classes
    trainset, valset, testset = get_datasets(args.dataset, args.dfpath,
                                             args.bfpath, args.valfpath,
                                             args.testfpath)
    joblib.dump(trainset.mean, args.filename+'-train_means.joblib')
    joblib.dump(trainset.std, args.filename+'-train_stds.joblib')

    pin_memory = True if device == 'cuda' else False
    print(f'pin_memory={pin_memory}')

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=pin_memory)
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            pin_memory=pin_memory)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             pin_memory=pin_memory)

    # Model
    print('==> Building model..')
    ##############################################################
    # Encoder
    ##############################################################
    if args.arch in ['minos', 'minos-curated',
                     'minos-2019', 'minos-2019-binary']:
        net = LinearNN(dim=args.in_dim, mid=args.mid,
                       n_layers=args.n_layers, dropout_rate=1.,
                       n_epochs=args.num_epochs, mid_bias=True,
                       out_bias=True, n_classes=None)
    else:
        raise ValueError("Bad architecture specification")
    net = net.to(device)
    print(f'net dimensions={net.representation_dim}')

    ##############################################################
    # Critic
    ##############################################################
    # projection head to reduce dimensionality for contrastive loss
    proj_head = LinearCritic(latent_dim=args.mid[-1]).to(device)
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

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no chkpt directory found!'
        resume_from = os.path.join('./checkpoint', args.resume)
        checkpoint = torch.load(resume_from)
        net.load_state_dict(checkpoint['net'])
        critic.load_state_dict(checkpoint['critic'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    base_optimizer = optim.SGD(list(net.parameters())
                               + list(proj_head.parameters())
                               + list(critic.parameters()),
                               # + list(latent_clf.parameters())
                               lr=args.lr, weight_decay=1e-6,
                               momentum=args.momentum)
    if args.cosine_anneal:
        scheduler = CosineAnnealingWithLinearRampLR(base_optimizer,
                                                    args.num_epochs)
    # encoder_optimizer = LARS(base_optimizer, trust_coef=1e-3)
    encoder_optimizer = base_optimizer

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        # critic.train()
        critic.train()
        train_loss = 0
        t = tqdm(enumerate(trainloader), desc='Loss: **** ',
                 total=len(trainloader), bar_format='{desc}{bar}{r_bar}')
        for batch_idx, (inputs, _, _) in t:
            x1, x2 = inputs
            x1, x2 = x1.to(device), x2.to(device)
            encoder_optimizer.zero_grad()
            representation1, representation2 = net(x1), net(x2)
            # projection head for contrastive loss
            # optional: instead pass representations directly; benefit?
            representation1 = proj_head.project(representation1)
            representation2 = proj_head.project(representation2)
            # labels1 = latent_clf(representation1)
            # labels2 = latent_clf(representation2)

            # each (x1i, x2i) is a positive pair
            labels = torch.arange(representation1.shape[0])
            # sub-batching to preserve memory
            all_losses = []
            for s in range(0, args.batch_size, sub_batch_size):
                # embedding/representation subset
                curr_emb = representation1[s:s+sub_batch_size]
                curr_labels = labels[s:s+sub_batch_size]
                # apply loss across all of the second representations
                curr_loss = critic(curr_emb, curr_labels,
                                   ref_emb=representation2, ref_labels=labels)
                all_losses.append(curr_loss['loss']['losses'])
                # ignore 0 loss when sub_batch is not full
                all_losses = [loss for loss in all_losses
                              if not isinstance(loss, int)]

            # summarize loss and calculate gradient
            all_losses = torch.cat(all_losses, dim=0)
            loss = torch.mean(all_losses)
            loss.backward()
            encoder_optimizer.step()
            train_loss += loss.item()
            # free memory used by loss graph of this batch
            del loss, all_losses, curr_loss
            x1.detach(), x2.detach()

            t.set_description('Loss: %.3f ' % (train_loss / (batch_idx + 1)))
        return train_loss

    bacc_curve = np.array([])
    train_loss_curve = np.array([])
    test_loss_curve = np.array([])
    confmat_curve = np.array([])
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler,
        with_stack=True
    ):  # as profiler:
        for epoch in range(start_epoch, start_epoch + args.num_epochs):
            train_loss = train(epoch)
            train_loss_curve = np.append(train_loss_curve, train_loss)
            if (args.test_freq > 0) and (epoch % args.test_freq
                                         == (args.test_freq - 1)):
                X, y = encode_train_set(valloader, device, net)
                clf = train_clf(X, y, net.representation_dim,
                                num_classes, device, reg_weight=1e-5)
                acc, bacc, cmat, test_loss = test(testloader, device,
                                                  net, clf, num_classes)
                bacc_curve = np.append(bacc_curve, bacc)
                test_loss_curve = np.append(test_loss_curve, test_loss)
                confmat_curve = np.append(confmat_curve, cmat)
                print(f'\t-> epoch {epoch} Balanced Accuracy = {bacc}')
                print(f'\t-> with confusion matrix = {cmat}')
                if acc > best_acc:
                    best_acc = acc
                save_checkpoint(net, clf, critic, epoch,
                                args, os.path.basename(__file__))
                results = {'bacc_curve': bacc_curve,
                           'train_loss_curve': train_loss_curve,
                           'test_loss_curve': test_loss_curve,
                           'confmat_curve': confmat_curve}
                joblib.dump(results,
                            './checkpoint/'
                            + args.filename+'-result_curves.joblib')
            elif args.test_freq == 0:
                save_checkpoint(net, clf, critic, epoch,
                                args, os.path.basename(__file__))
            if args.cosine_anneal:
                scheduler.step()


if __name__ == "__main__":
    main()
