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

# import torchvision
# import torchvision.transforms as transforms

import sys
import os
sys.path.append(os.getcwd()+'/scripts/')
sys.path.append(os.getcwd()+'/data/')
# from augmentation import ColourDistortion
from dataset import MINOSBiaugment, DataOrganizer, DataBiaugment
from specTools import read_h_file
# from models import *
import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def add_indices(dataset_cls):
    class NewClass(dataset_cls):
        def __getitem__(self, item):
            output = super(NewClass, self).__getitem__(item)
            return (*output, item)

    return NewClass


def get_datasets(dataset, dset_fpath, bckg_fpath, valsfpath=None,
                 testfpath=None, normalization=False, accounting=False,
                 augs=None, add_indices_to_data=False):
    # , augment_clf_train=False, num_positive=None):

    ssml_dset = None
    transform_dict = {
        'Background': transforms.Background(bckg_dir=bckg_fpath, mode='beads'),
        'Resample': transforms.Resample(),
        'Sig2Bckg': transforms.Sig2Bckg(bckg_dir=bckg_fpath, mode='beads', r=(0.5, 1.5)),
        'Nuclear': transforms.Nuclear(binE=3),
        'Resolution': transforms.Resolution(multiplier=(0.5, 1.5)),
        'Mask': transforms.Mask(),
        'GainShift': transforms.GainShift()
    }
    transform_train = []
    if augs is not None:
        for key in augs:
            transform_train.append(transform_dict[key])
    else:
        transform_train = [
            transforms.Background(bckg_dir=bckg_fpath, mode='beads'),
            transforms.Resample(),
            transforms.Sig2Bckg(bckg_dir=bckg_fpath, mode='beads', r=(0.5, 1.5)),
            transforms.Nuclear(binE=3),
            transforms.Resolution(multiplier=(0.5, 1.5)),
            transforms.Mask(),
            transforms.GainShift()
        ]
    print('list of transformations:')
    for t in transform_train:
        print(f'\t{t}')

    if dataset in ['minos', 'minos-ssml']:
        data = pd.read_hdf(dset_fpath, key='data')
        # print(f'\tclasses: {np.unique(targets, return_counts=True)}')
        # print(f'\t\tshape: {targets.shape}')
        ytr = np.full(data.shape[0], -1)
        Xtr = data.to_numpy()[:, np.arange(1000)].astype(float)
        print(f'\tNOTE: double check data indexing: {data.shape}')
        val = pd.read_hdf(valsfpath, key='data')
        Xval = val.to_numpy()[:, 1+np.arange(1000)].astype(float)
        yval = val['label'].values
        # yval[yval == 1] = 0
        yval[yval != 1] = 0
        test = read_h_file(testfpath, 60, 60)
        Xtest = test.to_numpy()[:, np.arange(1000)].astype(float)
        targets = test['event'].values
        # all test values are positives
        # ytest = np.full_like(ytest, 0, dtype=np.int32)
        ytest = np.ones_like(targets, dtype=np.int32)
        # metal transfers
        ytest[targets == 'ac225'] = 0
        ytest[targets == 'activated-metals'] = 0
        ytest[targets == 'spent-fuel'] = 0
        print(f'\ttraining instances = {Xtr.shape[0]}')
        print(f'\tvalidation instances = {Xval.shape[0]}')
        print(f'\ttest instances = {Xtest.shape[0]}')

        if add_indices_to_data:
            tr_dset = add_indices(MINOSBiaugment(Xtr, ytr,
                                                 transforms=transform_train,
                                                 normalization=normalization,
                                                 accounting=accounting))
            val_dset = add_indices(DataOrganizer(Xval, yval, tr_dset.mean,
                                                 tr_dset.std,
                                                 accounting=accounting))
            if dataset == 'minos-ssml':
                ssml_dset = add_indices(DataBiaugment(Xval.copy(), yval.copy(),
                                                      transform_train,
                                                      tr_dset.mean,
                                                      tr_dset.std,
                                                      accounting=accounting))
            test_dset = add_indices(DataOrganizer(Xtest, ytest, tr_dset.mean,
                                                  tr_dset.std,
                                                  accounting=accounting))
        else:
            tr_dset = MINOSBiaugment(Xtr, ytr, transforms=transform_train,
                                     normalization=normalization,
                                     accounting=accounting)
            val_dset = DataOrganizer(Xval, yval, tr_dset.mean, tr_dset.std,
                                     accounting=accounting)
            if dataset == 'minos-ssml':
                ssml_dset = DataBiaugment(Xval, yval, transform_train,
                                        tr_dset.mean, tr_dset.std,
                                        accounting=accounting)
            test_dset = DataOrganizer(Xtest, ytest, tr_dset.mean,
                                      tr_dset.std, accounting=accounting)
    elif dataset in ['minos-curated', 'minos-transfer-ssml']:
        data = pd.read_hdf(dset_fpath, key='data')
        # print(f'\tclasses: {np.unique(targets, return_counts=True)}')
        # print(f'\t\tshape: {targets.shape}')
        ytr = np.full(data.shape[0], -1)
        Xtr = data.to_numpy()[:, np.arange(1000)].astype(float)
        print(f'\tNOTE: double check data indexing: {data.shape}')

        test_data = read_h_file(testfpath, 60, 60)
        X = test_data.to_numpy()[:, np.arange(1000)].astype(float)
        y = test_data['event'].values
        Xval, Xtest, \
            val_targets, test_targets = train_test_split(X, y,
                                                         train_size=0.03,
                                                         stratify=y)
        # all test values are positives
        # ytest = np.full_like(ytest, 0, dtype=np.int32)
        yval = np.ones_like(val_targets, dtype=np.int32)
        ytest = np.ones_like(test_targets, dtype=np.int32)
        # metal transfers
        yval[val_targets == 'ac225'] = 0
        yval[val_targets == 'activated-metals'] = 0
        yval[val_targets == 'spent-fuel'] = 0
        ytest[test_targets == 'ac225'] = 0
        ytest[test_targets == 'activated-metals'] = 0
        ytest[test_targets == 'spent-fuel'] = 0

        print(f'\ttraining instances = {Xtr.shape[0]}')
        print(f'\tvalidation instances = {Xval.shape[0]}')
        print(f'\ttest instances = {Xtest.shape[0]}')

        if add_indices_to_data:
            tr_dset = add_indices(MINOSBiaugment(Xtr, ytr,
                                                 transforms=transform_train,
                                                 normalization=normalization,
                                                 accounting=accounting))
            val_dset = add_indices(DataOrganizer(Xval, yval, tr_dset.mean,
                                                 tr_dset.std,
                                                 accounting=accounting))
            if dataset == 'minos-transfer-ssml':
                ssml_dset = add_indices(DataBiaugment(Xval.copy(), yval.copy(),
                                                      transform_train,
                                                      tr_dset.mean,
                                                      tr_dset.std,
                                                      accounting=accounting))
            test_dset = add_indices(DataOrganizer(Xtest, ytest, tr_dset.mean,
                                                  tr_dset.std,
                                                  accounting=accounting))
        else:
            tr_dset = MINOSBiaugment(Xtr, ytr, transforms=transform_train,
                                     normalization=normalization,
                                     accounting=accounting)
            val_dset = DataOrganizer(Xval, yval, tr_dset.mean, tr_dset.std,
                                     accounting=accounting)
            if dataset == 'minos-transfer-ssml':
                ssml_dset = DataBiaugment(Xval, yval, transform_train,
                                        tr_dset.mean, tr_dset.std,
                                        accounting=accounting)
            test_dset = DataOrganizer(Xtest, ytest, tr_dset.mean, tr_dset.std,
                                      accounting=accounting)
    elif dataset == 'minos-2019':
        # Including unlabeled spectral data for contrastive learning
        data = pd.read_hdf(dset_fpath, key='data')
        # print(f'\tclasses: {np.unique(targets, return_counts=True)}')
        # print(f'\t\tshape: {targets.shape}')
        ytr = np.full(data.shape[0], -1)
        Xtr = data.to_numpy()[:, np.arange(1000)].astype(float)
        print(f'\tNOTE: double check data indexing: {data.shape}')

        X = pd.read_hdf(valsfpath, key='data')
        # events = np.unique(X['label'].values)
        y = X['label'].values
        y[y == 1] = 0
        y[y != 0] = 1
        X = X.to_numpy()[:, 1+np.arange(1000)].astype(float)
        run = True
        while run:
            Xval, Xtest, yval, ytest = train_test_split(X, y, test_size=213)
            if np.unique(ytest, return_counts=True)[1][0] == 125:
                run = False
        print(f'\ttraining instances = {Xtr.shape[0]}')
        print(f'\tvalidation instances = {Xval.shape[0]}')
        print(f'\ttest instances = {Xtest.shape[0]}')

        if add_indices_to_data:
            tr_dset = add_indices(MINOSBiaugment(Xtr, ytr,
                                                 transforms=transform_train,
                                                 normalization=normalization,
                                                 accounting=accounting))
            val_dset = add_indices(DataOrganizer(Xval, yval, tr_dset.mean,
                                                 tr_dset.std,
                                                 accounting=accounting))
            test_dset = add_indices(DataOrganizer(Xtest, ytest, tr_dset.mean,
                                                  tr_dset.std,
                                                  accounting=accounting))
        else:
            tr_dset = MINOSBiaugment(Xtr, ytr, transforms=transform_train,
                                     normalization=normalization,
                                     accounting=accounting)
            val_dset = DataOrganizer(Xval, yval, tr_dset.mean, tr_dset.std,
                                     accounting=accounting)
            test_dset = DataOrganizer(Xtest, ytest, tr_dset.mean, tr_dset.std,
                                      accounting=accounting)
    elif dataset == 'minos-2019-binary':
        # Using only the data that was used for the preliminary experiment
        data = pd.read_hdf(dset_fpath, key='data')
        targets = data['label'].values
        targets[targets == 1] = 0
        targets[targets != 0] = 1
        print(f'\tclasses: {np.unique(targets, return_counts=True)}')
        print(f'\t\tshape: {targets.shape}')
        data = data.to_numpy()[:, 1+np.arange(1000)].astype(float)
        print(f'\tNOTE: double check data indexing: {data.shape}')
        Xtr, X, ytr, y = train_test_split(data, targets, test_size=0.3)
        Xval, Xtest, yval, ytest = train_test_split(X, y, train_size=0.33)
        print(f'\ttraining instances = {Xtr.shape[0]}')
        print(f'\tvalidation instances = {Xval.shape[0]}')
        print(f'\ttest instances = {Xtest.shape[0]}')

        if add_indices_to_data:
            tr_dset = add_indices(MINOSBiaugment(np.append(Xtr, Xval, axis=0),
                                                 np.append(ytr, yval, axis=0),
                                                 transforms=transform_train,
                                                 normalization=normalization,
                                                 accounting=accounting))
            val_dset = add_indices(DataOrganizer(Xval, yval, tr_dset.mean,
                                                 tr_dset.std,
                                                 accounting=accounting))
            test_dset = add_indices(DataOrganizer(Xtest, ytest, tr_dset.mean,
                                                  tr_dset.std,
                                                  accounting=accounting))
        else:
            tr_dset = MINOSBiaugment(Xtr, ytr, transforms=transform_train,
                                     normalization=normalization,
                                     accounting=accounting)
            val_dset = DataOrganizer(Xval, yval, tr_dset.mean, tr_dset.std,
                                     accounting=accounting)
            test_dset = DataOrganizer(Xtest, ytest, tr_dset.mean, tr_dset.std,
                                      accounting=accounting)
    else:
        raise ValueError("Bad dataset value: {}".format(dataset))

    return tr_dset, val_dset, test_dset, ssml_dset
