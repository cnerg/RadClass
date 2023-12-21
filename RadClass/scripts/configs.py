'''
Author: Jordan Stomps

Largely adapted from a PyTorch conversion of SimCLR by Adam Foster.
More information found here: https://github.com/ae-foster/pytorch-simclr
'''

from .dataset import MINOSBiaugment, DataOrganizer, DataBiaugment
from .specTools import read_h_file
from .transforms import Background, Resample, Sig2Bckg, Nuclear, \
    Resolution, Mask, GainShift
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

    ssml_dset = None
    transform_dict = {
        'Background': Background(bckg_dir=bckg_fpath, mode='beads'),
        'Resample': Resample(),
        'Sig2Bckg': Sig2Bckg(bckg_dir=bckg_fpath, mode='beads', r=(0.5, 1.5)),
        'Nuclear': Nuclear(binE=3),
        'Resolution': Resolution(multiplier=(0.5, 1.5)),
        'Mask': Mask(),
        'GainShift': GainShift()
    }
    transform_train = []
    if augs is not None:
        for key in augs:
            transform_train.append(transform_dict[key])
    else:
        transform_train = [
            Background(bckg_dir=bckg_fpath, mode='beads'),
            Resample(),
            Sig2Bckg(bckg_dir=bckg_fpath, mode='beads', r=(0.5, 1.5)),
            Nuclear(binE=3),
            Resolution(multiplier=(0.5, 1.5)),
            Mask(),
            GainShift()
        ]
    print('list of transformations:')
    for t in transform_train:
        print(f'\t{t}')

    if dataset in ['minos', 'minos-ssml']:
        # Using anomalous data for training, labeled data from noisy heuristic
        # for validation, and labeled data for testing
        data = pd.read_hdf(dset_fpath, key='data')
        ytr = np.full(data.shape[0], -1)
        Xtr = data.to_numpy()[:, np.arange(1000)].astype(float)
        print(f'\tNOTE: double check data indexing: {data.shape}')
        val = pd.read_hdf(valsfpath, key='data')
        Xval = val.to_numpy()[:, 1+np.arange(1000)].astype(float)
        yval = val['label'].values
        yval[yval != 1] = 0
        test = read_h_file(testfpath, 60, 60)
        Xtest = test.to_numpy()[:, np.arange(1000)].astype(float)
        targets = test['event'].values
        # all test values are positives
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
        # Using weakly anomalous data for contrastive training and labeled data
        # for training and testing classifier
        data = pd.read_hdf(dset_fpath, key='data')
        ytr = np.full(data.shape[0], -1)
        Xtr = data.to_numpy()[:, np.arange(1000)].astype(float)
        print(f'\tNOTE: double check data indexing: {data.shape}')

        test_data = read_h_file(testfpath, 60, 60)
        X = test_data.to_numpy()[:, np.arange(1000)].astype(float)
        y = test_data['event'].values
        Xval, Xtest, \
            val_targets, test_targets = train_test_split(X, y,
                                                         train_size=0.2,
                                                         stratify=y)
        # all test values are positives
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
        ytr = np.full(data.shape[0], -1)
        Xtr = data.to_numpy()[:, np.arange(1000)].astype(float)
        print(f'\tNOTE: double check data indexing: {data.shape}')

        X = pd.read_hdf(valsfpath, key='data')
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
