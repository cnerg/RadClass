import numpy as np
import torch
import logging
from torch.utils.data import Dataset
from RadClass.scripts.augs import DANSE


def remove_bckg(X):
    auger = DANSE()
    if X.ndim > 1:
        newX = torch.zeros_like(X)
        for i in range(X.shape[0]):
            newX[i] = X[i] - auger._estimate(X[i], mode='beads')
        return newX
    else:
        return X - auger._estimate(X, mode='beads')


class DataOrganizer(Dataset):
    def __init__(self, X, y, mean, std, accounting=False):
        self.data = torch.FloatTensor(X.copy())
        self.targets = torch.LongTensor(y.copy())
        # whether or not to remove background in output spectra
        self.accounting = accounting

        self.mean = mean
        self.std = std

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]

        if self.accounting:
            x = remove_bckg(x)
        # normalize all data
        x = x - self.mean
        x = torch.where(self.std == 0, x, x/self.std)

        return x, y


class MINOSBiaugment(Dataset):
    def __init__(self, X, y, transforms,
                 normalization=False, accounting=False):
        self.data = torch.FloatTensor(X.copy())
        self.targets = torch.LongTensor(y.copy())
        self.transforms = transforms
        # whether or not to remove background in output spectra
        self.accounting = accounting

        # remove background for normalization
        if self.accounting:
            print('***************************\
                   conducting accounting')
            tmp = remove_bckg(self.data)
        else:
            tmp = self.data
        self.mean = torch.mean(tmp, axis=0)
        self.std = torch.std(tmp, axis=0)
        if normalization:
            print('***************************\
                   conducting min-max normalization')
            self.mean = torch.min(tmp, axis=0)[0]
            self.std = torch.max(tmp, axis=0)[0] - self.mean

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        spec, target = self.data[index], self.targets[index]

        # if self.transforms is not None:
        aug1, aug2 = np.random.choice(self.transforms, size=2, replace=False)
        logging.debug(f'{index}: aug1={aug1} and aug2={aug2}')
        spec1 = torch.FloatTensor(aug1(spec))
        spec2 = torch.FloatTensor(aug2(spec))

        # remove background
        if self.accounting:
            spec1 = remove_bckg(spec1)
            spec2 = remove_bckg(spec2)
        # normalize all data
        spec1 = spec1 - self.mean
        spec1 = torch.where(self.std == 0., spec1, spec1/self.std)
        spec2 = spec2 - self.mean
        spec2 = torch.where(self.std == 0., spec2, spec2/self.std)

        return (spec1, spec2), target, index


class DataBiaugment(Dataset):
    def __init__(self, X, y, transforms, mean, std, accounting=False):
        self.data = torch.FloatTensor(X.copy())
        self.targets = torch.LongTensor(y.copy())
        self.transforms = transforms
        # whether or not to remove background in output spectra
        self.accounting = accounting

        self.mean = mean
        self.std = std

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        spec, target = self.data[index], self.targets[index]

        # if self.transforms is not None:
        aug1, aug2 = np.random.choice(self.transforms, size=2, replace=False)
        logging.debug(f'{index}: aug1={aug1} and aug2={aug2}')
        spec1 = torch.FloatTensor(aug1(spec))
        spec2 = torch.FloatTensor(aug2(spec))

        # remove background
        if self.accounting:
            spec1 = remove_bckg(spec1)
            spec2 = remove_bckg(spec2)
        # normalize all data
        spec1 = spec1 - self.mean
        spec1 = torch.where(self.std == 0., spec1, spec1/self.std)
        spec2 = spec2 - self.mean
        spec2 = torch.where(self.std == 0., spec2, spec2/self.std)

        return (spec1, spec2), target, index
