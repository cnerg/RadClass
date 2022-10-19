from typing import Union, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

import sys
import os
sys.path.append(os.getcwd()+'/models/PyTorch/')
from critic import MSELoss

import torch
from torch import nn
import torch.nn.functional as F


class EarlyStopper:
    '''
    Early stopping mechanism for neural networks.
    Code adapted from user "isle_of_gods" from StackOverflow:
    https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    Use this class to break a training loop if the validation loss is low.
    Inputs:
    patience: integer; forces stop if validation loss has not improved
        for some time
    min_delta: "fudge value" for how much loss to tolerate before stopping
    '''

    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        '''
        Tests for the early stopping condition if the validation loss
        has not improved for a certain period of time (patience).
        Inputs:
        validation_loss: typically a float value for the loss function of
            a neural network training loop
        '''

        if validation_loss < self.min_validation_loss:
            # keep track of the smallest validation loss
            # if it has been beaten, restart patience
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            # keep track of whether validation loss has been decreasing
            # by a tolerable amount
            self.counter += 1
        return self.counter >= self.patience


class ConvNN(nn.Module):
    '''
    Neural Network constructor.
    Also includes method for forward pass.
    nn.Module: PyTorch object for neural networks.
    Inputs:
    layer1: int length for first layer.
    layer2: int length for second layer.
        Ideally a multiple of layer1.
    layer3: int length for third layer.
        Ideally a multiple of layer2.
    kernel: convolutional kernel size.
        NOTE: An optimal value is unclear for spectral data.
    drop_rate: float (<1.) probability for reset/dropout layer.
    length: single instance data length.
        NOTE: Assumed to be 1000 for spectral data.
    TODO: Allow hyperopt to optimize on arbitrary sized networks.
    '''

    def __init__(self, dim: int, mid: Union[int, list], kernel: int = 3,
                 n_layers: int = 1, dropout_rate: float = 1.,
                 n_epochs: int = 1000, out_bias: bool = False,
                 criterion: nn.Module = MSELoss(), n_classes: int = None):
        super().__init__()
        activation = nn.ReLU
        self.criterion = criterion
        self.p = dropout_rate
        self.n_epochs = n_epochs
        # default max_pool1d kernel set by Shadow MNIST example
        # NOTE: max_pool1d sets mp_kernel = mp_stride
        self.mp_kernel = 2
        if isinstance(mid, list) and len(mid) == 1 and n_layers > 1:
            mid = np.full(n_layers, mid[0])
        # if isinstance(mid, list) and (len(mid) != n_layers):
        if len(mid) != n_layers:
            raise ValueError('Specified layer architecture (mid)'
                             + 'should match n_layers')
        if isinstance(mid, int):
            mid = np.full(n_layers, mid)
        layers = [nn.Sequential(nn.Conv1d(1, mid[0], kernel, 1),
                                activation(),
                                nn.MaxPool1d(kernel_size=self.mp_kernel))]

        for i in range(n_layers-1):
            # max pooling after every convolution layer
            layers.append(nn.Sequential(nn.Conv1d(mid[i],
                                                  mid[i+1],
                                                  kernel, 1),
                                        activation(),
                                        nn.MaxPool1d(
                                            kernel_size=self.mp_kernel)))
        # dropout, and flatten after convolutions
        # layers.append(nn.MaxPool1d(kernel_size=self.mp_kernel))
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Flatten(1))
        self.m = nn.ModuleList(layers)
        if n_classes is not None:
            self.out = nn.Linear(mid[-1], n_classes, bias=out_bias)
        else:
            self.out = None

        # COMPUTE FLATTENED PARAMETERS FOR CNN
        # calculating the number of parameters/weights before the flattened
        # fully-connected layer:
        #   first, there are two convolution layers, so the output length is
        #   the input length (feature_vector.shape[0] - 2_layers*(kernel-1))
        #   if, in the future, more layers are desired, 2 must be adjusted
        #   next, calculate the output of the max_pool1d layer, which is
        #   round((conv_out - (kernel=stride - 1) - 1)/2 + 1)
        #   finally, multiply this by the number of channels in the last
        #   convolutional layer = layer2
        # NOTE: computation for max pooling after last convolution layer
        # conv_out = dim-n_layers*(kernel-1)
        # self.representation_dim = mid[-1]*self.pooling(conv_out)

        conv_out = dim
        for i in range(len(mid)):
            conv_out -= (kernel-1)
            conv_out = self.pooling(conv_out)
        self.representation_dim = mid[-1]*conv_out
        # self.var = nn.Linear(mid, 1, bias=out_bias)

        optimizer_kwargs = dict(lr=0.001, betas=(0.8, 0.99), weight_decay=1e-6)
        self.optimizer = torch.optim.AdamW(self.parameters(),
                                           **optimizer_kwargs)

    def pooling(self, conv_out):
        return ((conv_out - (self.mp_kernel - 1) - 1
                 )//self.mp_kernel) + 1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for m in self.m:
            # x = F.dropout(m(x), p=self.p, training=True)
            x = m(x.float())
        if self.out is not None:
            return self.out(x.float())  # , self.var(x)
        else:
            return x.float()

    # def predict(self, x:torch.Tensor, n_samp:int=25, l2:bool=True):
    def predict(self, X: torch.Tensor):
        """ return predictions of the model
            and the predicted model uncertainties """
        if isinstance(X, pd.DataFrame):
            X = torch.tensor(X.values)
        elif isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        # out = [self.forward(x) for _ in range(x.shape[0])]
        # out = [self(x) for _ in range(x.shape[0])]
        out = [self(x) for x in X]
        yhat = torch.stack([o[0] for o in out]).detach().cpu()
        # s = torch.stack([o[1] for o in out]).detach().cpu()
        # e, a = calc_uncertainty(yhat, s, l2)
        # return (torch.mean(yhat, dim=0), torch.mean(s, dim=0), e, a)
        return yhat

    def score(self, X: torch.Tensor, y: torch.Tensor):
        '''
        NOTE: REGRESSION SCORE
        '''
        if type(X) != torch.Tensor:
            X = torch.Tensor(X)
        yhat = self.predict(X)

        return r2_score(np.array(y), np.array(yhat))

    def fit(self, train_loader, valid_loader):
        self.device = torch.device('cpu')
        stopper = EarlyStopper(patience=int(0.02*self.n_epochs), min_delta=0)
        train_losses, valid_losses = [], []
        for t in range(1, self.n_epochs+1):
            # training
            # t_losses, t_ep, t_al, t_sb = [], [], [], []
            t_losses = []
            self.train()
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                out = self(x)
                loss = self.criterion(out, y)
                t_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                # if i % unc_rate == 0:
                #     _, _, ep, al, sb = get_metrics(self, x, y,
                #                                    n_samp, use_l2, eps)
                #     t_ep.append(ep); t_al.append(al); t_sb.append(sb)
            train_losses.append(t_losses)
            # t_ep_unc.append(t_ep)
            # t_al_unc.append(t_al)
            # t_sb_unc.append(t_sb)

            # validation
            # v_losses, v_ep, v_al, v_sb = [], [], [], []
            v_losses = []
            self.eval()
            with torch.no_grad():
                for i, (x, y) in enumerate(valid_loader):
                    x = x.to(self.device)
                    # loss, out, ep, al, sb = get_metrics(self, x, y,
                    #                                     n_samp, use_l2, eps)
                    out = self.predict(x)
                    loss = self.criterion(out, y.detach().cpu())
                    v_losses.append(loss.item())
                    # v_ep.append(ep); v_al.append(al); v_sb.append(sb)
                valid_losses.append(v_losses)
                # v_ep_unc.append(v_ep)
                # v_al_unc.append(v_al)
                # v_sb_unc.append(v_sb)

            if not np.all(np.isfinite(t_losses)):
                raise RuntimeError('NaN or Inf in training loss,\
                                    cannot recover. Exiting.')
            if t % 200 == 0:
                log = (f'Epoch: {t} - TL: {np.mean(t_losses):.2e},'
                       + ' VL: {np.mean(v_losses):.2e}, '
                       f'out: {out[:5]} and y: {y[:5]}')
                # f'tEU: {np.mean(t_ep):.2e}, vEU: {np.mean(v_ep):.2e}, '
                # f'tAU: {np.mean(t_al):.2e}, vAU: {np.mean(v_al):.2e}, '
                # f'tSU: {np.mean(t_sb):.2e}, vSU: {np.mean(v_sb):.2e}')
                print(log)
            # if use_scheduler: scheduler.step()
            if stopper.early_stop(np.mean(v_losses)):
                print(f'\t*-----Early stopping after {t} epochs b/c val-loss\
                        ({np.mean(v_losses)}) is not improving.')
                break

        return train_losses, valid_losses


class LinearNN(nn.Module):
    def __init__(self, dim: int, mid: Union[int, list], n_layers: int = 1,
                 dropout_rate: float = 1., n_epochs: int = 1000,
                 mid_bias: bool = True, out_bias: bool = False,
                 criterion: nn.Module = MSELoss(), n_classes: int = None):
        super().__init__()
        activation = nn.ReLU
        self.criterion = criterion
        self.p = dropout_rate
        self.n_epochs = n_epochs
        if isinstance(mid, list) and len(mid) == 1 and n_layers > 1:
            mid = np.full(n_layers, mid[0])
        # if isinstance(mid, list) and (len(mid) != n_layers):
        if len(mid) != n_layers:
            raise ValueError('Specified layer architecture (mid)'
                             + 'should match n_layers')
        if isinstance(mid, int):
            mid = np.full(n_layers, mid)
        layers = [nn.Sequential(nn.Linear(dim, mid[0], bias=mid_bias),
                                activation())]

        for i in range(n_layers-1):
            layers.append(nn.Sequential(nn.Linear(mid[i],
                                                  mid[i+1],
                                                  bias=mid_bias),
                                        activation()))
        self.m = nn.ModuleList(layers)
        if n_classes is not None:
            self.out = nn.Linear(mid[-1], n_classes, bias=out_bias)
        else:
            self.out = None
        self.representation_dim = mid[-1]
        # self.var = nn.Linear(mid, 1, bias=out_bias)

        optimizer_kwargs = dict(lr=0.001, betas=(0.8, 0.99), weight_decay=1e-6)
        self.optimizer = torch.optim.AdamW(self.parameters(),
                                           **optimizer_kwargs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for m in self.m:
            # x = F.dropout(m(x), p=self.p, training=True)
            x = m(x.float())
        if self.out is not None:
            return self.out(x.float())  # , self.var(x)
        else:
            return x.float()

    # def predict(self, x:torch.Tensor, n_samp:int=25, l2:bool=True):
    def predict(self, X: torch.Tensor):
        """ return predictions of the model
            and the predicted model uncertainties """
        if isinstance(X, pd.DataFrame):
            X = torch.tensor(X.values)
        elif isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        # out = [self.forward(x) for _ in range(x.shape[0])]
        # out = [self(x) for _ in range(x.shape[0])]
        out = [self(x) for x in X]
        yhat = torch.stack([o[0] for o in out]).detach().cpu()
        # s = torch.stack([o[1] for o in out]).detach().cpu()
        # e, a = calc_uncertainty(yhat, s, l2)
        # return (torch.mean(yhat, dim=0), torch.mean(s, dim=0), e, a)
        return yhat

    def score(self, X: torch.Tensor, y: torch.Tensor):
        '''
        NOTE: REGRESSION SCORE
        '''
        if type(X) != torch.Tensor:
            X = torch.Tensor(X)
        yhat = self.predict(X)

        return r2_score(np.array(y), np.array(yhat))

    def fit(self, train_loader, valid_loader):
        self.device = torch.device('cpu')
        stopper = EarlyStopper(patience=int(0.02*self.n_epochs), min_delta=0)
        train_losses, valid_losses = [], []
        for t in range(1, self.n_epochs+1):
            # training
            # t_losses, t_ep, t_al, t_sb = [], [], [], []
            t_losses = []
            self.train()
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                out = self(x)
                loss = self.criterion(out, y)
                t_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                # if i % unc_rate == 0:
                #     _, _, ep, al, sb = get_metrics(self, x, y,
                #                                    n_samp, use_l2, eps)
                #     t_ep.append(ep); t_al.append(al); t_sb.append(sb)
            train_losses.append(t_losses)
            # t_ep_unc.append(t_ep)
            # t_al_unc.append(t_al)
            # t_sb_unc.append(t_sb)

            # validation
            # v_losses, v_ep, v_al, v_sb = [], [], [], []
            v_losses = []
            self.eval()
            with torch.no_grad():
                for i, (x, y) in enumerate(valid_loader):
                    x = x.to(self.device)
                    # loss, out, ep, al, sb = get_metrics(self, x, y,
                    #                                     n_samp, use_l2, eps)
                    out = self.predict(x)
                    loss = self.criterion(out, y.detach().cpu())
                    v_losses.append(loss.item())
                    # v_ep.append(ep); v_al.append(al); v_sb.append(sb)
                valid_losses.append(v_losses)
                # v_ep_unc.append(v_ep)
                # v_al_unc.append(v_al)
                # v_sb_unc.append(v_sb)

            if not np.all(np.isfinite(t_losses)):
                raise RuntimeError('NaN or Inf in training loss,\
                                    cannot recover. Exiting.')
            if t % 200 == 0:
                log = (f'Epoch: {t} - TL: {np.mean(t_losses):.2e},'
                       + ' VL: {np.mean(v_losses):.2e}, '
                       f'out: {out[:5]} and y: {y[:5]}')
                # f'tEU: {np.mean(t_ep):.2e}, vEU: {np.mean(v_ep):.2e}, '
                # f'tAU: {np.mean(t_al):.2e}, vAU: {np.mean(v_al):.2e}, '
                # f'tSU: {np.mean(t_sb):.2e}, vSU: {np.mean(v_sb):.2e}')
                print(log)
            # if use_scheduler: scheduler.step()
            if stopper.early_stop(np.mean(v_losses)):
                print(f'\t*-----Early stopping after {t} epochs b/c val-loss\
                        ({np.mean(v_losses)}) is not improving.')
                break

        return train_losses, valid_losses

    ##############################
