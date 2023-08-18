import torch
import torch.nn as nn
import torch.optim as optim
# from torchlars import LARS
# from flash.core import LARS
from tqdm import tqdm

# import sys
# import os
# sys.path.append(os.getcwd()+'/scripts/')
# sys.path.append(os.getcwd()+'/models/PyTorch/')
# sys.path.append(os.getcwd()+'/models/SSL/')

from ...scripts.configs import get_datasets
from ...scripts.evaluate import save_checkpoint, encode_train_set, train_clf, test
# from models import *
from ...scripts.scheduler import CosineAnnealingWithLinearRampLR

from pytorch_metric_learning.losses import SelfSupervisedLoss, NTXentLoss
from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

import lightning.pytorch as pl

import numpy as np
from torchmetrics import ConfusionMatrix

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


''' Image implementation from PyTorch Lightning
class SimCLR(pl.LightningModule):
    # PyTorch Lightning Implementation of SimCLR as implemented in Tutorial 13
    def __init__(self, hidden_dim, lr, temperature,
                 weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature \
                                                must be a positive float!"
        # Base model f(.)
        self.convnet = torchvision.models.resnet18(
            pretrained=False, num_classes=4 * hidden_dim
        )  # num_classes is the output size of the last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs,
            eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.convnet(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :],
                                      feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool,
                              device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            # First position positive example
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")
'''


class LitSimCLR(pl.LightningModule):
    # PyTorch Lightning Implementation of SimCLR
    # as manually implemented via A E Foster
    def __init__(self, clf, net, proj, critic, batch_size, sub_batch_size, lr,
                 momentum, cosine_anneal, num_epochs, alpha, n_classes,
                 test_freq, testloader, convolution, betas=(0.8, 0.99), weight_decay=1e-6):
        super().__init__()
        # intiialize linear classifier used in validation and testing
        self.clf = clf
        self.net = net
        self.proj = proj
        self.critic = critic
        self.batch_size = batch_size
        self.sub_batch_size = sub_batch_size
        self.lr, self.momentum, self.cosine_anneal, self.num_epochs, self.alpha, self.n_classes, self.test_freq, self.testloader = lr, momentum, cosine_anneal, num_epochs, alpha, n_classes, test_freq, testloader
        self.betas, self.weight_decay = betas, weight_decay
        self.save_hyperparameters(ignore=['critic', 'proj', 'net', 'testloader'])

        # True if net is CNN
        self.convolution = convolution

        # EMA update for projection head to boost performance (see SSL Cookbook)
        # must use additional library: https://github.com/fadel/pytorch_ema
        # self.ema = ExponentialMovingAverage(self.encoder.parameters(), decay=0.995)

    # def custom_histogram_adder(self):
    #     # iterating through all parameters
    #     for name, params in self.named_parameters():
    #         self.logger.experiment.add_histogram(name,
    #                                              params,
    #                                              self.current_epoch)

    def configure_optimizers(self):
        # base_optimizer = optim.SGD(list(self.net.parameters())
        #                            + list(self.proj.parameters()),
        #                            #    + list(self.critic.parameters()),
        #                            lr=self.lr, weight_decay=1e-6,
        #                            momentum=self.momentum)
        optimizer_kwargs = dict(lr=self.lr, betas=self.betas,
                                weight_decay=self.weight_decay)
        base_optimizer = torch.optim.AdamW(list(self.net.parameters())
                                           + list(self.critic.parameters()),
                                           **optimizer_kwargs)

        if self.cosine_anneal:
            self.scheduler = CosineAnnealingWithLinearRampLR(base_optimizer,
                                                             self.num_epochs)
        # encoder_optimizer = LARS(base_optimizer, trust_coef=1e-3)
        encoder_optimizer = base_optimizer
        return encoder_optimizer

    # see above for EMA update
    # def on_before_zero_grad(self, *args, **kwargs):
    #     self.ema.update(self.proj.parameters())

    def training_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        x1, x2 = inputs
        if self.convolution:
            x1, x2 = x1.unsqueeze(1), x2.unsqueeze(1)

        # graph logging
        # if self.current_epoch == 0:
        #     self.logger.experiment.add_graph(self.net,
        #                                      torch.randn(self.batch_size,
        #                                                  1,
        #                                                  1000))
        # if (self.test_freq > 0) and (self.current_epoch %
        #                              (self.test_freq*2) ==
        #                              ((self.test_freq*2) - 1)):
        #     self.custom_histogram_adder()

        # x1, x2 = x1.to(device), x2.to(device)
        # encoder_optimizer.zero_grad()
        representation1, representation2 = self.net(x1), self.net(x2)
        # projection head for contrastive loss
        # optional: instead pass representations directly; benefit?
        representation1 = self.proj.project(representation1)
        representation2 = self.proj.project(representation2)
        # labels1 = latent_clf(representation1)
        # labels2 = latent_clf(representation2)

        # semi-supervised: define labels for labeled data
        # each (x1i, x2i) is a positive pair
        labels = targets.detach().clone()
        # -1 for the unlabeled class in targets
        # providing a unique label for each unlabeled instance
        # self.n_classes avoids repeating supervised class labels
        labels[labels == -1] = torch.arange(self.n_classes,
                                            len(labels[labels == -1])
                                            + self.n_classes)
        # sub-batching to preserve memory
        all_losses = []
        for s in range(0, self.batch_size, self.sub_batch_size):
            # embedding/representation subset
            curr_emb = representation1[s:s+self.sub_batch_size]
            curr_labels = labels[s:s+self.sub_batch_size]
            # apply loss across all of the second representations
            curr_loss = self.critic(curr_emb, curr_labels,
                                    ref_emb=representation2,
                                    ref_labels=labels)

            # scaled (only) for supervised contrastive loss term
            # NOTE: if multiple positive samples appear, there will be one loss
            # for each positive pair (i.e. more than one loss per class).
            for c in range(self.n_classes):
                if torch.any(curr_labels == c):
                    # check for more than one positive pair
                    indices = torch.where(
                        torch.isin(curr_loss['loss']['indices'][1],
                                   torch.where(labels == c)[0]))[0]
                    # scale losses that match with indices for positive pairs
                    curr_loss['loss']['losses'][indices] *= self.alpha
            all_losses.append(curr_loss['loss']['losses'])
            # ignore 0 loss when sub_batch is not full
            all_losses = [loss for loss in all_losses
                          if not isinstance(loss, int)]

        # summarize loss and calculate gradient
        all_losses = torch.cat(all_losses, dim=0)
        loss = torch.mean(all_losses)
        self.log("train_loss", loss)
        # loss.backward()
        # encoder_optimizer.step()
        # train_loss += loss.item()
        # free memory used by loss graph of this batch
        # del loss, all_losses, curr_loss
        # x1.detach(), x2.detach()

        # return train_loss
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            reg_weight = 1e-3
            # encode validation set
            inputs, targets = batch
            if self.convolution:
                inputs = inputs.unsqueeze(1)
            representations = self.net(inputs)

            criterion = nn.CrossEntropyLoss()
            n_lbfgs_steps = 500
            # Should be reset after each epoch
            # for a completely independent evaluation
            self.clf = nn.Linear(representations[1].shape[0], 2)
            clf_optimizer = optim.LBFGS(self.clf.parameters(), lr=1e-2)
            self.clf.train()

            for i in range(n_lbfgs_steps):
                def closure():
                    clf_optimizer.zero_grad()
                    raw_scores = self.clf(representations)
                    loss = criterion(raw_scores, targets)
                    loss += reg_weight * self.clf.weight.pow(2).sum()
                    loss.backward(retain_graph=True)

                    return loss

                clf_optimizer.step(closure)

            raw_scores = self.clf(representations)
            _, predicted = raw_scores.max(1)
            correct = predicted.eq(targets).sum().item()
            loss = criterion(raw_scores, targets).item()
            total = targets.size(0)
            cmat = ConfusionMatrix(task='binary',
                                   num_classes=self.n_classes)(predicted,
                                                               targets)
            acc = 100. * correct / total
            bacc = 0.5 * ((cmat[0][0] / (cmat[0][0] + cmat[0][1]))
                          + (cmat[1][1] / (cmat[1][1] + cmat[1][0])))
            print('Loss: %.3f | Train Acc: %.3f%% ' %
                  (loss, 100. * correct / targets.shape[0]))
            self.log_dict({'val_acc': acc,
                           'val_bacc': bacc,
                           'val_tn': float(cmat[0][0]),
                           'val_fp': float(cmat[0][1]),
                           'val_fn': float(cmat[1][0]),
                           'val_tp': float(cmat[1][1]),
                           'val_loss': loss})

        # rolling test/validation
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.testloader):
                _, bacc = self.test_step(batch, batch_idx)
                print('Test BAcc: %.3f%% ' % (bacc))
        return predicted

    def test_step(self, batch, batch_idx):
        criterion = nn.CrossEntropyLoss()
        test_clf_loss = 0
        correct = 0
        total = 0
        # if n_classes > 2:
        #     confmat = ConfusionMatrix(task='multiclass',
        #                               num_classes=n_classes)
        #     cmat = torch.zeros(n_classes, n_classes)
        # else:
        confmat = ConfusionMatrix(task='binary', num_classes=self.n_classes)
        cmat = torch.zeros(self.n_classes, self.n_classes)
        inputs, targets = batch
        if self.convolution:
            inputs = inputs.unsqueeze(1)
        representation = self.net(inputs)
        # test_repr_loss = criterion(representation, targets)
        raw_scores = self.clf(representation)
        clf_loss = criterion(raw_scores, targets)

        test_clf_loss += clf_loss.item()
        _, predicted = raw_scores.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        cmat += confmat(predicted, targets)

        print('Loss: %.3f | Test Acc: %.3f%% ' %
              (test_clf_loss / (batch_idx + 1), 100. * correct / total))

        acc = 100. * correct / total
        bacc = 0.5 * ((cmat[0][0] / (cmat[0][0] + cmat[0][1]))
                      + (cmat[1][1] / (cmat[1][1] + cmat[1][0])))
        self.log_dict({'test_acc': acc, 'test_bacc': bacc,
                       'test_tn': float(cmat[0][0]),
                       'test_fp': float(cmat[0][1]),
                       'test_fn': float(cmat[1][0]),
                       'test_tp': float(cmat[1][1]),
                       'test_loss': test_clf_loss})
        return predicted, bacc

    # def validation_step(self):
    #     X, y = encode_train_set(valloader, device, net)
    #     clf = train_clf(X, y, self.net.representation_dim,
    #                     2, reg_weight=1e-5)
    #     acc, bacc, cmat, test_loss = test(testloader, device,
    #                                       net, clf, num_classes)
    #     bacc_curve = np.append(bacc_curve, bacc)
    #     test_loss_curve = np.append(test_loss_curve, test_loss)
    #     confmat_curve = np.append(confmat_curve, cmat)
    #     print(f'\t-> epoch {epoch} Balanced Accuracy = {bacc}')
    #     print(f'\t-> with confusion matrix = {cmat}')
    #     if acc > best_acc:
    #         best_acc = acc
    #     save_checkpoint(net, clf, critic, epoch,
    #                     args, os.path.basename(__file__))
    #     results = {'bacc_curve': bacc_curve,
    #                'train_loss_curve': train_loss_curve,
    #                'test_loss_curve': test_loss_curve,
    #                'confmat_curve': confmat_curve}
    #     joblib.dump(results,
    #                 './checkpoint/'+args.filename+'-result_curves.joblib')
