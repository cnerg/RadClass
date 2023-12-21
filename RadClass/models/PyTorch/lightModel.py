import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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
                                           + list(self.proj.parameters())
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
        # NOTE: Contrastive learning algorithm below
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

            # scaled (only) for semi-supervised contrastive loss term
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
