import torch
from torch import nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """ use just MSE loss with UncertainLinear network """
    def forward(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # yhat, _ = out
        # print('out: {}'.format(out))
        # print('y: {}'.format(y))
        loss = F.mse_loss(out.reshape(-1, 1), y.reshape(-1, 1))
        return loss


class L1Loss(nn.Module):
    """ use just L1 loss with UncertainLinear network """
    def forward(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # yhat, _ = out
        loss = F.smooth_l1_loss(out.reshape(-1, 1), y.reshape(-1, 1))
        return loss


class LinearCritic(nn.Module):
    '''
    Largely adapted from a PyTorch conversion of SimCLR by Adam Foster.
    More information found here: https://github.com/ae-foster/pytorch-simclr
    '''

    def __init__(self, latent_dim, projection_dim=128, temperature=1.):
        super(LinearCritic, self).__init__()
        self.temperature = temperature
        self.projection_dim = projection_dim
        self.w1 = nn.Linear(latent_dim, latent_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        # self.bn1 = nn.BatchNorm1d(1)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(latent_dim, self.projection_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(self.projection_dim, affine=False)
        self.cossim = nn.CosineSimilarity(dim=-1)

    def project(self, h):
        return self.bn2(self.w2(self.relu(self.bn1(self.w1(h)))))

    def forward(self, h1, h2):
        z1, z2 = self.project(h1), self.project(h2)
        sim11 = self.cossim(z1.unsqueeze(-2),
                            z1.unsqueeze(-3)) / self.temperature
        sim22 = self.cossim(z2.unsqueeze(-2),
                            z2.unsqueeze(-3)) / self.temperature
        sim12 = self.cossim(z1.unsqueeze(-2),
                            z2.unsqueeze(-3)) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        raw_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
        targets = torch.arange(2 * d, dtype=torch.long,
                               device=raw_scores.device)
        return raw_scores, targets
