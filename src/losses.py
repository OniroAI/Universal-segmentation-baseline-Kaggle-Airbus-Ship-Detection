import torch
from torch import nn


# Source https://github.com/NikolasEnt/Lyft-Perception-Challenge/blob/master/loss.py
def fb_loss(preds, trues, beta):
    smooth = 1e-4
    beta2 = beta*beta
    batch = preds.size(0)
    classes = preds.size(1)
    preds = preds.view(batch, classes, -1)
    trues = trues.view(batch, classes, -1)
    weights = torch.clamp(trues.sum(-1), 0., 1.)
    TP = (preds * trues).sum(2)
    FP = (preds * (1-trues)).sum(2)
    FN = ((1-preds) * trues).sum(2)
    Fb = ((1+beta2) * TP + smooth)/((1+beta2) * TP + beta2 * FN + FP + smooth)
    Fb = Fb * weights
    score = Fb.sum() / (weights.sum() + smooth)
    return torch.clamp(score, 0., 1.)


class FBLoss(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, output, target):
        return 1 - fb_loss(output, target, self.beta)


class ShipLoss(nn.Module):
    def __init__(self, fb_weight=0.25, fb_beta=1, bce_weight=0.25,
                 prob_weight=0.25, mse_weight=0.25):
        super().__init__()
        self.fb_weight = fb_weight
        self.bce_weight = bce_weight
        self.prob_weight = prob_weight
        self.mse_weight = mse_weight

        self.fb_loss = FBLoss(beta=fb_beta)
        self.bce_loss = nn.BCELoss()
        self.prob_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, targets):
        pred, prob_pred = output
        segm = pred[:, :3, :, :]
        target = targets[:, :3, :, :]

        if self.fb_weight > 0:
            fb = self.fb_loss(segm, target) * self.fb_weight
        else:
            fb = 0

        if self.bce_weight > 0:
            bce = self.bce_loss(segm, target) * self.bce_weight
        else:
            bce = 0

        prob_trg = target[:, 0, : , :].unsqueeze(1)\
                   .view(target.size(0), -1).sum(dim=1) > 0
        prob_trg = prob_trg.to(torch.float32)
        if self.prob_weight > 0:
            prob = self.prob_loss(prob_pred, prob_trg) * self.prob_weight
        else:
            prob = 0

        if self.mse_weight > 0:
            pred = pred[:, 3:, :, :]
            target = targets[:, 3:, :, :]
            mse = self.mse_loss(pred, target) * self.mse_weight
        else:
            mse = 0

        return fb + bce + prob + mse
