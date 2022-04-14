import torch
from torch.nn.modules.loss import _Loss


class MaskedCriterion(torch.nn.Module):
    """
    calculates average loss on x of 2d image while enabling the user to specify a float mask ranging from 0 ... 1
    specifying the weights of the different regions to the loss.
    Can be used to exclude the background for loss calculation.
    """

    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, x, y, mask=None):
        """

        :param x: N x C x H x W
        :param y: N x C x H x W
        :param mask: N x 1 x H x W range from 0 ... 1, if mask not given, mask = torch.ones_like(x)
                -> falls back to standard mean reduction
        :return:
        """
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        assert mask.dtype == torch.float
        assert 0 <= mask[0, 0, 0, 0] <= 1.  # exemplary check first tensor entry for range
        assert x.shape == y.shape

        mask = mask.expand_as(x)  # adding channels to mask if necessary

        mask_sum = mask.sum()
        if mask_sum > 0:
            loss = self.criterion(x, y)
            if len(loss.shape) == 3:
                loss.unsqueeze_(1)
            loss = (loss * mask).sum() / mask_sum
        else:
            loss = 0

        return loss


class LeakyHingeLoss(torch.nn.Module):
    """
    """

    def __init__(self, m_small, m_big, thr, metric=torch.nn.L1Loss(reduction="none")):
        super().__init__()
        self.metric = metric
        self.metric.reduction = "none"
        self.m_small = m_small
        self.m_big = m_big
        self.thr = thr

    def forward(self, pred, gt):
        """
        :param pred:
        :param gt:
        :return:
        """

        distance = self.metric(pred, gt)
        loss = distance[distance <= self.thr].sum() * self.m_small + \
               distance[distance > self.thr].sum() * self.m_big
        loss = loss / torch.numel(distance)
        return loss
