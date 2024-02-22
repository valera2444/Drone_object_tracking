from __future__ import division, absolute_import
import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt() # for numerical stability

        # For each anchor, find the hardest positive and negative
        #torch.set_printoptions(threshold=10_000)
        #print('hard_mine_truple, 39',targets)
        
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        #mask.fill_diagonal_(2)#can't take positive the same as anchor, NO such in source repo????
        #print('hard_mine_truple, 40',mask)
        for i in range(n):#походу берет одну картину из треклета и считает лосс на них, а не на всем треклете
            
            dist_ap.append(dist[i][mask[i] == 1].max().unsqueeze(0))    #мб берет саму эту картинку
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
            
        
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        #print('triplet 54')
        #print('dist_ap',dist_ap)
        #print('dist_an',dist_an)
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)
