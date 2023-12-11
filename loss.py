import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable
from scipy.sparse import block_diag
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss

class CosLoss(nn.Module):
    def __init__(self):
        super(CosLoss, self).__init__()
    def forward(self, cos_sim):
        base_block = np.ones([2,2])
        base_block_list = []
        for i in range(8):
            base_block_list.append(base_block)
        label_matrix = block_diag(tuple(base_block_list)).toarray()
        label_matrix = torch.tensor(label_matrix).cuda()
        
        cos_loss = torch.sqrt(torch.pow(cos_sim-label_matrix, 2).sum(dim=1))
        loss = (1/16)*torch.sum(cos_loss, dim=0)
        return loss

class CosLoss2(nn.Module):
    def __init__(self):
        super(CosLoss2, self).__init__()
    def forward(self, cos_sim):

        
        cos_sim = torch.clamp(cos_sim, min=0.0)
        base_block = np.ones([2,2])
        base_block_list = []
        for i in range(8):
            base_block_list.append(base_block)
        label_matrix = block_diag(tuple(base_block_list)).toarray()
        label_matrix = torch.tensor(label_matrix).cuda()
        
        cos_loss = torch.sqrt(torch.pow(cos_sim-label_matrix, 2).sum(dim=1))
        loss = (1/16)*torch.sum(cos_loss, dim=0)
        return loss

class CenterClusterLoss(nn.Module):
    def __init__(self):
        super(CenterClusterLoss, self).__init__()


    def forward(self, x_clusters, x1_rep, x2_rep):
        
        x_clusters_dual = rearrange(repeat(x_clusters.unsqueeze(dim=1), 'b 1 n->b t n', t=2), 'b t n->(b t) n')
        dist_x1 = torch.sqrt(torch.pow(x1_rep-x_clusters_dual, 2).sum(dim=1))
        dist_x2 = torch.sqrt(torch.pow(x2_rep-x_clusters_dual, 2).sum(dim=1))
        dist = torch.cat((dist_x1, dist_x2), dim=0)
        loss_1 = torch.mean(dist, dim=0)

        # Compute pairwise distance, replace by the official when merged


        dist_2 = torch.pow(x_clusters, 2).sum(dim=1, keepdim=True).expand(8, 8)
        dist_2 = dist_2 + dist_2.t()
        dist_2.addmm_(1, -2, x_clusters, x_clusters.t())
        dist_2 = dist_2.clamp(min=1e-12).sqrt()  # for numerical stability
        
     
        dist_2 = dist_2.clamp(max=1.0)
        dist_2 = dist_2.clamp(min=0.1)  
        ref_matrix = (torch.ones((8,8)) - torch.eye(8)).cuda()
        loss_2 = torch.sqrt(torch.pow(dist_2-ref_matrix, 2).sum(dim=1))
        loss_2 = torch.mean(loss_2, dim=0)

        loss = loss_1 + loss_2
        return loss

class CosineCenterClusterLoss(nn.Module):
    def __init__(self):
        super(CosineCenterClusterLoss, self).__init__()
        self.margin = 0.5
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, x_clusters, x1_rep, x2_rep):
        
        x_clusters_dual = rearrange(repeat(x_clusters.unsqueeze(dim=1), 'b 1 n->b t n', t=2), 'b t n->(b t) n')
        x_clusters_dual = x_clusters_dual.permute(1,0).unsqueeze(dim=0)
        # dist_x1 = torch.sqrt(torch.pow(x1_rep-x_clusters_dual, 2).sum(dim=1))
        # dist_x2 = torch.sqrt(torch.pow(x2_rep-x_clusters_dual, 2).sum(dim=1))

        sim_x1 = self.cosine_similarity(x1_rep.unsqueeze(dim=-1), x_clusters_dual)
        sim_x2 = self.cosine_similarity(x2_rep.unsqueeze(dim=-1), x_clusters_dual)

        sim_x = torch.cat((sim_x1, sim_x2), dim=1)
        sim_x = sim_x.clamp(max=self.margin)  # 同类之间相似度大于0.5即可 

        ref_matrix = 0.5*torch.ones(sim_x.shape).cuda()

        loss_1 = torch.mean(torch.sqrt(torch.pow(sim_x-ref_matrix, 2).sum(dim=1)))

        # Compute pairwise distance, replace by the official when merged
       
        sim_cluster = self.cosine_similarity(x_clusters.unsqueeze(dim=-1), x_clusters.permute(1,0).unsqueeze(0))
        sim_cluster = sim_cluster.clamp(min=self.margin)
        
        ref_matrix = (0.5*torch.eye(8) + 0.5*torch.ones(8,8)).cuda()

        loss_2 = torch.sqrt(torch.pow(sim_cluster-ref_matrix, 2).sum(dim=1))
        loss_2 = torch.mean(loss_2, dim=0)

        loss = loss_1 + loss_2
        return loss

class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    def __init__(self, batch_size, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.batch_size = batch_size
        self.mask = torch.eye(batch_size)
    def forward(self, input, target):
        """
        Args:
        - input: feature matrix with shape (batch_size, feat_dim)
        - target: ground truth labels with shape (num_classes)
        """
        n = self.batch_size
        input1 = input.narrow(0,0,n)
        input2 = input.narrow(0,n,n)
        
        # Compute pairwise distance, replace by the official when merged
        dist = pdist_torch(input1, input2)
        
        # For each anchor, find the hardest positive and negative
        # mask = target1.expand(n, n).eq(target1.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i,i].unsqueeze(0))
            dist_an.append(dist[i][self.mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct*2
        
class BiTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.suffix
    """
    def __init__(self, batch_size, margin=0.5):
        super(BiTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.batch_size = batch_size
        self.mask = torch.eye(batch_size)
    def forward(self, input, target):
        """
        Args:
        - input: feature matrix with shape (batch_size, feat_dim)
        - target: ground truth labels with shape (num_classes)
        """
        n = self.batch_size
        input1 = input.narrow(0,0,n)
        input2 = input.narrow(0,n,n)
        
        # Compute pairwise distance, replace by the official when merged
        dist = pdist_torch(input1, input2)
        
        # For each anchor, find the hardest positive and negative
        # mask = target1.expand(n, n).eq(target1.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i,i].unsqueeze(0))
            dist_an.append(dist[i][self.mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss1 = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct1  =  torch.ge(dist_an, dist_ap).sum().item() 
        
        # Compute pairwise distance, replace by the official when merged
        dist2 = pdist_torch(input2, input1)
        
        # For each anchor, find the hardest positive and negative
        dist_ap2, dist_an2 = [], []
        for i in range(n):
            dist_ap2.append(dist2[i,i].unsqueeze(0))
            dist_an2.append(dist2[i][self.mask[i] == 0].min().unsqueeze(0))
        dist_ap2 = torch.cat(dist_ap2)
        dist_an2 = torch.cat(dist_an2)
        
        # Compute ranking hinge loss
        y2 = torch.ones_like(dist_an2)
        # loss2 = self.ranking_loss(dist_an2, dist_ap2, y2)
        
        loss2 = torch.sum(torch.nn.functional.relu(dist_ap2 + self.margin - dist_an2))
        
        # compute accuracy
        correct2  =  torch.ge(dist_an2, dist_ap2).sum().item()
        
        loss = torch.add(loss1, loss2)
        return loss, correct1 + correct2
        
        
class BDTRLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.suffix
    """
    def __init__(self, batch_size, margin=0.5):
        super(BDTRLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.batch_size = batch_size
        self.mask = torch.eye(batch_size)
    def forward(self, inputs, targets):
        """
        Args:
        - input: feature matrix with shape (batch_size, feat_dim)
        - target: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        correct  =  torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct
        
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx