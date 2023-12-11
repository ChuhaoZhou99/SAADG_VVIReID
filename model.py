import copy
from multiprocessing import pool
from turtle import forward
from unicodedata import bidirectional
from sklearn.metrics import pair_confusion_matrix
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from resnet import resnet50, resnet18
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch_geometric.nn import GCNConv

import numpy as np

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if m.bias is not None:
            init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

def calc_mean_std(features):
    """
    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """

    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean,features_std

def adain(content_features, style_features):
    """
    Adaptive Instance Normalization
    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features


class gcn_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(gcn_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x, task="training"):
        x1 = self.base.layer1(x)  # [256, 72, 36]     
        x2 = self.base.layer2(x1)  # [512, 36, 18]   
        x3 = self.base.layer3(x2)  # [1024, 18, 9]
        x4 = self.base.layer4(x3)  # [2048, 18, 9]
        
        if task == "training":
            dis_x2 = self.same_modal_disturbance(x2)
            dis_x3 = self.base.layer3(dis_x2)
            dis_x3 = self.cross_modal_disturbance(dis_x3)
            x_dis = self.base.layer4(dis_x3) 
            return x4, x_dis
        else:
            return x4

    def same_modal_disturbance(self,x):
        B = x.size(0)
        x_v = x[:B//2]
        x_t = x[B//2:]
        noise_v = x_v[torch.randperm(x_v.size(0))]  # randomly select a turbulent frame
        noise_t = x_t[torch.randperm(x_t.size(0))]
        ##############################################
        distur_v = adain(x_v, noise_v)
        distur_t = adain(x_t, noise_t)
        distur_x = torch.cat((distur_v,distur_t),dim=0)
        return distur_x

    def cross_modal_disturbance(self,x):

        B = x.size(0)
        x_v = x[:B//2]
        x_t = x[B//2:]
        noise_v = x_v[torch.randperm(x_v.size(0))]
        noise_t = x_t[torch.randperm(x_t.size(0))]
        distur_v = adain(x_v,noise_t)
        distur_t = adain(x_t,noise_v)
        distur_x = torch.cat((distur_v,distur_t),dim=0)

        return distur_x




class embed_net(nn.Module):
    def __init__(self, class_num, drop=0.2, arch="resnet50"):
        super(embed_net, self).__init__()

        # hyper parameters
        pool_dim = 2048
        self.dropout = drop

        # feature extract
        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = gcn_resnet(arch=arch)

        # feature interact
        self.intra_gcn_conv = GCNConv(in_channels=2048, out_channels=2048)
        self.inter_gcn_conv = GCNConv(in_channels=2048, out_channels=2048)

        # classification layers
        self.l2norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.local_bottleneck = nn.BatchNorm1d(pool_dim)
        self.local_bottleneck.bias.requires_grad_(False)  # no shift
        self.local_classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.global_bottleneck = nn.BatchNorm1d(pool_dim)
        self.global_bottleneck.bias.requires_grad_(False)  # no shift
        self.global_classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.intra_gcn_bottleneck = nn.BatchNorm1d(pool_dim)
        self.intra_gcn_bottleneck.bias.requires_grad_(False)  # no shift
        self.intra_gcn_classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.inter_gcn_bottleneck = nn.BatchNorm1d(pool_dim)
        self.inter_gcn_bottleneck.bias.requires_grad_(False)  # no shift
        self.inter_gcn_classifier = nn.Linear(pool_dim, class_num, bias=False)

        # initialize
        self.local_bottleneck.apply(weights_init_kaiming)
        self.local_classifier.apply(weights_init_classifier)
        self.global_bottleneck.apply(weights_init_kaiming)
        self.global_classifier.apply(weights_init_classifier)
        self.intra_gcn_bottleneck.apply(weights_init_kaiming)
        self.intra_gcn_classifier.apply(weights_init_classifier)
        self.inter_gcn_bottleneck.apply(weights_init_kaiming)
        self.inter_gcn_classifier.apply(weights_init_classifier)


    def forward(self, x1, x2, modal=0, seq_len=6, edges=None):
        b, c, h, w = x1.size()
        t = seq_len
        x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
        x2 = x2.view(int(b * seq_len), int(c / seq_len), h, w)



        # style augmentation
        if self.training:
            # IR modality

            frame_batch = seq_len*16
            delta = torch.rand(frame_batch) + 0.5*torch.ones(frame_batch) # [0.5-1.5]
            inter_map = delta.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1).cuda()
            x2 = x2*inter_map
            
            # RGB modality
            alpha = (torch.rand(frame_batch) + 0.5*torch.ones(frame_batch)).unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
            beta = (torch.rand(frame_batch) + 0.5*torch.ones(frame_batch)).unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
            gamma = (torch.rand(frame_batch) + 0.5*torch.ones(frame_batch)).unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
            inter_map = torch.cat((alpha, beta, gamma), dim=1).cuda()
            x1 = x1*inter_map
            for i in range(x1.shape[0]):
                x1[i] = x1[i,torch.randperm(3),:,:]    

        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        if self.training:
            x, x_dis = self.base_resnet(x)
        else:
            x = self.base_resnet(x, task="testing")
   

        if self.training:
             
            intra_edges = edges[0]  # Cross-view edges
            inter_edges = edges[1]  # Cross-modal edges

            x_local = self.avgpool(x).squeeze()
            x_local_2 = rearrange(x_local, '(b t) n->b t n', t=seq_len)
          
            x_dis_local = self.avgpool(x_dis).squeeze()
            x_dis_local_2 = rearrange(x_dis_local, '(b t) n->b t n', t=seq_len)

            # cross-view and cross-modal interaction
            x_intra_gcn = self.intra_interact(x_local_2, intra_edges, t)
            x_inter_gcn = self.inter_interact(x_local, inter_edges, t)
            
            x_local_feat = self.local_bottleneck(x_local)
            x_local_logits = self.local_classifier(x_local_feat)

            x_dis_feat = self.local_bottleneck(x_dis_local)
            x_dis_logits = self.local_classifier(x_dis_feat) 

            x_intra_gcn_feat = self.intra_gcn_bottleneck(x_intra_gcn)
            x_intra_gcn_logits = self.intra_gcn_classifier(x_intra_gcn_feat)

            x_inter_gcn_feat = self.inter_gcn_bottleneck(x_inter_gcn)
            x_inter_gcn_logits = self.inter_gcn_classifier(x_inter_gcn_feat)

            p = 3.0
            x_global = (torch.mean(x_local_2**p, dim=1) + 1e-12)**(1/p)

            global_feat = self.global_bottleneck(x_global)
            logits = self.global_classifier(global_feat)
 
            defense_loss = torch.mean(torch.sqrt((x_local-x_dis_local).pow(2).sum(1)))  

            return x_global, x_local, logits, x_local_logits, x_dis_logits, x_intra_gcn_logits, x_inter_gcn_logits, defense_loss
        else:

            x_local = self.avgpool(x).squeeze()
            x_local_2 = rearrange(x_local, '(b t) n->b t n', t=seq_len)
            
            p=3.0
            x_global = (torch.mean(x_local_2**p, dim=1) + 1e-12)**(1/p)
            global_feat = self.global_bottleneck(x_global)
            return self.l2norm(global_feat)

    def intra_interact(self, x_gcn, edges, seq_len):
        x_gcn_feat_list = []

        for k in range(0, x_gcn.shape[0], 2):  # [192, 2048]
            x_gcn_tmp_1 = x_gcn[k]
            x_gcn_tmp_2 = x_gcn[k+1]
            x_gcn_tmp = torch.cat((x_gcn_tmp_1, x_gcn_tmp_2), dim=0)
            edge_weights_1 = rearrange(torch.cosine_similarity(x_gcn_tmp_1.unsqueeze(dim=-1), x_gcn_tmp_2.permute(1,0).unsqueeze(0), dim=1), 'm n ->(m n)')
            edge_weights_2 = rearrange(torch.cosine_similarity(x_gcn_tmp_2.unsqueeze(dim=-1), x_gcn_tmp_1.permute(1,0).unsqueeze(0), dim=1), 'm n ->(m n)')
            
            edge_weights = torch.cat((edge_weights_1, edge_weights_2), dim=0)            
            x_gcn_feat_tmp = self.intra_gcn_conv(x_gcn_tmp, edges, edge_weights) 

            x_gcn_feat_list.append(x_gcn_feat_tmp[0:seq_len,:])
            x_gcn_feat_list.append(x_gcn_feat_tmp[seq_len:,:])

        x_gcn_feat = rearrange(torch.stack(x_gcn_feat_list), 'b t n->(b t) n')

        return x_gcn_feat

    def inter_interact(self, x_gcn, edges, seq_len):
        x_v_gcn = x_gcn[0:seq_len*16,:]
        x_i_gcn = x_gcn[seq_len*16:,:]
        x_v_gcn_feat_list = []
        x_i_gcn_feat_list = []
        for k in range(0, x_v_gcn.shape[0], seq_len*2):  # [192, 2048]
            x_v_gcn_tmp = x_v_gcn[k:k+seq_len*2,:]
            x_i_gcn_tmp = x_i_gcn[k:k+seq_len*2,:]
            
            edge_weights_v2i = rearrange(torch.cosine_similarity(x_v_gcn_tmp.unsqueeze(dim=-1), x_i_gcn_tmp.permute(1,0).unsqueeze(0), dim=1), 'm n ->(m n)')
            edge_weights_i2v = rearrange(torch.cosine_similarity(x_i_gcn_tmp.unsqueeze(dim=-1), x_v_gcn_tmp.permute(1,0).unsqueeze(0), dim=1), 'm n ->(m n)')

            edge_weights = torch.cat((edge_weights_v2i, edge_weights_i2v))
            x_gcn_tmp = torch.cat((x_v_gcn_tmp, x_i_gcn_tmp), dim=0)
            x_gcn_feat_tmp = self.inter_gcn_conv(x_gcn_tmp, edges, edge_weights)

            x_v_gcn_feat_list.append(x_gcn_feat_tmp[0:seq_len*2,:])
            x_i_gcn_feat_list.append(x_gcn_feat_tmp[seq_len*2:,:])
        
        x_v_gcn_feat = torch.stack(x_v_gcn_feat_list)
        x_i_gcn_feat = torch.stack(x_i_gcn_feat_list)

        x_v_gcn_feat = rearrange(x_v_gcn_feat, 'b n d->(b n) d')
        x_i_gcn_feat = rearrange(x_i_gcn_feat, 'b n d->(b n) d')

        x_gcn_feat = torch.cat((x_v_gcn_feat, x_i_gcn_feat), dim=0) # [96, 18, 1024] 
        return x_gcn_feat

