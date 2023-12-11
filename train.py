from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
#import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from eval_metrics import eval_sysu, eval_regdb, evaluate

from utils import *
from loss import OriTripletLoss
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import math

from data_manager import VCM
from data_loader import VideoDataset_train, VideoDataset_test, VideoDataset_train_evaluation
import transforms as T

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='VCM', help='dataset name: VCM(Video Cross-modal)')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet50')
parser.add_argument('--exp_name', default='test', type=str, help='exp name')
parser.add_argument('--gml', default=False, type=bool, help='global mutual learning')
parser.add_argument('--lml', default=False, type=bool, help='local mutual learning')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='/data/zhouchuhao/ReID/save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log_vcm/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=512, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=32, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--part', default=3, type=int,
                    metavar='tb', help=' part number')
parser.add_argument('--method', default='id+tri', type=str,
                    metavar='m', help='method type')
parser.add_argument('--drop', default=0.2, type=float,
                    metavar='drop', help='dropout ratio')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=2, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='1', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--lambda0', default=1.0, type=float,
                    metavar='lambda0', help='graph attention weights')
parser.add_argument('--graph', action='store_true', help='either add graph attention or not')
parser.add_argument('--wpa', action='store_true', help='either add weighted part attention')
parser.add_argument('--a', default=1, type=float,
                    metavar='lambda1', help='dropout ratio')


args = parser.parse_args()
os.environ['CUDA_DEVICE_ORDER'] ='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from model import embed_net  # put it after gpu setting to correctly assign GPUs (May be a bug in torch_geometric)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = True
dataset = args.dataset

seq_lenth = 6
test_batch = 32
data_set = VCM()
current_time = time.localtime()
cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
test_mode = [1, 2]
height = args.img_h
width = args.img_w
global_mutual_learning = args.gml
local_mutual_learning = args.lml
warm_up_epochs = 0

intra_edges = []    # cross-view edges
inter_edges = []    # cross-modal edges

for i in range(0,seq_lenth):
    for j in range(seq_lenth,seq_lenth*2):
        intra_edges.append(torch.LongTensor([i,j]))

for i in range(seq_lenth,seq_lenth*2):
    for j in range(0,seq_lenth):
        intra_edges.append(torch.LongTensor([i,j]))

for i in range(0,seq_lenth*2):
    for j in range(seq_lenth*2,seq_lenth*4):
        inter_edges.append(torch.LongTensor([i,j]))
for i in range(seq_lenth*2, seq_lenth*4):
    for j in range(0,seq_lenth*2):
        inter_edges.append(torch.LongTensor([i,j]))

intra_edges = torch.stack(intra_edges).permute(1,0).cuda()
inter_edges = torch.stack(inter_edges).permute(1,0).cuda()

edges = (intra_edges, inter_edges)

if global_mutual_learning and local_mutual_learning:
    log_path = args.log_path + cur_time + '_G_L_VCM_log_' + args.exp_name + '/'
elif global_mutual_learning:
    log_path = args.log_path + cur_time + '_G_VCM_log_' + args.exp_name + '/'
else:
    log_path = args.log_path + cur_time + '_base_VCM_log_' + args.exp_name + '/'

checkpoint_path = args.model_path + cur_time + args.exp_name + "/"

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

# log file name
suffix = "_" + dataset

#suffix = suffix + '_drop_{}_{}_{}_lr_{}_seed_{}'.format(args.drop, args.num_pos, args.batch_size, args.lr, args.seed)
suffix = suffix + '_lr_{}_batchsize_{}'.format(args.lr, args.batch_size)
if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

test_log_file = open(log_path + suffix + '.txt', "w")
sys.stdout = Logger(log_path + suffix + '_os.txt')

# vis_log_dir = args.vis_log_path + suffix + '/'
vis_log_dir = args.vis_log_path + cur_time + suffix + "_" + args.exp_name + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_acc_v2t = 0

best_map_acc = 0  # best test accuracy
best_map_acc_v2t = 0

start_epoch = 0
feature_dim = args.low_dim
wG = 0
end = time.time()

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_train = [
    transforms.ToPILImage(),
    transforms.Resize((288, 144)),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
]


transform_train = transform_train + [ChannelRandomErasing(probability = 0.5)]
# transform_train = transform_train + [ChannelAdapGray(probability = 0.5)]
transform_train = transforms.Compose(transform_train)

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])


if dataset == 'VCM':
    rgb_pos, ir_pos = GenIdx(data_set.rgb_label, data_set.ir_label)
queryloader = DataLoader(
    VideoDataset_test(data_set.query, seq_len=seq_lenth, sample='video_test', transform=transform_test),
    batch_size=test_batch, shuffle=False, num_workers=args.workers)

galleryloader = DataLoader(
    VideoDataset_test(data_set.gallery, seq_len=seq_lenth, sample='video_test', transform=transform_test),
    batch_size=test_batch, shuffle=False, num_workers=args.workers)


# ----------------visible to infrared----------------
queryloader_1 = DataLoader(
    VideoDataset_test(data_set.query_1, seq_len=seq_lenth, sample='video_test', transform=transform_test),
    batch_size=test_batch, shuffle=False, num_workers=args.workers)

galleryloader_1 = DataLoader(
    VideoDataset_test(data_set.gallery_1, seq_len=seq_lenth, sample='video_test', transform=transform_test),
    batch_size=test_batch, shuffle=False, num_workers=args.workers)

nquery_1 = data_set.num_query_tracklets_1
ngall_1 = data_set.num_gallery_tracklets_1

n_class = data_set.num_train_pids
nquery = data_set.num_query_tracklets
ngall = data_set.num_gallery_tracklets

print('==> Building model..')

net = embed_net(class_num=n_class)
net.to(device)


if len(args.resume) > 0:
    # model_path = checkpoint_path + args.resume
    model_path = args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
loader_batch = args.batch_size * args.num_pos
criterion1 = nn.CrossEntropyLoss()
criterion2 = OriTripletLoss(batch_size=loader_batch, margin=args.margin)
criterion3 = nn.KLDivLoss(reduction='batchmean') 
criterion1.to(device)
criterion2.to(device)
criterion3.to(device)


# optimizer
if args.optim == 'sgd':

    ignored_params = list(map(id, net.base_resnet.parameters())) \
                    + list(map(id, net.visible_module.parameters())) \
                    + list(map(id, net.thermal_module.parameters())) 

    base_params = filter(lambda p: id(p) in ignored_params, net.parameters())
    main_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': main_params, 'lr': args.lr}
        ],
        weight_decay=5e-4, momentum=0.9, nesterov=True)
    
    # scheduler = warm_up_cosine_lr_scheduler(optimizer, epochs=200, warm_up_epochs=10, eta_min=1e-8)  # from util
        
elif args.optim == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

#modal_classifier_optimizer_1 = torch.optim.SGD(net_modal_classifier1.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9, nesterov=True)

def adjust_learning_rate(optimizer_P, epoch):
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif 10 <= epoch < 60:
        lr = args.lr
    elif 60 <= epoch < 120:
        lr = args.lr * 0.1
    elif epoch >= 120:
        lr = args.lr * 0.01

    # cur_lr = optimizer_P.param_groups[0]['lr'] 
    optimizer_P.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer_P.param_groups) - 1):
        optimizer_P.param_groups[i + 1]['lr'] = lr
    # optimizer_P.param_groups[0]['lr'] = lr
    
    return lr

def train(epoch, wG):
    # adjust learning rate
    # scheduler.step()
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    lid_loss = AverageMeter()
    did_loss = AverageMeter()
    glid_loss = AverageMeter()
    tri_loss = AverageMeter()
    ml_loss = AverageMeter()
    de_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    net.train()
    end = time.time()

    for batch_idx, (imgs_ir, pids_ir, camid_ir, imgs_rgb, pids_rgb, camid_rgb) in enumerate(trainloader):
        
        input1 = imgs_rgb
        input2 = imgs_ir

        label1 = pids_rgb
        label2 = pids_ir
        labels = torch.cat((label1, label2), 0)

        
        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())

        labels = Variable(labels.cuda())
        label1 = Variable(label1.cuda())
        label2 = Variable(label2.cuda())

        data_time.update(time.time() - end)

        feat, x_local, logits, l_logits, l_d_logits, intra_gcn_l_logits, inter_gcn_l_logits, loss_defense = net(input1, input2, seq_len=seq_lenth, edges=edges)
        loss_id = criterion1(logits, labels) 
        loss_tri, batch_acc = criterion2(feat, labels)  # Triplet loss
        correct += (batch_acc / 2)
        _, predicted = logits.max(dim=1)
        correct += (predicted.eq(labels).sum().item() / 2)

        local_label_1 = rearrange(repeat(label1.unsqueeze(dim=1), 'b 1 -> b n', n=seq_lenth), 'b n -> (b n)') 
        local_label_2 = rearrange(repeat(label2.unsqueeze(dim=1), 'b 1 -> b n', n=seq_lenth), 'b n -> (b n)') 
        local_label = torch.cat((local_label_1, local_label_2), dim=0)

        l_id_loss = criterion1(l_logits, local_label)
        d_id_loss = criterion1(l_d_logits, local_label) 
        g_l_id_loss = 0.5*(criterion1(intra_gcn_l_logits, local_label) + criterion1(inter_gcn_l_logits, local_label))
        # loss_ml = criterion3(F.log_softmax(gcn_logits, dim=1), F.softmax(logits, dim=1))
        loss_ml = 0.5*(criterion3(F.log_softmax(intra_gcn_l_logits, dim=1), F.softmax(l_logits, dim=1)) + \
                  criterion3(F.log_softmax(inter_gcn_l_logits, dim=1), F.softmax(l_logits, dim=1)))

        loss = loss_id + loss_tri + l_id_loss + d_id_loss + g_l_id_loss + loss_ml + loss_defense
        # loss = loss_id + loss_tri + l_id_loss + d_id_loss + g_l_id_loss + loss_defense
        
        loss_total = loss
        optimizer.zero_grad()  
        loss_total.backward()
        optimizer.step()

        # log different loss components
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        lid_loss.update(l_id_loss.item(), 2 * input1.size(0))
        did_loss.update(d_id_loss.item(), 2 * input1.size(0))
        glid_loss.update(g_l_id_loss.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        ml_loss.update(loss_ml.item(), 2 * input1.size(0))
        de_loss.update(loss_defense.item(), 2 * input1.size(0))

        total += labels.size(0)

        # measure elapsed time  
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 10 == 0:

            print('Epoch: [{}][{}/{}] '
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                'lr:{} '
                'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                'i Loss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                'li Loss: {lid_loss.val:.4f} ({lid_loss.avg:.4f}) '
                'di Loss: {did_loss.val:.4f} ({did_loss.avg:.4f}) '
                'glid Loss: {glid_loss.val:.4f} ({glid_loss.avg:.4f}) '
                't Loss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) ' 
                'ml Loss: {ml_loss.val:.4f} ({ml_loss.avg:.4f}) ' 
                'de Loss: {de_loss.val:.4f} ({de_loss.avg:.4f}) ' 
                'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss = train_loss, id_loss = id_loss, lid_loss=lid_loss, did_loss=did_loss,
                tri_loss = tri_loss, glid_loss=glid_loss, ml_loss=ml_loss, de_loss=de_loss))

            writer.add_scalar('total_loss', train_loss.avg, epoch)
            writer.add_scalar('id_loss', id_loss.avg, epoch)
            writer.add_scalar('lid_loss', lid_loss.avg, epoch)
            writer.add_scalar('did_loss', did_loss.avg, epoch)
            writer.add_scalar('glid_loss', glid_loss.avg, epoch)
            writer.add_scalar('tri_loss', tri_loss.avg, epoch)
            writer.add_scalar('ml_loss', ml_loss.avg, epoch)
            writer.add_scalar('de_loss', de_loss.avg, epoch)
            writer.add_scalar('lr', current_lr, epoch)
    return 1. / (1. + train_loss.avg)

def test2(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall_1, 2048))
    q_pids, q_camids = [], []
    g_pids, g_camids = [], []
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader_1):
            input = imgs
            input = Variable(input.cuda())
            label = pids
            batch_num = input.size(0)
            #feat = net(input, input, 0, test_mode[1], seq_len=seq_lenth)
            feat = net(input, input, test_mode[1], seq_len=seq_lenth, edges=edges)  
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
            #
            g_pids.extend(pids)
            g_camids.extend(camids)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery_1, 2048))
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader_1):
            input = imgs
            label = pids

            batch_num = input.size(0)
            input = Variable(input.cuda())
            #feat = net(input, input, 0, test_mode[0], seq_len=seq_lenth)
            feat = net(input, input, test_mode[0], seq_len=seq_lenth, edges=edges)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num

            q_pids.extend(pids)
            q_camids.extend(camids)

    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))

    # evaluation
    cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)

    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    ranks = [1, 5, 10, 20]
    print("Results ----------")
    print("testmAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    return cmc,mAP

def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    q_pids, q_camids = [], []
    g_pids, g_camids = [], []
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            input = imgs
            label = pids
            batch_num = input.size(0)

            input = Variable(input.cuda())
            #feat = net(input, input, 0, test_mode[0], seq_len=seq_lenth)
            feat = net(input, input, test_mode[0], seq_len=seq_lenth, edges=edges)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()

            ptr = ptr + batch_num

            g_pids.extend(pids)
            g_camids.extend(camids)

    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))

    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            input = imgs
            label = pids

            batch_num = input.size(0)

            input = Variable(input.cuda())
            #feat = net(input, input, 0, test_mode[1], seq_len=seq_lenth)
            feat = net(input, input, test_mode[1], seq_len=seq_lenth, edges=edges)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()

            ptr = ptr + batch_num

            q_pids.extend(pids)
            q_camids.extend(camids)

    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)

    ranks = [1, 5, 10, 20]
    print("Results ----------")
    print("testmAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    return cmc,mAP

# training
print('==> Start Training...')

for epoch in range(start_epoch, 201 - start_epoch):

    print('==> Preparing Data Loader...')
    # IdentitySampler
    # rgb_pos/ir_pos is a list.
    # the i-th item in rgb_pos/ir_pos is also a list, represents all tracklets id for the person with pid "i" from rgb/ir modality.
    # there are totally 5460/4291 for rgb/ir modality, every person has multiple tracklets.
    # num_pos=2: the number of positive samples for every person under certain modality.
    # batch_size=8:the number of pids in a mini-batch.
    # data number in a minibatch=batch_size * 2 * num_pos (2 for 2 modalities)

    sampler = IdentitySampler(data_set.ir_label, data_set.rgb_label, rgb_pos, ir_pos, args.num_pos, args.batch_size)

    index1 = sampler.index1  # ndarray,all tracklets for rgb modality
    index2 = sampler.index2  # ndarray,all tracklets for ir modality
    
    loader_batch = args.batch_size * args.num_pos  

    trainloader = DataLoader(
        VideoDataset_train(data_set.train_ir, data_set.train_rgb, seq_len=seq_lenth, sample='video_train',
                           transform=transform_train, index1=index1, index2=index2),
        sampler=sampler,
        batch_size=loader_batch, num_workers=args.workers,
        drop_last=True,
    )

    # training
    wG = train(epoch, wG)

    if epoch >= 0 and epoch % 10 == 0:
        print('Test Epoch: {}'.format(epoch))
        print('Test Epoch: {}'.format(epoch), file=test_log_file)

        # testing
        cmc, mAP = test(epoch)

        if cmc[0] > best_acc:
            best_acc = cmc[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + 't2v_rank1_best.t')

        if mAP > best_map_acc:
            best_map_acc = mAP
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + 't2v_map_best.t')

        print(
            'FC(t2v):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        print('Best t2v epoch [{}]'.format(best_epoch))
        print(
            'FC(t2v):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP), file=test_log_file)
#-------------------------------------------------------------------------------------------------------------------
        cmc, mAP = test2(epoch)
        if cmc[0] > best_acc_v2t:
            best_acc_v2t = cmc[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + 'v2t_rank1_best.t')

        if mAP > best_map_acc_v2t:
            best_map_acc_v2t = mAP
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + 'v2t_map_best.t')

        print(
            'FC(v2t):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        print('Best v2t epoch [{}]'.format(best_epoch))
        print(
            'FC(v2t):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP), file=test_log_file)


        test_log_file.flush()
