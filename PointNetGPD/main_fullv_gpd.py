import argparse
import os
import time
import pickle

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from model.dataset import PointGraspDataset
from model.gpd import *

parser = argparse.ArgumentParser(description='pointnetGPD, gpd baseline')
parser.add_argument('--tag', type=str, default='default')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--mode', choices=['train', 'test'], required=True)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--load-model', type=str, default='')
parser.add_argument('--load-epoch', type=int, default=-1)
parser.add_argument('--model-path', type=str, default='./assets/learned_models',
                   help='pre-trained model path')
parser.add_argument('--data-path', type=str, default='./data', help='data path')
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--save-interval', type=int, default=1)

args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available else False

if args.cuda:
    torch.cuda.manual_seed(1)

logger = SummaryWriter(os.path.join('./assets/log/', 'GPD_{}'.format(args.tag)))
np.random.seed(int(time.time()))

def worker_init_fn(pid):
    np.random.seed(torch.initial_seed() % (2**31-1))

def my_collate(batch):
    batch = list(filter(lambda x:x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

grasp_points_num=1000
obj_points_num=50000
pc_file_used_num=20
thresh_good=0.6
thresh_bad=0.6

input_size=60
input_chann=3 # 3 # 12

train_loader = torch.utils.data.DataLoader(
    PointGraspDataset(
        obj_points_num=obj_points_num,
        grasp_points_num=grasp_points_num,
        pc_file_used_num=pc_file_used_num,
        path=args.data_path,
        tag='train',
        grasp_amount_per_file=6500,
        thresh_good=thresh_good,
        thresh_bad=thresh_bad,
        projection=True,
        project_chann=input_chann,
        project_size=input_size,
    ),
    batch_size=args.batch_size,
    num_workers=32,
    pin_memory=True,
    shuffle=True,
    worker_init_fn=worker_init_fn,
    collate_fn=my_collate,
)

test_loader = torch.utils.data.DataLoader(
    PointGraspDataset(
        obj_points_num=obj_points_num,
        grasp_points_num=grasp_points_num,
        pc_file_used_num=pc_file_used_num,
        path=args.data_path,
        tag='test',
        grasp_amount_per_file=500,
        thresh_good=thresh_good,
        thresh_bad=thresh_bad,
        with_obj=True,
        projection=True,
        project_chann=input_chann,
        project_size=input_size,
    ),
    batch_size=args.batch_size,
    num_workers=32,
    pin_memory=True,
    shuffle=True,
    worker_init_fn=worker_init_fn,
    collate_fn=my_collate,
)

is_resume = 0
if args.load_model and args.load_epoch != -1:
    is_resume = 1

if is_resume or args.mode == 'test':
    model = torch.load(args.load_model, map_location='cuda:{}'.format(args.gpu))
    model.device_ids = [args.gpu]
    print('load model {}'.format(args.load_model))
else:
    model = GPDClassifier(input_chann)
if args.cuda:
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    else:
        device_id = [0,1]
        torch.cuda.set_device(device_id[0])
        model = nn.DataParallel(model, device_ids=device_id).cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

def train(model, loader, epoch):
    scheduler.step()
    model.train()
    torch.set_grad_enabled(True)
    correct = 0
    dataset_size = 0
    for batch_idx, (data, target) in enumerate(loader):
        dataset_size += data.shape[0]
        data, target = data.float(), target.long().squeeze()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{}'.format(
            epoch, batch_idx * len(data), len(loader.dataset),
            100. * batch_idx * len(data) / len(loader.dataset), loss.item(), args.tag))
            logger.add_scalar('train_loss', loss.cpu().item(),
                    batch_idx + epoch * len(loader))
    return float(correct)/float(dataset_size)


def test(model, loader):
    model.eval()
    torch.set_grad_enabled(False)
    test_loss = 0
    correct = 0
    dataset_size = 0
    da = {}
    db = {}
    res = []
    for batch_idx, (data, target, obj_name) in enumerate(loader):
        dataset_size += data.shape[0]
        data, target = data.float(), target.long().squeeze()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data) # N*C
        test_loss += F.nll_loss(output, target, size_average=False).cpu().item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        for i, j, k in zip(obj_name, pred.data.cpu().numpy(), target.data.cpu().numpy()):
            res.append((i, j[0], k))

    test_loss /= len(loader.dataset)
    acc = float(correct)/float(dataset_size)
    return acc, test_loss


def main():
    if args.mode == 'train':
        for epoch in range(is_resume*args.load_epoch, args.epoch):
            acc_train = train(model, train_loader, epoch)
            print('Train done, acc={}'.format(acc_train))
            acc, loss = test(model, test_loader)
            print('Test done, acc={}, loss={}'.format(acc, loss))
            logger.add_scalar('train_acc', acc_train, epoch)
            logger.add_scalar('test_acc', acc, epoch)
            logger.add_scalar('test_loss', loss, epoch)
            if epoch % args.save_interval == 0:
                path = os.path.join(args.model_path, args.tag + '_{}.model'.format(epoch))
                torch.save(model, path)
                print('Save model @ {}'.format(path))
    else:
        print('testing...')
        acc, loss = test(model, test_loader)
        print('Test done, acc={}, loss={}'.format(acc, loss))

if __name__ == "__main__":
    main()
