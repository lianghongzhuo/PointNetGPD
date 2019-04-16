#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import time
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.stats import mode
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath("__file__"))))
from model.dataset import PointGraspDataset
from model.pointnet import PointNetCls, DualPointNetCls

parser = argparse.ArgumentParser(description='pointnetGPD')
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--load-model', type=str, default=os.environ['HOME'] + '/code/grasp-pointnet/PointNetGPD/data/pointgpd_chann3_local.model')
parser.add_argument('--show_final_grasp', action='store_true', default=False)
parser.add_argument('--tray_grasp', action='store_true', default=False)
parser.add_argument('--using_mp', action='store_true', default=False)
parser.add_argument('--model_type', type=str)

args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available else False

if args.cuda:
    torch.cuda.manual_seed(1)

np.random.seed(int(time.time()))

grasp_points_num=1000
point_channel=3

if args.model_type == "100":
    args.load_model = os.environ['HOME'] + '/code/grasp-pointnet/PointNetGPD/data/pointgpd_chann3_local.model'
elif args.model_type == "50":
    args.load_model = os.environ['HOME'] + '/code/grasp-pointnet/PointNetGPD/data/pointgpd_50_points.model'
elif args.model_type == "3class":
    args.load_model = os.environ['HOME'] + '/code/grasp-pointnet/PointNetGPD/data/pointgpd_3class.model'
else:
    print("Using default model file")
model = torch.load(args.load_model, map_location='cpu')
model.device_ids = [args.gpu]
print('load model {}'.format(args.load_model))

if args.cuda:
    model = torch.load(args.load_model, map_location='cuda:{}'.format(args.gpu))
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    else:
        device_id = [0,1]
        torch.cuda.set_device(device_id[0])
        model = nn.DataParallel(model, device_ids=device_id).cuda()


def test_network(model_, local_pc):
    local_pc = local_pc.T
    local_pc = local_pc[np.newaxis, ...]
    local_pc = torch.FloatTensor(local_pc)
    if args.cuda:
        local_pc = local_pc.cuda()
    output, _ = model_(local_pc)  # N*C
    output = output.softmax(1)
    pred = output.data.max(1, keepdim=True)[1]
    output = output.cpu()
    return pred[0], output.data.numpy()


def main():
    repeat = 10
    num_point = 50
    model.eval()
    torch.set_grad_enabled(False)

    # load pc(should be in local gripper coordinate)
    # local_pc: (N, 3)
    local_pc = np.load('test.npy')
    predict = []
    for _ in range(repeat):
        if len(local_pc) >= num_point:
            local_pc = local_pc[np.random.choice(len(local_pc), num_point, replace=False)]
        else:
            local_pc = local_pc[np.random.choice(len(local_pc), num_point, replace=True)]

        # run model
        predict.append(test_network(model, local_pc).item())
    print('voting: ', predict)
    predict = mode(predict).mode[0]

    # output
    print('Test result:', predict)


if __name__ == "__main__":
    main()

