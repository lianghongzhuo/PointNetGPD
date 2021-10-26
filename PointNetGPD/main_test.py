#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import time
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from scipy.stats import mode
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath("__file__"))))
parser = argparse.ArgumentParser(description="pointnetGPD")
parser.add_argument("--cuda", action="store_true", default=False)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--load-model", type=str,
                    default="../data/pointnetgpd_3class.model")
parser.add_argument("--show_final_grasp", action="store_true", default=False)
parser.add_argument("--tray_grasp", action="store_true", default=False)
parser.add_argument("--using_mp", action="store_true", default=False)
parser.add_argument("--model_type", type=str)

args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available else False

if args.cuda:
    torch.cuda.manual_seed(1)

np.random.seed(int(time.time()))

if args.model_type == "100":
    args.load_model = "../data/pointgpd_chann3_local.model"
elif args.model_type == "50":
    args.load_model = "../data/pointgpd_50_points.model"
elif args.model_type == "3class":  # input points number is 500
    args.load_model = "../data/pointnetgpd_3class.model"
else:
    print("Using default model file")
model = torch.load(args.load_model, map_location="cpu")
model.device_ids = [args.gpu]
print("load model {}".format(args.load_model))

if args.cuda:
    model = torch.load(args.load_model, map_location="cuda:{}".format(args.gpu))
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    else:
        device_id = [0, 1]
        torch.cuda.set_device(device_id[0])
        model = nn.DataParallel(model, device_ids=device_id).cuda()
if isinstance(model, torch.nn.DataParallel):
    model = model.module


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
    num_point = 500
    model.eval()
    torch.set_grad_enabled(False)

    # load pc(should be in local gripper coordinate)
    # local_pc: (N, 3)
    # local_pc = np.load("test.npy")
    local_pc = np.random.random([500, 3])  # test only
    predict = []
    for _ in range(repeat):
        if len(local_pc) >= num_point:
            local_pc = local_pc[np.random.choice(len(local_pc), num_point, replace=False)]
        else:
            local_pc = local_pc[np.random.choice(len(local_pc), num_point, replace=True)]

        # run model
        predict.append(test_network(model, local_pc)[0])
    print("voting: ", predict)
    predict = mode(predict).mode[0]

    # output
    print("Test result:", predict)


if __name__ == "__main__":
    main()
