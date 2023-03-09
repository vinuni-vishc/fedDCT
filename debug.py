import os
import sys
import time
import random
import shutil
import argparse
import warnings
import setproctitle
import json
import torch
import torch.cuda.amp as amp
from torch import nn, distributed
from torch.backends import cudnn
from tensorboardX import SummaryWriter
from config import *
from params import train_params
from utils import label_smoothing, norm, summary, metric, lr_scheduler, rmsprop_tf, prefetch
from model import splitnet
from utils.thop import profile, clever_format
from dataset import factory

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
norm_layer = nn.BatchNorm2d
in_channels = 32
net = nn.Sequential(
			nn.Conv2d(3, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
			norm_layer(in_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
			norm_layer(in_channels),
			nn.ReLU(inplace=True),
			# output channle = inplanes * 2
			nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
			norm_layer(in_channels * 2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
			)

dummy_tensor = torch.rand(128, 3, 64, 64)
print(net)
print(net(dummy_tensor).shape)