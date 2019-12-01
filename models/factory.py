# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.backends.cudnn as cudnn

from .DSFD_vgg import build_net_vgg
from .DSFD_resnet import build_net_resnet
from .DSFD_efficient_b0 import build_net_efficient_b0
from .DSFD_efficient_b1 import build_net_efficient_b1


def build_net(phase, num_classes=2, model='vgg'):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return

    if model != 'vgg' and 'resnet' not in model and model != 'efficient':
        print("ERROR: model:" + model + " not recognized")
        return

    if model == 'vgg':
        return build_net_vgg(phase, num_classes)
    elif model == 'efficient_b0':
        return build_net_efficient_b0(phase, num_classes) 
    elif model == 'efficient_b1':
        return build_net_efficient_b1(phase, num_classes)
    else:
        return build_net_resnet(phase, num_classes, model)

def basenet_factory(model='vgg'):
    if model == 'vgg':
        return 'vgg16_reducedfc.pth'
    elif model == 'efficient_b0':
        return 'efficientnet-b0-355c32eb.pth'
    elif model == 'efficient_b1':
        return 'efficientnet-b1-f1951068.pth'
    elif 'resnet' in model:
        return '{}.pth'.format(model) 

