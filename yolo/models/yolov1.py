import torch
import torch.nn as nn
import numpy as np

from ..backbone import resnet18
from ..utils import Conv, SPP, BottleneckCSP, SAM


class YoloV1(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.5, hr=False):
        super(YoloV1, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = 32
        self.grid_cell = self.create_grid(input_size)
        self.input_size = input_size
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=device).float()

        # we use resnet18 as backbone
        self.backbone = resnet18(pretrained=True)
        # neck
        self.SPP = nn.Sequential(
            Conv(512, 256, k=1),
            SPP(),
            BottleneckCSP(256*4, 512, n=1, shortcut=False)
        )
        self.SAM = SAM(512)
        self.conv_set = BottleneckCSP(512, 512, n=3, shortcut=False)

        self.pred = nn.Conv2d(512, 1 + self.num_classes + 4, 1)

    def create_grid(self, input_size):
        print('')

    def set_grid(self, input_size):
        print('')

    def decode_boxes(self, pred):
        print('')

    def nms(self, dets, scores):
        print('')

    def postprocess(self, all_local, all_conf, exchange=True, im_shape=None):
        print('')

    def forward(self, x, target=None):
        print()
