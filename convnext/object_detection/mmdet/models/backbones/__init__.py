from mmdet.models.backbones.darknet import Darknet
from mmdet.models.backbones.detectors_resnet import DetectoRS_ResNet
from mmdet.models.backbones.detectors_resnext import DetectoRS_ResNeXt
from mmdet.models.backbones.hourglass import HourglassNet
from mmdet.models.backbones.hrnet import HRNet
from mmdet.models.backbones.regnet import RegNet
from mmdet.models.backbones.res2net import Res2Net
from mmdet.models.backbones.resnest import ResNeSt
from mmdet.models.backbones.resnet import ResNet, ResNetV1d
from mmdet.models.backbones.resnext import ResNeXt
from mmdet.models.backbones.ssd_vgg import SSDVGG
from mmdet.models.backbones.trident_resnet import TridentResNet
from mmdet.models.backbones.swin import SwinTransformer
from .convnext import ConvNeXt

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'Res2Net',
    'HourglassNet', 'DetectoRS_ResNet', 'DetectoRS_ResNeXt', 'Darknet',
    'ResNeSt', 'TridentResNet', 'SwinTransformer', 'ConvNeXt'
]
