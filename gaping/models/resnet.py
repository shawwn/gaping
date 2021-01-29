#import torch
#from torch import Tensor
#from .utils import load_state_dict_from_url
from .. import tftorch as nn
Tensor = nn.Tensor
from typing import Type, Any, Callable, Union, List, Optional


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, **kws) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, **kws)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, **kws) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, **kws)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        scope: str = 'block',
        **kws,
    ) -> None:
        super(BasicBlock, self).__init__(scope=scope, **kws)
        with self.scope():
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            if groups != 1 or base_width != 64:
                raise ValueError('BasicBlock only supports groups=1 and base_width=64')
            if dilation > 1:
                raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            # Both self.conv1 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv3x3(inplanes, planes, stride, scope='conv1')
            self.bn1 = norm_layer(planes, scope='bn1')
            self.relu = nn.ReLU(inplace=True, scope='relu')
            self.conv2 = conv3x3(planes, planes, scope='conv2')
            self.bn2 = norm_layer(planes, scope='bn2')
            self.downsample = downsample
            self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        with self.scope():
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out



class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        scope: str = 'bottleneck',
        **kws
    ) -> None:
        super(Bottleneck, self).__init__(scope=scope, **kws)
        with self.scope():
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            width = int(planes * (base_width / 64.)) * groups
            # Both self.conv2 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv1x1(inplanes, width, scope='conv1')
            self.bn1 = norm_layer(width, scope='bn1')
            self.conv2 = conv3x3(width, width, stride, groups, dilation, scope='conv2')
            self.bn2 = norm_layer(width, scope='bn2')
            self.conv3 = conv1x1(width, planes * self.expansion, scope='conv3')
            self.bn3 = norm_layer(planes * self.expansion, scope='bn3')
            self.relu = nn.ReLU(inplace=True, scope='relu')
            self.downsample = downsample
            self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        with self.scope():
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

