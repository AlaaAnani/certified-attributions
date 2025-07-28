'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init
from attribution_certification.models.layers_for_lrp import *

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'vgg11_lessconv', 'vgg11_lessconv_lrp',
]


class VGG(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 5 * 5, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGGAP(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, features):
        super(VGGAP, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False, inplace_relu=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d,
                           nn.BatchNorm2d(v), nn.ReLU(inplace=inplace_relu)]
            else:
                layers += [conv2d, nn.ReLU(inplace=inplace_relu)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGS(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, features):
        super(VGGS, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 12 * 12, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGGLC(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, features, inplace_relu=True):
        super(VGGLC, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(inplace_relu),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace_relu),
            nn.Linear(512, 10),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGGLC_LRP(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, features):
        super(VGGLC_LRP, self).__init__()
        self.features = features
        self.avgpool = AdaptiveAvgPool2d((2, 2))
        self.classifier = Sequential(
            Dropout(),
            Linear(512 * 2 * 2, 512),
            ReLU(True),
            Dropout(),
            Linear(512, 512),
            ReLU(True),
            Linear(512, 10),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

    def relprop(self, R, alpha):
        x = self.classifier.relprop(R, alpha)
        # print(next(reversed(self.features._modules.values())))
        x = x.reshape_as(next(reversed(self.features._modules.values())).Y)
        x = self.avgpool.relprop(x, alpha)
        x = self.features.relprop(x, alpha)
        return x

    def m_relprop(self, R, pred, alpha):
        x = self.classifier.m_relprop(R, pred, alpha)
        if torch.is_tensor(x) == False:
            for i in range(len(x)):
                x[i] = x[i].reshape_as(
                    next(reversed(self.features._modules.values())).Y)
        else:
            x = x.reshape_as(next(reversed(self.features._modules.values())).Y)
        x = self.avgpool.m_relprop(x, pred, alpha)
        x = self.features.m_relprop(x, pred, alpha)
        return x

    def RAP_relprop(self, R):
        x1 = self.classifier.RAP_relprop(R)
        if torch.is_tensor(x1) == False:
            for i in range(len(x1)):
                x1[i] = x1[i].reshape_as(
                    next(reversed(self.features._modules.values())).Y)
        else:
            x1 = x1.reshape_as(
                next(reversed(self.features._modules.values())).Y)
        x1 = self.avgpool.RAP_relprop(x1)
        x1 = self.features.RAP_relprop(x1)
        return x1


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'N', 512, 512, 'N', 512, 512, 'N'],
    'ALC': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'AS': [64, 'M', 128, 'N', 256, 256, 'N', 512, 512, 'N', 512, 512, 'N'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_s():
    """VGG 11-layer model (configuration "A")"""
    return VGGS(make_layers(cfg['AS']))


def vgg11_ap():
    """VGG 11-layer model (configuration "A")"""
    return VGGAP(make_layers(cfg['A']))


def vgg11_lessconv(inplace_relu=True):
    return VGGLC(make_layers(cfg['ALC'], inplace_relu=inplace_relu), inplace_relu=inplace_relu)


def vgg11_lessconv_lrp():
    return VGGLC_LRP(make_layers(cfg['ALC']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    # print("HI3")
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))
