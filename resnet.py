from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, layers, num_cls, pre_conv="full", **kwargs):
        super(ResNet, self).__init__()
        if pre_conv == "full":
            self.pre_conv = nn.Sequential(*layers[0:4])
        elif pre_conv == "small":
            self.pre_conv = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
        else:
            raise ValueError("pre_conv must be one of ['full', 'small']")

        self.layer1 = layers[4]
        self.layer2 = layers[5]
        self.layer3 = layers[6]
        self.layer4 = layers[7]
        self.linear = nn.Linear(512 * self.layer1[0].expansion, num_cls)

    def forward(self, x, return_feat=False):
        out = self.pre_conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size(-1))
        feat = out.view(out.size(0), -1)
        out = self.linear(feat)
        if return_feat:
            return feat, out
        else:
            return out


def _resnet(net, **kwargs):
    layers = list(net(pretrained=True).children())[:-1]
    return ResNet(layers, **kwargs)


def resnet18(num_cls, **kwargs):
    return _resnet(models.resnet18, num_cls=num_cls, **kwargs)


def resnet34(num_cls, **kwargs):
    return _resnet(models.resnet34, num_cls=num_cls, **kwargs)


def resnet50(num_cls, **kwargs):
    return _resnet(models.resnet50, num_cls=num_cls, **kwargs)


def resnet101(num_cls, **kwargs):
    return _resnet(models.resnet101, num_cls=num_cls, **kwargs)
