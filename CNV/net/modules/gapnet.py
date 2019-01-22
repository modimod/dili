import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GAPNet02(nn.Module):
    def __init__(self, fc_units=2048, dropout=0.5, gap1=True, gap2=True, gap3=True, gap4=True, num_classes=478, input_shape=None):
        super(GAPNet02, self).__init__()
        assert input_shape
        in_c = input_shape[0]
        in_h = input_shape[1]
        in_w = input_shape[2]
        # fc_units = model_params.get_value("fc_units", 1024)
        # drop_prob = model_params.get_value("dropout", 0.5)

        self.block1 = nn.Sequential(
            # input downsampling
            nn.Conv2d(in_c, 32, kernel_size=3, stride=2, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.SELU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.SELU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.SELU(inplace=True)
        )
        # global average pooling
        h, w = calc_out_shape(in_h, in_w, self.block1)
        if gap1:
            self.gap1 = nn.AvgPool2d(kernel_size=(h, w))
        h, w = calc_out_shape(h, w, self.block2)
        if gap2:
            self.gap2 = nn.AvgPool2d(kernel_size=(h, w))
        h, w = calc_out_shape(h, w, self.block3)
        if gap3:
            self.gap3 = nn.AvgPool2d(kernel_size=(h, w))
        h, w = calc_out_shape(h, w, self.block4)
        if gap4:
            self.gap4 = nn.AvgPool2d(kernel_size=(h, w))

        # classifier
        # gap_shape = 32 + 64 + 128 + 256
        gap_shape = (32 if gap1 else 0) + (64 if gap2 else 0) + (128 if gap3 else 0) + (256 if gap4 else 0)
        self.classifier = nn.Sequential(
            nn.Linear(gap_shape, fc_units),
            nn.SELU(inplace=True),
            nn.AlphaDropout(p=dropout),
            nn.Linear(fc_units, fc_units),
            nn.SELU(inplace=True),
            nn.AlphaDropout(p=dropout),
            nn.Linear(fc_units, num_classes)
        )

        # init
        self.init_parameters()

    def init_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_in = np.prod(module.weight.size()[1:])
                nn.init.normal_(module.weight, 0, np.sqrt(1 / fan_in))
                nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Linear):
                fan_in = module.weight.size()[0]
                nn.init.normal_(module.weight, 0, np.sqrt(1 / fan_in))
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        batchsize = x.size(0)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        gaps = []
        if hasattr(self, "gap1"):
            gaps.append(self.gap1(x1).view(batchsize, -1))
        if hasattr(self, "gap2"):
            gaps.append(self.gap2(x2).view(batchsize, -1))
        if hasattr(self, "gap3"):
            gaps.append(self.gap3(x3).view(batchsize, -1))
        if hasattr(self, "gap4"):
            gaps.append(self.gap4(x4).view(batchsize, -1))
        x = torch.cat(gaps, dim=1)
        x = self.classifier(x)
        return x

    # def loss(self, prediction, target):
    #     y = target / 2 + 0.5
    #     p = prediction
    #     eps = 1e-7
    #     mask = (y != 0.5).float().detach()
    #     # ce = -(y * F.logsigmoid(p) + (1 - y) * torch.log(1 - torch.sigmoid(p) + 1e-7))
    #     bce = p.clamp(min=0) - p * y + torch.log(1.0 + torch.exp(-p.abs()))
    #     bce[mask == 0] = 0
    #     loss = bce.sum() / (mask.sum() + eps)
    #     if math.isnan(loss.item()):
    #         print("NAN LOSS ENCOUNTERED!!!")
    #         print("Prediction: {}".format(prediction))
    #         print("Target: {}".format(target))
    #         print("Target Mask Sum: {}".format(mask.sum()))
    #     return loss

def calc_out_shape(in_h, in_w, layers):
    relevant = []
    for l in layers:
        k, s, p, d = None, None, None, (1, 1)
        if hasattr(l, "kernel_size"):
            if type(l.kernel_size) is int:
                k = (l.kernel_size, l.kernel_size)
            else:
                k = l.kernel_size
        if hasattr(l, "stride"):
            if type(l.stride) is int:
                s = (l.stride, l.stride)
            else:
                s = l.stride
        if hasattr(l, "padding"):
            if type(l.padding) is int:
                p = (l.padding, l.padding)
            else:
                p = l.padding
        if hasattr(l, "dilation"):
            if type(l.dilation) is int:
                d = (l.dilation, l.dilation)
            else:
                d = l.dilation
        if k is not None:
            relevant.append([k, s, p, d])
    h, w = in_h, in_w
    for l in relevant:
        h, w = calc_conv2d_out_shape([h, w], ksize=l[0], padding=l[2], stride=l[1], dilation=l[3])
    return h, w

def calc_conv2d_out_shape(input, ksize, padding, stride, dilation):
    return [int(((input[i] + 2 * padding[i] - (ksize[i] - 1) * dilation[i] - 1) / stride[i]) + 1) for i in range(0, len(input))]
