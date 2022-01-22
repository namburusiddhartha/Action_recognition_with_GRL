# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead

from torch.autograd import Function

class ReverseGradLayer(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.alpha
        return grad_input, None

@HEADS.register_module()
class I3DHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_clss = nn.Linear(self.in_channels, 2)
        #self.fc_dom1 = nn.Linear(self.in_channels, 1024)
        self.fc_dom = nn.Sequential(
            nn.Linear(self.in_channels, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2))
        #self.fc_dom3 = nn.Linear(1024, 2)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool_d = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.avg_pool = nn.AvgPool3d((1, 7, 7))
        else:
            self.avg_pool = None

        self.headc = nn.Conv3d(2048, 8, (1, 1, 1))

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.headc, std=self.init_std)

    def forward(self, x, alpha):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        # print(x.shape)
        #print(x)

        reverse_feat = ReverseGradLayer.apply(x, alpha)
        if self.avg_pool is not None:
            y = self.avg_pool_d(reverse_feat)
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        #if self.dropout is not None:
            #x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        # print(x.shape)
        # x = x.view(x.shape[0], -1)
        # print("FINAL SHAPE")
        # print(x.shape)
        x = self.headc(x)
        x = torch.squeeze(x, 4)
        x = torch.squeeze(x, 3)

        y = y.view(y.shape[0], -1)

        # [N, in_channels]
        dom_score = self.fc_dom(y)
        # [N, num_classes]
        # print(cls_score.shape)
        #print("Head result")
        #print(x)
        #print("\n")
        #print("HEAD")
        #print(x.shape)
        #print(dom_score.shape)
        #z = torch.cat((x.reshape(x.shape[0], 64),dom_score),1)
        #print(z.shape)
        #print(z.shape)
        #lal = x[:, :2, 0]
        return x, dom_score
