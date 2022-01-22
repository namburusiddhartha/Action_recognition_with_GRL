# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


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
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            # self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.avg_pool = nn.AvgPool3d((1, 7, 7))
        else:
            self.avg_pool = None

        self.headc = nn.Conv3d(2048, 8, (1, 1, 1))

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.headc, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        # print(x.shape)
        #print(x)
        if self.avg_pool is not None:
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

        # [N, in_channels]
        # cls_score = self.fc_cls(x)
        # [N, num_classes]
        # print(cls_score.shape)
        #print("Head result")
        #print(x)
        #print("\n")
        return x
