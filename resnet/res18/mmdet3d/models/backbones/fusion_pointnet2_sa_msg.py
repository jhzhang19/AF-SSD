import torch
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16
from torch import nn as nn
from .. import builder
from mmdet3d.ops import build_sa_module
from mmdet.models import BACKBONES
from .base_pointnet import BasePointNet
from mmdet3d.ops import build_attention_module


@BACKBONES.register_module()
class FusionPointNet2SAMSG(BasePointNet):
    """PointNet2 with Multi-scale grouping.

    Args:
        添加了fusion_layer和attention_layer
        fusion_layer:用于从传入的图像特征当中提取出传入的点的特征向量
        attention_layer:用于将点的特征和图像特征经过注意力机制融合

        in_channels (int): Input channels of point cloud.
        num_points (tuple[int]): The number of points which each SA
            module samples.
        radii (tuple[float]): Sampling radii of each SA module.
        num_samples (tuple[int]): The number of samples for ball
            query in each SA module.
        sa_channels (tuple[tuple[int]]): Out channels of each mlp in SA module.
        aggregation_channels (tuple[int]): Out channels of aggregation
            multi-scale grouping features.
        fps_mods (tuple[int]): Mod of FPS for each SA module.
        fps_sample_range_lists (tuple[tuple[int]]): The number of sampling
            points which each SA module samples.
        dilated_group (tuple[bool]): Whether to use dilated ball query for
        out_indices (Sequence[int]): Output from which stages.
        norm_cfg (dict): Config of normalization layer.
        sa_cfg (dict): Config of set abstraction module, which may contain
            the following keys and values:

            - pool_mod (str): Pool method ('max' or 'avg') for SA modules.
            - use_xyz (bool): Whether to use xyz as a part of features.
            - normalize_xyz (bool): Whether to normalize xyz with radii in
              each SA module.
    """

    def __init__(self,
                 in_channels,
                 num_points=(2048, 1024, 512, 256),
                 radii=((0.2, 0.4, 0.8), (0.4, 0.8, 1.6), (1.6, 3.2, 4.8)),
                 num_samples=((32, 32, 64), (32, 32, 64), (32, 32, 32)),
                 sa_channels=(((16, 16, 32), (16, 16, 32), (32, 32, 64)),
                              ((64, 64, 128), (64, 64, 128), (64, 96, 128)),
                              ((128, 128, 256), (128, 192, 256), (128, 256,
                                                                  256))),
                 aggregation_channels=(64, 128, 256),
                 fps_mods=(('D-FPS'), ('FS'), ('F-FPS', 'D-FPS')),
                 fps_sample_range_lists=((-1), (-1), (512, -1)),
                 dilated_group=(True, True, True),
                 out_indices=(2, ),
                 norm_cfg=dict(type='BN2d'),
                 sa_cfg=dict(
                     type='PointSAModuleMSG',
                     pool_mod='max',
                     use_xyz=True,
                     normalize_xyz=False),
                 fusion_layer=None,
                 att_img_inchannel=(256,256,256),
                 att_pts_inchannel=(64,128,256),
                 att_out_channel=(64,128,256),
                 att_cfg=None):
        super().__init__()
        self.num_sa = len(sa_channels)
        self.out_indices = out_indices
        assert max(out_indices) < self.num_sa
        assert len(num_points) == len(radii) == len(num_samples) == len(
            sa_channels) == len(aggregation_channels)

        self.SA_modules = nn.ModuleList()
        self.aggregation_mlps = nn.ModuleList()

        self.attention_layers = nn.ModuleList()
        sa_in_channel = in_channels - 3  # number of channels without xyz
        skip_channel_list = [sa_in_channel]

        # build the fusion_layer
        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)

        for sa_index in range(self.num_sa):
            cur_sa_mlps = list(sa_channels[sa_index])
            sa_out_channel = 0
            for radius_index in range(len(radii[sa_index])):
                cur_sa_mlps[radius_index] = [sa_in_channel] + list(
                    cur_sa_mlps[radius_index])
                sa_out_channel += cur_sa_mlps[radius_index][-1]
                # print("sa_out_channel:",sa_out_channel)

            if isinstance(fps_mods[sa_index], tuple):
                cur_fps_mod = list(fps_mods[sa_index])
            else:
                cur_fps_mod = list([fps_mods[sa_index]])

            if isinstance(fps_sample_range_lists[sa_index], tuple):
                cur_fps_sample_range_list = list(
                    fps_sample_range_lists[sa_index])
            else:
                cur_fps_sample_range_list = list(
                    [fps_sample_range_lists[sa_index]])

            self.SA_modules.append(
                build_sa_module(
                    num_point=num_points[sa_index],
                    radii=radii[sa_index],
                    sample_nums=num_samples[sa_index],
                    mlp_channels=cur_sa_mlps,
                    fps_mod=cur_fps_mod,
                    fps_sample_range_list=cur_fps_sample_range_list,
                    dilated_group=dilated_group[sa_index],
                    norm_cfg=norm_cfg,
                    cfg=sa_cfg,
                    bias=True))
            skip_channel_list.append(sa_out_channel)



            # build the attention layers
            # 需要写一个attention module 以及相应的builder
            if att_cfg is not None:
                self.attention_layers.append(
                    build_attention_module(
                        cfg=att_cfg,
                        img_channel=att_img_inchannel[sa_index],
                        pts_channel=att_pts_inchannel[sa_index],
                        out_channel=att_out_channel[sa_index]
                    )

                )

            cur_aggregation_channel = aggregation_channels[sa_index]
            if cur_aggregation_channel is None:
                self.aggregation_mlps.append(None)
                sa_in_channel = sa_out_channel
            else:
                self.aggregation_mlps.append(
                    ConvModule(
                        sa_out_channel,
                        cur_aggregation_channel,
                        conv_cfg=dict(type='Conv1d'),
                        norm_cfg=dict(type='BN1d'),
                        kernel_size=1,
                        bias=True))
                sa_in_channel = cur_aggregation_channel

    @auto_fp16(apply_to=('points', 'img_feats'))
    def forward(self,
                points,
                img_feats=None,
                img_metas=None):
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).
                img_feats:img_backbone返回的图像特征列表，包含５个level的特征图
                img_metas：图像信息，是一个列表，包含ｂａｔｃｈ_size个列表
        Returns:
            dict[str, torch.Tensor]: Outputs of the last SA module.

                - sa_xyz (torch.Tensor): The coordinates of sa features.
                - sa_features (torch.Tensor): The features from the
                    last Set Aggregation Layers.
                - sa_indices (torch.Tensor): Indices of the \
                    input points.
        """
        xyz, features = self._split_point_feats(points)
        # 使用前三层特征
        img_feat_ = img_feats[0:3]
        batch, num_points = xyz.shape[:2]
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(
            batch, 1).long()

        sa_xyz = [xyz]
        sa_features = [features]
        sa_indices = [indices]

        out_sa_xyz = [xyz]
        out_sa_features = [features]
        out_sa_indices = [indices]

        for i in range(self.num_sa):
          # sa_xyz sa_feature　每一次的输入都是上一次的输出
            cur_xyz, cur_features, cur_indices = self.SA_modules[i](
                sa_xyz[i], sa_features[i])
            """
              get the pts_img_feature
                input: img_feats_ list(3[B,256,h,w])
                        img_meats list(B)
                        cur_xyz tensor(B,N,3)
                output:pts_img_feats tensor(B,N,C)
            """
            pts_img_feats = self.fusion_layer(img_feats=img_feat_[i],
                                              pts=cur_xyz,
                                              pts_feats=None,
                                              img_metas=img_metas)
            # print("$"*20)
            # print("pts_img_feats shape:",pts_img_feats.shape)
            if self.aggregation_mlps[i] is not None:
                cur_features = self.aggregation_mlps[i](cur_features)

            # print("@"*30)
            # print(cur_features.shape)
            # get the fusion feature by attention layer
            """
                input: pts_img_feats tensor(B,N,C) 点云对应的图像特征
                        cur_features tensor(B,C,N)　点云的特征
                output: fusion_feats tensor(B,C,N)
            """
            fusion_feats = self.attention_layers[i](
                point_feas=cur_features,
                img_feas=pts_img_feats)

            sa_xyz.append(cur_xyz)
            sa_features.append(fusion_feats)
            sa_indices.append(
                torch.gather(sa_indices[-1], 1, cur_indices.long()))
            if i in self.out_indices:
                out_sa_xyz.append(sa_xyz[-1])
                out_sa_features.append(sa_features[-1])
                out_sa_indices.append(sa_indices[-1])
        # print(len(out_sa_xyz),"and",len(out_sa_features))
        # print("out_sa_xyz[0]",out_sa_xyz[0].shape)
        # print("out_sa_features",out_sa_features[1].shape)
        return dict(
            sa_xyz=out_sa_xyz,
            sa_features=out_sa_features,
            sa_indices=out_sa_indices)
