import torch
from mmcv.cnn import ConvModule
from torch import nn as nn
from torch.nn import functional as F
from typing import List
from mmdet3d.ops import GroupAll, Points_Sampler, QueryAndGroup, gather_points
from .builder import ATTENTION_MODULES


@ATTENTION_MODULES.register_module()
class Attention_Module(nn.Module):
    """
    １．

    Args:

    """

    def __init__(self,
                 img_channel: int,
                 pts_channel: int,
                 out_channel: int):
        super(Attention_Module, self).__init__()
        self.img_channel = img_channel
        self.pts_channel = pts_channel
        self.out_channel = out_channel

        # 构建最终融合输出层
        self.conv1 = torch.nn.Conv1d(pts_channel*2, out_channel, 1)
        self.bn1 = torch.nn.BatchNorm1d(out_channel)

        self.ic = img_channel
        self.pc = pts_channel
        self.line_out = self.pc // 2
        # self.conv2 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
        #                            nn.BatchNorm1d(self.pc),
        #                            nn.ReLU())
        # img_feats channel to [32,64,128]
        self.fc1 = nn.Linear(self.ic, self.line_out)
        # pts_feats channel to [32,64,128]
        self.fc2 = nn.Linear(self.pc, self.line_out)
        # output the attention factors
        self.fc3 = nn.Linear(self.line_out, 1)
    def forward(
        self,
        point_feas,
        img_feas):

        """

        Args:
            point_feas: 点云特征(B,C,N)
            img_feas: 图像特征(B,N,C)

        Returns:
            fusion_feats: 融合后的特征(B,C,N) 和aggregation阶段对应
        """

        batch = img_feas.size(0)
        img_out_f = img_feas.transpose(1, 2).contiguous()  # BNC->BCN
        # print("img_out_f shape:",img_out_f.shape)
        img_feas_f = img_feas.view(-1, self.ic)  # BCN->BNC->(BN)C
        # print("img_feas_f.shape", img_feas_f.shape)
        point_feas_f = point_feas.transpose(1, 2).contiguous().view(-1, self.pc)  # BCN->BNC->(BN)C'
        # print("point_feas_f shape:", point_feas_f.shape)
        ri = self.fc1(img_feas_f)
        rp = self.fc2(point_feas_f)
        # print("ri rp:", ri.shape, rp.shape)
        att = F.sigmoid(self.fc3(F.tanh(ri + rp)))  # BNx1
        # print("att1:", att.shape)
        att = att.squeeze(1)
        # print("att2:", att.shape)
        att = att.view(batch, 1, -1)  # B1N
        # print("att3:", att.shape)
        # print("att 模块输出：")
        # print(att)
        # img_feas * att
        # img_feas_new = self.conv2(img_out_f)
        img_feats_att = img_out_f * att

        # fusion_feats BCN
        fusion_feats = torch.cat([point_feas, img_feats_att], dim=1)
        fusion_feats = F.relu(self.bn1(self.conv1(fusion_feats)))

        return fusion_feats


