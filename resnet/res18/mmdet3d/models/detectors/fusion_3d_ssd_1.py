from mmdet.models import DETECTORS
from .votenet import VoteNet
from .. import builder
import torch
import torch.nn as nn
from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)

import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from os import path as osp
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)

@DETECTORS.register_module()
class FusionSSD3DNet(VoteNet):
    """3DSSDNet model.
    single stage detector + image attention fusion
    add new:
    img_backbone:the backbone of the image
    backbone:
        fusion_layer:Pointfusion ,return the image feature related to each sample points
        attention_layer:generater the attention factor for the fusion
    https://arxiv.org/abs/2002.10187.pdf
    """

    def __init__(self,
                 backbone=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 img_backbone=None,
                 img_neck=None):
        super(FusionSSD3DNet, self).__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.img_backbone=builder.build_backbone(img_backbone)
        self.img_neck=builder.build_neck(img_neck)

    def init_weights(self, pretrained=None):
        """Initialize weights of detector."""
        super(FusionSSD3DNet, self).init_weights(pretrained)
        if pretrained is None:
            img_pretrained = None
            pts_pretrained = None
        elif isinstance(pretrained,dict):
            img_pretrained =  pretrained.get('img',None)
            pts_pretrained = pretrained.get('pts',None)
        else:
            raise ValueError(
                f'pretrained should be a dict, got {type(pretrained)}'
            )
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()
        # img_backbone init
        if self.with_img_backbone:
            self.img_backbone.init_weights(pretrained=img_pretrained)
        # img_neck init
        if self.with_img_neck:
            if isinstance(self.img_neck, nn.Sequential):
                for m in self.img_neck:
                    m.init_weights()
            else:
                self.img_neck.init_weights()

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None


    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_img_rpn(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_rpn') and self.img_rpn is not None

    @property
    def with_img_bbox(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_bbox') and self.img_bbox is not None

    def extract_img_feat(self, img, img_metas):
        """提取图像特征，得到图像特征列表"""
        input_shape = img.shape[-2:]
        # update real input shape of each single img
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        if img.dim() == 5 and img.size(0) == 1:
            # squeeze只会去除维度为１的维度，这里是去除Ｂ
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)
        img_feat = self.img_backbone(img)
        if self.with_img_neck:
            img_feat = self.img_neck(img_feat)
        return img_feat

    def extract_pts_feat(self, points, img_feat, img_metas):
        """提取点云特征"""
        pts_feat = self.backbone(points,img_feat, img_metas)
        return pts_feat

    def extract_feats_fusion(self, points, img, img_metas):
        """建立两个backbone之间的关系"""
        img_feat = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feat, img_metas)
        return (img_feat, pts_feats)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      proposals=None,
                      img=None,
                      gt_bboxes_ignore=None):
        """
        Forward_train function,
        Args:
            points: 每一帧点云（１６３８４）
            img_metas: 图像标签信息
            gt_bboxes_3d: ３d bbox
            gt_labels_3d: 3d label
            gt_labels: img label
            gt_bboxes: img bbox
            proposals:
            gt_bboxes_ignore:
            img:image
        Returns:
            training loss

        """
        points_cat = torch.stack(points)
        img_feats, pts_feats=self.extract_feats_fusion(points_cat,img_metas=img_metas,img=img)
        losses = dict()
        if(pts_feats):
            losses_pts = self.forward_pts_train(points=points,
                                                pts_feats=pts_feats,
                                                gt_bboxes_3d=gt_bboxes_3d,
                                                gt_labels_3d=gt_labels_3d,
                                                img_metas=img_metas,
                                                gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(img_feats,
                                                img_metas,
                                                gt_labels,
                                                gt_bboxes,
                                                gt_bboxes_ignore,
                                                proposals)
            losses.update(losses_img)
        return losses



    def forward_pts_train(self,
                          points,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          pts_semantic_mask=None,
                          pts_instance_mask=None,
                          gt_bboxes_ignore=None):
        """

        Args:
            pts_feats:
            gt_bboxes_3d:
            gt_labels_3d:
            img_metas:
            gt_bboxes_ignore:

        Returns:
            pts_loss dict
        """
        # loss input is bbox_preds by head
        bbox_preds = self.bbox_head(pts_feats, self.train_cfg.sample_mod)
        loss_inputs = (points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask,
                       pts_instance_mask, img_metas)
        losses = self.bbox_head.loss(
            bbox_preds, *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward_img_train(self,
                          img_feats,
                          img_metas,
                          gt_labels,
                          gt_bboxes,
                          gt_bboxes_ignore=None,
                          proposals=None,
                          **kwargs):
        """

        Args:
            img_feats:
            img_metas:
            gt_labels:
            gt_bboxes:
            gt_bboxes_ignore:
            proposals:
            **kwargs:

        Returns:
            loss of image branch

        """
        losses = dict()
        # RPN forward and loss
        if self.with_img_rpn:
            rpn_outs = self.img_rpn_head(img_feats)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas,
                                          self.train_cfg.img_rpn)
            rpn_losses = self.img_rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('img_rpn_proposal',
                                              self.test_cfg.img_rpn)
            proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
            proposal_list = self.img_rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # bbox head forward and loss
        if self.with_img_bbox:
            # bbox head forward and loss
            img_roi_losses = self.img_roi_head.forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_labels,
                gt_bboxes_ignore, **kwargs)
            losses.update(img_roi_losses)

        return losses


    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list): Image metas.
            rescale (bool): Whether to rescale results.
            imgs: image of the sample
        Returns:
            list: Predicted 3d boxes.
        """
        points_cat = torch.stack(points)
        _, x = self.extract_feats_fusion(points_cat, img=img, img_metas=img_metas)
        # x = self.extract_feat(points_cat)
        bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
        bbox_list = self.bbox_head.get_bboxes(
            points_cat, bbox_preds, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test with augmentation."""
        points_cat = [torch.stack(pts) for pts in points]
        _, feats = self.extract_feats_fusion(points_cat, img=imgs, img_metas=img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, pts_cat, img_meta in zip(feats, points_cat, img_metas):
            bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
            bbox_list = self.bbox_head.get_bboxes(
                pts_cat, bbox_preds, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]

    # def show_results(self, data, result, out_dir):
    #     """Results visualization.

    #     Args:
    #         data (dict): Input points and the information of the sample.
    #         result (dict): Prediction results.
    #         out_dir (str): Output directory of visualization result.
    #     """
    #     for batch_id in range(len(result)):
    #         if isinstance(data['points'][0], DC):
    #             points = data['points'][0]._data[0][batch_id].numpy()
    #         elif mmcv.is_list_of(data['points'][0], torch.Tensor):
    #             points = data['points'][0][batch_id]
    #         else:
    #             ValueError(f"Unsupported data type {type(data['points'][0])} "
    #                        f'for visualization!')
    #         if isinstance(data['img_metas'][0], DC):
    #             pts_filename = data['img_metas'][0]._data[0][batch_id][
    #                 'pts_filename']
    #             box_mode_3d = data['img_metas'][0]._data[0][batch_id][
    #                 'box_mode_3d']
    #         elif mmcv.is_list_of(data['img_metas'][0], dict):
    #             pts_filename = data['img_metas'][0][batch_id]['pts_filename']
    #             box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
    #         else:
    #             ValueError(
    #                 f"Unsupported data type {type(data['img_metas'][0])} "
    #                 f'for visualization!')
    #         file_name = osp.split(pts_filename)[-1].split('.')[0]

    #         assert out_dir is not None, 'Expect out_dir, got none.'
    #         inds = result[batch_id]['pts_bbox']['scores_3d'] > 0.1
    #         pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d'][inds]

    #         # for now we convert points and bbox into depth mode
    #         if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d
    #                                               == Box3DMode.LIDAR):
    #             points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
    #                                                Coord3DMode.DEPTH)
    #             pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d,
    #                                             Box3DMode.DEPTH)
    #         elif box_mode_3d != Box3DMode.DEPTH:
    #             ValueError(
    #                 f'Unsupported box_mode_3d {box_mode_3d} for convertion!')

    #         pred_bboxes = pred_bboxes.tensor.cpu().numpy()
    #         show_result(points, None, pred_bboxes, out_dir, file_name)