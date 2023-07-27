# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from utils.lovasz_softmax import LovaszSoftmax

class Criterion_pose(nn.Module):
    def __init__(self, out_len=1, use_target_weight=False):
        super(Criterion_pose, self).__init__()
        self.criterion = JointsMSELoss(use_target_weight).cuda()
        self.use_target_weight = use_target_weight
        self.lamda = nn.Parameter(-2.5 * torch.ones(out_len))
        # self.lamda2 = nn.Parameter(2.3 * torch.ones(2))
        

    # def joint_loss(self, output, target, target_weight=None):

    #     if isinstance(output, list):
    #         loss_pose = self.criterion(output[0], target[0].cuda(non_blocking=True), target_weight[0].cuda(non_blocking=True))
    #         for output in output[1:]:
    #             loss_pose += self.criterion(output, target[1].cuda(non_blocking=True), target_weight[1].cuda(non_blocking=True))
    #     else:
    #         loss_pose = self.criterion(output, target, target_weight)

        
    #     return loss_pose

    def forward(self, output, target, target_weight=None):
        loss = 0.

        if isinstance(output, list):
            #weights = [1, 1, 1, 1, 1, 1, 1]
            # weights = [(i+1)*(i+1) for i in range(len(output))]
            # sum_w = sum(weights)
            # for i in range(len(output)):
            #    weights[i] = weights[i]/sum_w
            for i in range(len(output)):
                pred = output[i]
                # loss+=self.joint_loss(pred,target)*weights[i]
                loss += self.criterion(pred, target, target_weight) * torch.exp(-self.lamda[i]) + self.lamda[i]
        else:
            loss += self.criterion(output, target, target_weight) * torch.exp(-self.lamda) + self.lamda
        return loss


class Criterion_par(nn.Module):
    def __init__(self, out_len=1, ignore_index=255,
                 thres=0.9, min_kept=131072):
        super(Criterion_par, self).__init__()
        self.ignore_index = ignore_index
        self.criterion_parsing = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255)
        self.lovasz = LovaszSoftmax(ignore_index=255)
        # self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.lamda = nn.Parameter(2.3 * torch.ones(out_len))

    def parsing_loss(self, parsing, label_parsing):
        h, w = label_parsing[0].size(1), label_parsing[0].size(2)
        scale_pred = F.interpolate(input=parsing, size=(h, w),
                                           mode='bilinear', align_corners=True)
        loss_parsing = 0.5  * self.lovasz(scale_pred, label_parsing)+0.5 * self.criterion_parsing(scale_pred, label_parsing)


        return loss_parsing

    def forward(self, preds, target):
        loss = 0.
        if isinstance(preds, list):
            weights = [1, 1, 1, 1, 1, 1]
            # weights = [(i+1)*(i+1) for i in range(len(preds))]
            # sum_w = sum(weights)
            # for i in range(len(preds)):
            #    weights[i] = weights[i]/sum_w
            for i in range(len(preds)):
                # loss += self.parsing_loss(preds[i], target)*weights[i]
                loss += self.parsing_loss(preds[i], target) * torch.exp(-self.lamda[i]) + self.lamda[i]
        else:
            loss += self.parsing_loss(preds, target) * torch.exp(-self.lamda) + self.lamda
        return loss 

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)
