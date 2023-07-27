import numpy as np
import cv2
import os
import random

import json
import logging
from collections import OrderedDict
import argparse
from PIL import Image as PILImage
from utils.transforms import transform_parsing
logger = logging.getLogger(__name__)
LABELS = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', \
          'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg',
          'Right-leg', 'Left-shoe', 'Right-shoe']

part_dict = {
        'dress':6,
        'skirt':12,
        'scarf':11,
        'sunglass':4
        }

def get_lip_palette():
    palette = [0,0,0,
            128,0,0,
            255,0,0,
            0,85,0,
            170,0,51,
            255,85,0,
            0,0,85,
            0,119,221,
            85,85,0,
            0,85,85,
            85,51,0,
            52,86,128,
            0,128,0,
            0,0,255,
            51,170,221,
            0,255,255,
            85,255,170,
            170,255,85,
            255,255,0,
            255,170,0]
    return palette

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def get_confusion_matrix(gt_label, pred_label, num_classes):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param num_classes: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * num_classes + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i_label in range(num_classes):
        for i_pred_label in range(num_classes):
            cur_index = i_label * num_classes + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix

def compute_mean_ioU_head(preds, scales, centers, num_classes, datadir, input_size=[473, 473], dataset='val'):
    list_path = os.path.join(datadir, dataset + '_id.txt')
    val_id = [i_id.strip() for i_id in open(list_path)]
    num_classes_head = 5
    num_classes_body = 14
    confusion_matrix = np.zeros((num_classes, num_classes))
    confusion_matrix_head = np.zeros((num_classes_head, num_classes_head))
    confusion_matrix_body = np.zeros((num_classes_body, num_classes_body))
    palette = get_palette(18)

    for i, im_name in enumerate(val_id):
        gt_path = os.path.join(datadir, dataset + '_segmentations', im_name + '.png')
        
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        h, w = gt.shape
        pred_out = preds[i]
        s = scales[i]
        c = centers[i]
        pred = transform_parsing(pred_out, c, s, 0, w, h, input_size)

        gt = np.asarray(gt, dtype=np.int32)
        pred_head = np.asarray(pred, dtype=np.int32)
        pred_save = np.array(pred, dtype=np.uint8)
        gt_head = np.zeros(pred_save.shape).astype('uint8')
        gt_body = np.zeros(pred_save.shape).astype('uint8')
        #pred_head = np.zeros(pred_save.shape).astype('uint8')
        #pred_body = np.zeros(pred_save.shape).astype('uint8')
        st_head = {0:0,1:1,2:2,3:3,11:4,255:255}
        # st = {0:0,1:4,2:5,3:6,4:7,5:8,6:9,7:10,8:12,9:13,10:14,11:15,12:16,13:17,255:255}
        # st_body = {0:0,4:1,5:2,6:3,7:4,8:5,9:6,10:7,12:8,13:9,14:10,15:11,16:12,17:13,255:255}
        # for kk in  [4,5,6,7,8,9,10,12,13,14,15,16,17,255]:
        #     if kk in np.unique(gt):
        #        for ii, jj in zip(*np.where((gt == kk))):
        #            gt_body[ii, jj] =st_body[kk]

        # for kk in  [4,5,6,7,8,9,10,12,13,14,15,16,17,255]:
        #     if kk in np.unique(pred):
        #        for ii, jj in zip(*np.where((pred == kk))):
        #            pred_body[ii, jj] =st_body[kk]

        for kk in  [1,2,3,11,255]:
            if kk in np.unique(gt):
               for ii, jj in zip(*np.where((gt == kk))):
                   gt_head[ii, jj] =st_head[kk] 

        for kk in  [1,2,3,11,255]:
            if kk in np.unique(pred):
               for ii, jj in zip(*np.where((pred == kk))):
                   pred_head[ii, jj] =st_head[kk]              
        
        gd_save = np.array(gt, dtype=np.uint8)
      
        


        # output_im = PILImage.fromarray(pred_save)
        # output_im.putpalette(palette)
        # if not os.path.exists('/home/tracy/concat/scripts/CE2P_head_body_new2_atr_body/'):
        #     os.mkdir('/home/tracy/concat/scripts/CE2P_head_body_new2_atr_body/')
        # output_im.save('/home/tracy/concat/scripts/CE2P_head_body_new2_atr_body/' + im_name + '.png')

        # output_im = PILImage.fromarray(gd_save)
        # output_im.putpalette(palette)
        # if not os.path.exists('/home/tracy/concat/scripts/gd_atr_all/'):
        #     os.mkdir('/home/tracy/concat/scripts/gd_atr_all/')
        # output_im.save('/home/tracy/concat/scripts/gd_atr_all/' + im_name + '.png')



         
        ignore_index = gt != 255


        gt_head = gt_head[ignore_index]
        pred_head = pred_head[ignore_index]


        confusion_matrix_head += get_confusion_matrix(gt_head, pred_head, num_classes_head)

 

    pos = confusion_matrix_head.sum(1)
    res = confusion_matrix_head.sum(0)
    tp = np.diag(confusion_matrix_head)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()

    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)
    name_value = []

    pos = confusion_matrix_head.sum(1)  # TP + FP
    res = confusion_matrix_head.sum(0)  # p
    tp = np.diag(confusion_matrix_head)

    pixel_accuracy = tp.sum() / pos.sum()  # mean Acc
    fg_accuracy = tp[1:].sum() / pos[1:].sum()  # foreground Acc
    mean_accuracy = (tp / np.maximum(1.0, pos)).mean()  # mean Precision
    # cal_list = 0
    # for i in range(len(res)):
    #     if res[i] != 0:
    #         cal_list += (tp[i] / res[i])
    # mean_recall = cal_list / len(res)  # mean Recall
    mean_recall = (tp / res).mean()  # mean Recall
    f1_score = 2 * mean_accuracy * mean_recall / (mean_accuracy + mean_recall)
    accuracy = (tp / np.maximum(1.0, pos))
    recall = (tp / res)
    cls_f1_score = 2 * accuracy * recall / (accuracy + recall)

    print('pixelAcc. --> {}'.format(pixel_accuracy))
    print('foregroundAcc. --> {}'.format(fg_accuracy))
    print('meanPrecision --> {}'.format(mean_accuracy))
    print('meanRecall --> {}'.format(mean_recall))
    print('meanF1-score --> {}'.format(f1_score))
    for i in range(num_classes_head):
        print('cls {}, F1-score --> {}'.format(i, cls_f1_score[i]))

    for i, (label, iou) in enumerate(zip(LABELS, IoU_array)):
        name_value.append((label, iou))

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value)
  

    return name_value

def compute_mean_ioU_body(preds, scales, centers, num_classes, datadir, input_size=[473, 473], dataset='val'):
    list_path = os.path.join(datadir, dataset + '_id.txt')
    val_id = [i_id.strip() for i_id in open(list_path)]
    num_classes_head = 5
    num_classes_body = 14
    confusion_matrix = np.zeros((num_classes, num_classes))
    confusion_matrix_head = np.zeros((num_classes_head, num_classes_head))
    confusion_matrix_body = np.zeros((num_classes_body, num_classes_body))
    palette = get_palette(18)

    for i, im_name in enumerate(val_id):
        gt_path = os.path.join(datadir, dataset + '_segmentations', im_name + '.png')
        
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        h, w = gt.shape
        pred_out = preds[i]
        s = scales[i]
        c = centers[i]
        pred = transform_parsing(pred_out, c, s, 0, w, h, input_size)

        gt = np.asarray(gt, dtype=np.int32)
        pred_body = np.asarray(pred, dtype=np.int32)
        pred_save = np.array(pred, dtype=np.uint8)
        gt_head = np.zeros(pred_save.shape).astype('uint8')
        gt_body = np.zeros(pred_save.shape).astype('uint8')
        #pred_head = np.zeros(pred_save.shape).astype('uint8')
        #pred_body = np.zeros(pred_save.shape).astype('uint8')
        st_head = {0:0,1:1,2:2,3:3,11:4,255:255}
        # st = {0:0,1:4,2:5,3:6,4:7,5:8,6:9,7:10,8:12,9:13,10:14,11:15,12:16,13:17,255:255}
        st_body = {0:0,4:1,5:2,6:3,7:4,8:5,9:6,10:7,12:8,13:9,14:10,15:11,16:12,17:13,255:255}
        for kk in  [4,5,6,7,8,9,10,12,13,14,15,16,17,255]:
            if kk in np.unique(gt):
               for ii, jj in zip(*np.where((gt == kk))):
                   gt_body[ii, jj] =st_body[kk]

        # for kk in  [4,5,6,7,8,9,10,12,13,14,15,16,17,255]:
        #     if kk in np.unique(pred):
        #        for ii, jj in zip(*np.where((pred == kk))):
        #            pred_body[ii, jj] =st_body[kk]

      
        


        # output_im = PILImage.fromarray(pred_save)
        # output_im.putpalette(palette)
        # if not os.path.exists('/home/tracy/concat/scripts/CE2P_head_body_new2_atr_body/'):
        #     os.mkdir('/home/tracy/concat/scripts/CE2P_head_body_new2_atr_body/')
        # output_im.save('/home/tracy/concat/scripts/CE2P_head_body_new2_atr_body/' + im_name + '.png')

        # output_im = PILImage.fromarray(gd_save)
        # output_im.putpalette(palette)
        # if not os.path.exists('/home/tracy/concat/scripts/gd_atr_all/'):
        #     os.mkdir('/home/tracy/concat/scripts/gd_atr_all/')
        # output_im.save('/home/tracy/concat/scripts/gd_atr_all/' + im_name + '.png')



         
        ignore_index = gt != 255


        gt_body = gt_body[ignore_index]
        pred_body = pred_body[ignore_index]


        confusion_matrix_body += get_confusion_matrix(gt_body, pred_body, num_classes_body)

 

    pos = confusion_matrix_body.sum(1)
    res = confusion_matrix_body.sum(0)
    tp = np.diag(confusion_matrix_body)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()

    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)
    name_value = []

    pos = confusion_matrix_body.sum(1)  # TP + FP
    res = confusion_matrix_body.sum(0)  # p
    tp = np.diag(confusion_matrix_body)

    pixel_accuracy = tp.sum() / pos.sum()  # mean Acc
    fg_accuracy = tp[1:].sum() / pos[1:].sum()  # foreground Acc
    mean_accuracy = (tp / np.maximum(1.0, pos)).mean()  # mean Precision
    # cal_list = 0
    # for i in range(len(res)):
    #     if res[i] != 0:
    #         cal_list += (tp[i] / res[i])
    # mean_recall = cal_list / len(res)  # mean Recall
    mean_recall = (tp / res).mean()  # mean Recall
    f1_score = 2 * mean_accuracy * mean_recall / (mean_accuracy + mean_recall)
    accuracy = (tp / np.maximum(1.0, pos))
    recall = (tp / res)
    cls_f1_score = 2 * accuracy * recall / (accuracy + recall)

    print('pixelAcc. --> {}'.format(pixel_accuracy))
    print('foregroundAcc. --> {}'.format(fg_accuracy))
    print('meanPrecision --> {}'.format(mean_accuracy))
    print('meanRecall --> {}'.format(mean_recall))
    print('meanF1-score --> {}'.format(f1_score))
    for i in range(num_classes_body):
        print('cls {}, F1-score --> {}'.format(i, cls_f1_score[i]))

    for i, (label, iou) in enumerate(zip(LABELS, IoU_array)):
        name_value.append((label, iou))

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value)
  

    return name_value

def compute_mean_ioU2(preds, scales, centers, num_classes, datadir, input_size=[473, 473], dataset='val'):
    list_path = os.path.join(datadir, dataset + '_id.txt')
    val_id = [i_id.strip() for i_id in open(list_path)]
    num_classes_head = 5
    num_classes_body = 14
    confusion_matrix = np.zeros((num_classes, num_classes))
    confusion_matrix_head = np.zeros((num_classes_head, num_classes_head))
    confusion_matrix_body = np.zeros((num_classes_body, num_classes_body))
    palette = get_palette(18)

    for i, im_name in enumerate(val_id):
        gt_path = os.path.join(datadir, dataset + '_segmentations', im_name + '.png')
        
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        h, w = gt.shape
        pred_out = preds[i]
        s = scales[i]
        c = centers[i]
        pred = transform_parsing(pred_out, c, s, 0, w, h, input_size)

        gt = np.asarray(gt, dtype=np.int32)
        pred = np.asarray(pred, dtype=np.int32)
        pred_save = np.array(pred, dtype=np.uint8)
        gt_head = np.zeros(pred_save.shape).astype('uint8')
        gt_body = np.zeros(pred_save.shape).astype('uint8')
        pred_head = np.zeros(pred_save.shape).astype('uint8')
        pred_body = np.zeros(pred_save.shape).astype('uint8')
        st_head = {0:0,1:1,2:2,3:3,11:4,255:255}
        # st = {0:0,1:4,2:5,3:6,4:7,5:8,6:9,7:10,8:12,9:13,10:14,11:15,12:16,13:17,255:255}
        st_body = {0:0,4:1,5:2,6:3,7:4,8:5,9:6,10:7,12:8,13:9,14:10,15:11,16:12,17:13,255:255}
        for kk in  [4,5,6,7,8,9,10,12,13,14,15,16,17,255]:
            if kk in np.unique(gt):
               for ii, jj in zip(*np.where((gt == kk))):
                   gt_body[ii, jj] =st_body[kk]

        for kk in  [4,5,6,7,8,9,10,12,13,14,15,16,17,255]:
            if kk in np.unique(pred):
               for ii, jj in zip(*np.where((pred == kk))):
                   pred_body[ii, jj] =st_body[kk]

        for kk in  [1,2,3,11,255]:
            if kk in np.unique(gt):
               for ii, jj in zip(*np.where((gt == kk))):
                   gt_head[ii, jj] =st_head[kk] 

        for kk in  [1,2,3,11,255]:
            if kk in np.unique(pred):
               for ii, jj in zip(*np.where((pred == kk))):
                   pred_head[ii, jj] =st_head[kk]              
        
        gd_save = np.array(gt, dtype=np.uint8)
      
        


        # output_im = PILImage.fromarray(pred_save)
        # output_im.putpalette(palette)
        # if not os.path.exists('/home/tracy/concat/scripts/CE2P_head_body_new2_atr_body/'):
        #     os.mkdir('/home/tracy/concat/scripts/CE2P_head_body_new2_atr_body/')
        # output_im.save('/home/tracy/concat/scripts/CE2P_head_body_new2_atr_body/' + im_name + '.png')

        # output_im = PILImage.fromarray(gd_save)
        # output_im.putpalette(palette)
        # if not os.path.exists('/home/tracy/concat/scripts/gd_atr_all/'):
        #     os.mkdir('/home/tracy/concat/scripts/gd_atr_all/')
        # output_im.save('/home/tracy/concat/scripts/gd_atr_all/' + im_name + '.png')



         
        ignore_index = gt != 255

        gt = gt[ignore_index]
        pred = pred[ignore_index]

        gt_head = gt_head[ignore_index]
        pred_head = pred_head[ignore_index]

        gt_body = gt_body[ignore_index]
        pred_body = pred_body[ignore_index]

        confusion_matrix += get_confusion_matrix(gt, pred, num_classes)
        confusion_matrix_head += get_confusion_matrix(gt_head, pred_head, num_classes_head)
        confusion_matrix_body += get_confusion_matrix(gt_body, pred_body, num_classes_body)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()

    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)
    name_value = []

    pos = confusion_matrix.sum(1)  # TP + FP
    res = confusion_matrix.sum(0)  # p
    tp = np.diag(confusion_matrix)

    pixel_accuracy = tp.sum() / pos.sum()  # mean Acc
    fg_accuracy = tp[1:].sum() / pos[1:].sum()  # foreground Acc
    mean_accuracy = (tp / np.maximum(1.0, pos)).mean()  # mean Precision
    # cal_list = 0
    # for i in range(len(res)):
    #     if res[i] != 0:
    #         cal_list += (tp[i] / res[i])
    # mean_recall = cal_list / len(res)  # mean Recall
    mean_recall = (tp / res).mean()  # mean Recall
    f1_score = 2 * mean_accuracy * mean_recall / (mean_accuracy + mean_recall)
    accuracy = (tp / np.maximum(1.0, pos))
    recall = (tp / res)
    cls_f1_score = 2 * accuracy * recall / (accuracy + recall)

    print('pixelAcc. --> {}'.format(pixel_accuracy))
    print('foregroundAcc. --> {}'.format(fg_accuracy))
    print('meanPrecision --> {}'.format(mean_accuracy))
    print('meanRecall --> {}'.format(mean_recall))
    print('meanF1-score --> {}'.format(f1_score))
    for i in range(num_classes):
        print('cls {}, F1-score --> {}'.format(i, cls_f1_score[i]))

    for i, (label, iou) in enumerate(zip(LABELS, IoU_array)):
        name_value.append((label, iou))

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value)

    pos = confusion_matrix_head.sum(1)
    res = confusion_matrix_head.sum(0)
    tp = np.diag(confusion_matrix_head)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()

    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)
    name_value = []

    pos = confusion_matrix_head.sum(1)  # TP + FP
    res = confusion_matrix_head.sum(0)  # p
    tp = np.diag(confusion_matrix_head)

    pixel_accuracy = tp.sum() / pos.sum()  # mean Acc
    fg_accuracy = tp[1:].sum() / pos[1:].sum()  # foreground Acc
    mean_accuracy = (tp / np.maximum(1.0, pos)).mean()  # mean Precision
    # cal_list = 0
    # for i in range(len(res)):
    #     if res[i] != 0:
    #         cal_list += (tp[i] / res[i])
    # mean_recall = cal_list / len(res)  # mean Recall
    mean_recall = (tp / res).mean()  # mean Recall
    f1_score = 2 * mean_accuracy * mean_recall / (mean_accuracy + mean_recall)
    accuracy = (tp / np.maximum(1.0, pos))
    recall = (tp / res)
    cls_f1_score = 2 * accuracy * recall / (accuracy + recall)

    print('pixelAcc. --> {}'.format(pixel_accuracy))
    print('foregroundAcc. --> {}'.format(fg_accuracy))
    print('meanPrecision --> {}'.format(mean_accuracy))
    print('meanRecall --> {}'.format(mean_recall))
    print('meanF1-score --> {}'.format(f1_score))
    for i in range(num_classes_head):
        print('cls {}, F1-score --> {}'.format(i, cls_f1_score[i]))

    for i, (label, iou) in enumerate(zip(LABELS, IoU_array)):
        name_value.append((label, iou))

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value)

    pos = confusion_matrix_body.sum(1)
    res = confusion_matrix_body.sum(0)
    tp = np.diag(confusion_matrix_body)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()

    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)
    name_value = []

    pos = confusion_matrix_body.sum(1)  # TP + FP
    res = confusion_matrix_body.sum(0)  # p
    tp = np.diag(confusion_matrix_body)

    pixel_accuracy = tp.sum() / pos.sum()  # mean Acc
    fg_accuracy = tp[1:].sum() / pos[1:].sum()  # foreground Acc
    mean_accuracy = (tp / np.maximum(1.0, pos)).mean()  # mean Precision
    # cal_list = 0
    # for i in range(len(res)):
    #     if res[i] != 0:
    #         cal_list += (tp[i] / res[i])
    # mean_recall = cal_list / len(res)  # mean Recall
    mean_recall = (tp / res).mean()  # mean Recall
    f1_score = 2 * mean_accuracy * mean_recall / (mean_accuracy + mean_recall)
    accuracy = (tp / np.maximum(1.0, pos))
    recall = (tp / res)
    cls_f1_score = 2 * accuracy * recall / (accuracy + recall)

    print('pixelAcc. --> {}'.format(pixel_accuracy))
    print('foregroundAcc. --> {}'.format(fg_accuracy))
    print('meanPrecision --> {}'.format(mean_accuracy))
    print('meanRecall --> {}'.format(mean_recall))
    print('meanF1-score --> {}'.format(f1_score))
    for i in range(num_classes_body):
        print('cls {}, F1-score --> {}'.format(i, cls_f1_score[i]))

    for i, (label, iou) in enumerate(zip(LABELS, IoU_array)):
        name_value.append((label, iou))

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value)
    return name_value

def compute_mean_ioU(preds, scales, centers, num_classes, datadir, input_size=[473, 473], dataset='val'):
    list_path = os.path.join(datadir, dataset + '_id.txt')
    val_id = [i_id.strip() for i_id in open(list_path)]
    num_classes_head = 5
    num_classes_body = 14
    confusion_matrix = np.zeros((num_classes, num_classes))
    confusion_matrix_head = np.zeros((num_classes_head, num_classes_head))
    confusion_matrix_body = np.zeros((num_classes_body, num_classes_body))
    palette = get_palette(18)

    for i, im_name in enumerate(val_id):
        gt_path = os.path.join(datadir, dataset + '_segmentations', im_name + '.png')
        
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        h, w = gt.shape
        pred_out = preds[i]
        s = scales[i]
        c = centers[i]
        pred = transform_parsing(pred_out, c, s, 0, w, h, input_size)

        gt = np.asarray(gt, dtype=np.int32)
        pred = np.asarray(pred, dtype=np.int32)
        pred_save = np.array(pred, dtype=np.uint8)
           
        
        gd_save = np.array(gt, dtype=np.uint8)
      
        


        # output_im = PILImage.fromarray(pred_save)
        # output_im.putpalette(palette)
        # if not os.path.exists('/home/tracy/concat/scripts/CE2P_head_body_new2_atr_body/'):
        #     os.mkdir('/home/tracy/concat/scripts/CE2P_head_body_new2_atr_body/')
        # output_im.save('/home/tracy/concat/scripts/CE2P_head_body_new2_atr_body/' + im_name + '.png')

        # output_im = PILImage.fromarray(gd_save)
        # output_im.putpalette(palette)
        # if not os.path.exists('/home/tracy/concat/scripts/gd_atr_all/'):
        #     os.mkdir('/home/tracy/concat/scripts/gd_atr_all/')
        # output_im.save('/home/tracy/concat/scripts/gd_atr_all/' + im_name + '.png')



         
        ignore_index = gt != 255

        gt = gt[ignore_index]
        pred = pred[ignore_index]


        confusion_matrix += get_confusion_matrix(gt, pred, num_classes)


    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()

    logger.info('Pixel accuracy: %f \n' % pixel_accuracy)
    logger.info('Mean accuracy: %f \n' % mean_accuracy)
    logger.info('Mean IU: %f \n' % mean_IoU)
    name_value = []

    pos = confusion_matrix.sum(1)  # TP + FP
    res = confusion_matrix.sum(0)  # p
    tp = np.diag(confusion_matrix)

    pixel_accuracy = tp.sum() / pos.sum()  # mean Acc
    fg_accuracy = tp[1:].sum() / pos[1:].sum()  # foreground Acc
    mean_accuracy = (tp / np.maximum(1.0, pos)).mean()  # mean Precision
    # cal_list = 0
    # for i in range(len(res)):
    #     if res[i] != 0:
    #         cal_list += (tp[i] / res[i])
    # mean_recall = cal_list / len(res)  # mean Recall
    mean_recall = (tp / res).mean()  # mean Recall
    f1_score = 2 * mean_accuracy * mean_recall / (mean_accuracy + mean_recall)
    accuracy = (tp / np.maximum(1.0, pos))
    recall = (tp / res)
    cls_f1_score = 2 * accuracy * recall / (accuracy + recall)

    logger.info('pixelAcc. --> {}'.format(pixel_accuracy))
    logger.info('foregroundAcc. --> {}'.format(fg_accuracy))
    logger.info('meanPrecision --> {}'.format(mean_accuracy))
    logger.info('meanRecall --> {}'.format(mean_recall))
    logger.info('meanF1-score --> {}'.format(f1_score))
    #logging.info('pixelAcc. --> {}'.format(pixel_accuracy))
    for i in range(num_classes_body):
        logger.info('cls {}, F1-score --> {}'.format(LABELS[i], cls_f1_score[i]))

    for i, (label, iou) in enumerate(zip(LABELS, IoU_array)):
        logger.info('cls {}, iou --> {}'.format(label, iou))
        name_value.append((label, iou))

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value)

    return name_value




def compute_mean_ioU_head_body(preds, scales, centers, num_classes, datadir, input_size=[473, 473], dataset='val'):
    list_path = os.path.join(datadir, dataset + '_id.txt')
    val_id = [i_id.strip() for i_id in open(list_path)]
    num_classes_head = 5
    num_classes_body = 14
    confusion_matrix = np.zeros((num_classes, num_classes))
    confusion_matrix_head = np.zeros((num_classes_head, num_classes_head))
    confusion_matrix_body = np.zeros((num_classes_body, num_classes_body))
    palette = get_palette(18)

    for i, im_name in enumerate(val_id):
        gt_path = os.path.join(datadir, dataset + '_segmentations', im_name + '.png')
        
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        h, w = gt.shape
        pred_out = preds[i]
        s = scales[i]
        c = centers[i]
        pred = transform_parsing(pred_out, c, s, 0, w, h, input_size)

        gt = np.asarray(gt, dtype=np.int32)
        pred = np.asarray(pred, dtype=np.int32)
        pred_save = np.array(pred, dtype=np.uint8)
        gt_head = np.zeros(pred_save.shape).astype('uint8')
        gt_body = np.zeros(pred_save.shape).astype('uint8')
        pred_head = np.zeros(pred_save.shape).astype('uint8')
        pred_body = np.zeros(pred_save.shape).astype('uint8')
        st_head = {0:0,1:1,2:2,3:3,11:4,255:255}
        # st = {0:0,1:4,2:5,3:6,4:7,5:8,6:9,7:10,8:12,9:13,10:14,11:15,12:16,13:17,255:255}
        st_body = {0:0,4:1,5:2,6:3,7:4,8:5,9:6,10:7,12:8,13:9,14:10,15:11,16:12,17:13,255:255}
        for kk in  [4,5,6,7,8,9,10,12,13,14,15,16,17,255]:
            if kk in np.unique(gt):
               for ii, jj in zip(*np.where((gt == kk))):
                   gt_body[ii, jj] =st_body[kk]

        for kk in  [4,5,6,7,8,9,10,12,13,14,15,16,17,255]:
            if kk in np.unique(pred):
               for ii, jj in zip(*np.where((pred == kk))):
                   pred_body[ii, jj] =st_body[kk]

        for kk in  [1,2,3,11,255]:
            if kk in np.unique(gt):
               for ii, jj in zip(*np.where((gt == kk))):
                   gt_head[ii, jj] =st_head[kk] 

        for kk in  [1,2,3,11,255]:
            if kk in np.unique(pred):
               for ii, jj in zip(*np.where((pred == kk))):
                   pred_head[ii, jj] =st_head[kk]              
        
        gd_save = np.array(gt, dtype=np.uint8)
      
        


        # output_im = PILImage.fromarray(pred_save)
        # output_im.putpalette(palette)
        # if not os.path.exists('/home/tracy/concat/scripts/CE2P_head_body_new2_atr_body/'):
        #     os.mkdir('/home/tracy/concat/scripts/CE2P_head_body_new2_atr_body/')
        # output_im.save('/home/tracy/concat/scripts/CE2P_head_body_new2_atr_body/' + im_name + '.png')

        # output_im = PILImage.fromarray(gd_save)
        # output_im.putpalette(palette)
        # if not os.path.exists('/home/tracy/concat/scripts/gd_atr_all/'):
        #     os.mkdir('/home/tracy/concat/scripts/gd_atr_all/')
        # output_im.save('/home/tracy/concat/scripts/gd_atr_all/' + im_name + '.png')



         
        ignore_index = gt != 255

        gt = gt[ignore_index]
        pred = pred[ignore_index]

        gt_head = gt_head[ignore_index]
        pred_head = pred_head[ignore_index]

        gt_body = gt_body[ignore_index]
        pred_body = pred_body[ignore_index]

        confusion_matrix += get_confusion_matrix(gt, pred, num_classes)
        confusion_matrix_head += get_confusion_matrix(gt_head, pred_head, num_classes_head)
        confusion_matrix_body += get_confusion_matrix(gt_body, pred_body, num_classes_body)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()

    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)
    name_value = []

    pos = confusion_matrix.sum(1)  # TP + FP
    res = confusion_matrix.sum(0)  # p
    tp = np.diag(confusion_matrix)

    pixel_accuracy = tp.sum() / pos.sum()  # mean Acc
    fg_accuracy = tp[1:].sum() / pos[1:].sum()  # foreground Acc
    mean_accuracy = (tp / np.maximum(1.0, pos)).mean()  # mean Precision
    # cal_list = 0
    # for i in range(len(res)):
    #     if res[i] != 0:
    #         cal_list += (tp[i] / res[i])
    # mean_recall = cal_list / len(res)  # mean Recall
    mean_recall = (tp / res).mean()  # mean Recall
    f1_score = 2 * mean_accuracy * mean_recall / (mean_accuracy + mean_recall)
    accuracy = (tp / np.maximum(1.0, pos))
    recall = (tp / res)
    cls_f1_score = 2 * accuracy * recall / (accuracy + recall)

    print('pixelAcc. --> {}'.format(pixel_accuracy))
    print('foregroundAcc. --> {}'.format(fg_accuracy))
    print('meanPrecision --> {}'.format(mean_accuracy))
    print('meanRecall --> {}'.format(mean_recall))
    print('meanF1-score --> {}'.format(f1_score))
    for i in range(num_classes):
        print('cls {}, F1-score --> {}'.format(i, cls_f1_score[i]))

    for i, (label, iou) in enumerate(zip(LABELS, IoU_array)):
        name_value.append((label, iou))

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value)

    pos = confusion_matrix_head.sum(1)
    res = confusion_matrix_head.sum(0)
    tp = np.diag(confusion_matrix_head)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()

    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)
    name_value = []

    pos = confusion_matrix_head.sum(1)  # TP + FP
    res = confusion_matrix_head.sum(0)  # p
    tp = np.diag(confusion_matrix_head)

    pixel_accuracy = tp.sum() / pos.sum()  # mean Acc
    fg_accuracy = tp[1:].sum() / pos[1:].sum()  # foreground Acc
    mean_accuracy = (tp / np.maximum(1.0, pos)).mean()  # mean Precision
    # cal_list = 0
    # for i in range(len(res)):
    #     if res[i] != 0:
    #         cal_list += (tp[i] / res[i])
    # mean_recall = cal_list / len(res)  # mean Recall
    mean_recall = (tp / res).mean()  # mean Recall
    f1_score = 2 * mean_accuracy * mean_recall / (mean_accuracy + mean_recall)
    accuracy = (tp / np.maximum(1.0, pos))
    recall = (tp / res)
    cls_f1_score = 2 * accuracy * recall / (accuracy + recall)

    print('pixelAcc. --> {}'.format(pixel_accuracy))
    print('foregroundAcc. --> {}'.format(fg_accuracy))
    print('meanPrecision --> {}'.format(mean_accuracy))
    print('meanRecall --> {}'.format(mean_recall))
    print('meanF1-score --> {}'.format(f1_score))
    for i in range(num_classes_head):
        print('cls {}, F1-score --> {}'.format(i, cls_f1_score[i]))

    for i, (label, iou) in enumerate(zip(LABELS, IoU_array)):
        name_value.append((label, iou))

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value)

    pos = confusion_matrix_body.sum(1)
    res = confusion_matrix_body.sum(0)
    tp = np.diag(confusion_matrix_body)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()

    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)
    name_value = []

    pos = confusion_matrix_body.sum(1)  # TP + FP
    res = confusion_matrix_body.sum(0)  # p
    tp = np.diag(confusion_matrix_body)

    pixel_accuracy = tp.sum() / pos.sum()  # mean Acc
    fg_accuracy = tp[1:].sum() / pos[1:].sum()  # foreground Acc
    mean_accuracy = (tp / np.maximum(1.0, pos)).mean()  # mean Precision
    # cal_list = 0
    # for i in range(len(res)):
    #     if res[i] != 0:
    #         cal_list += (tp[i] / res[i])
    # mean_recall = cal_list / len(res)  # mean Recall
    mean_recall = (tp / res).mean()  # mean Recall
    f1_score = 2 * mean_accuracy * mean_recall / (mean_accuracy + mean_recall)
    accuracy = (tp / np.maximum(1.0, pos))
    recall = (tp / res)
    cls_f1_score = 2 * accuracy * recall / (accuracy + recall)

    print('pixelAcc. --> {}'.format(pixel_accuracy))
    print('foregroundAcc. --> {}'.format(fg_accuracy))
    print('meanPrecision --> {}'.format(mean_accuracy))
    print('meanRecall --> {}'.format(mean_recall))
    print('meanF1-score --> {}'.format(f1_score))
    for i in range(num_classes_body):
        print('cls {}, F1-score --> {}'.format(i, cls_f1_score[i]))

    for i, (label, iou) in enumerate(zip(LABELS, IoU_array)):
        name_value.append((label, iou))

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value)
    return name_value


