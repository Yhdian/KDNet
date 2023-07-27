# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.loss import JointsMSELoss,Criterion_par,Criterion_pose
import torch.nn as nn
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from torch.nn import functional as F
from utils.lovasz_softmax import LovaszSoftmax
from utils.miou import compute_mean_ioU
from utils.utils import adjust_learning_rate
import csv
import cv2
import pandas as pd

logger = logging.getLogger(__name__)
criterion_parsing = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255)
lovasz = LovaszSoftmax(ignore_index=255)


def test(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs[1], list):
                output = outputs[1][-1]
            else:
                output = outputs[1]

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped[1], list):
                    output_flipped = outputs_flipped[1][-1]
                else:
                    output_flipped = outputs_flipped[1]

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                #if config.TEST.SHIFT_HEATMAP:
                #    output_flipped[:, :, :, 1:] = \
                #        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5
            
      
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)
            save_hpe_results_to_lip_format(im_name_list=meta['filename'], pose_list=preds, save_path='pred_keypoints_lip_test.csv', eval_num=-1)

def validate_pose_parsing(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    interp = torch.nn.Upsample(size=(384, 384), mode='bilinear', align_corners=True)

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    parsing_preds = np.zeros((num_samples, 384,384),
                             dtype=np.uint8)
    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input,label_parsing, target, target_weight, meta) in enumerate(val_loader):
        # if i<2:
            # compute output
            outputs = model(input)
            if config.TEST.MULTI_SCALE==[1.0] and config.TEST.FLIP_TEST==True:
                
                h, w = label_parsing.size(1), label_parsing.size(2)
                

                if isinstance(outputs[0], list):
                    parsing = outputs[0][-1]
                else:
                    parsing = outputs[0]
            else:
                parsing = multi_scale_testing(model, input, crop_size=[384,384], flip=config.TEST.FLIP_TEST, multi_scales=config.TEST.MULTI_SCALE)

            

            if config.TEST.FLIP_TEST:
                if isinstance(outputs[1], list):
                    output = outputs[1][-1]
                else:
                    output = outputs[1]
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])

                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped[1], list):
                    output_flipped = outputs_flipped[1][-1]
                else:
                    output_flipped = outputs_flipped[1]

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                #if config.TEST.SHIFT_HEATMAP:
                #    output_flipped[:, :, :, 1:] = \
                #        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss_pose = criterion(output, target, target_weight)
            loss = loss_pose

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            scales[idx:idx + num_images, :] = s[:, :]
            centers[idx:idx + num_images, :] = c[:, :]

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)
            save_hpe_results_to_lip_format(im_name_list=meta['filename'], pose_list=preds, save_path='pred_keypoints_lip.csv', eval_num=-1)
            parsing = interp(parsing).data.cpu().numpy()
            parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
            parsing_preds[idx:idx + num_images, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        mIoU = compute_mean_ioU(parsing_preds, scales, centers, 20, config.DATASET.ROOT, config.MODEL.IMAGE_SIZE)
        # miou = mIoU['Mean IU']
        # all.append([epoch,mIoU])
        # jsobj = json.dumps(all)
        # fileobject = open(results_save,'a+')
        # fileobject.write(jsobj)
        # fileobject.close()
        # with open(save_file,'a+') as f:
        #         f.writelines('epoch'+' '+ str(epoch)+' '+'loss'+' '+str(loss.data.cpu().numpy())+' '+'mean IoU'+' '+ str(miou)+'\n')

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc_parsing',
                mIoU['Mean IU'],
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

def validate_pose_parsing_multi(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    interp = torch.nn.Upsample(size=(384, 384), mode='bilinear', align_corners=True)

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    parsing_preds = np.zeros((num_samples, 384,384),
                             dtype=np.uint8)
    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input,label_parsing, target, target_weight, meta) in enumerate(val_loader):
        # if i<2:
            # compute output
            outputs = model(input)
            if config.TEST.MULTI_SCALE==[1.0] and config.TEST.FLIP_TEST==True:
                
                h, w = label_parsing.size(1), label_parsing.size(2)
                

                if isinstance(outputs[0], list):
                    parsing = outputs[0][-1]
                else:
                    parsing = outputs[0]
            else:
                parsing = multi_scale_testing(model, input, crop_size=[384,384], flip=config.TEST.FLIP_TEST, multi_scales=config.TEST.MULTI_SCALE)

            

            if config.TEST.FLIP_TEST:
                if isinstance(outputs[1], list):
                    output = outputs[1][-1]
                else:
                    output = outputs[1]
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])

                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped[1], list):
                    output_flipped = outputs_flipped[1][-1]
                else:
                    output_flipped = outputs_flipped[1]

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                #if config.TEST.SHIFT_HEATMAP:
                #    output_flipped[:, :, :, 1:] = \
                #        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss_pose = criterion(output, target, target_weight)
            loss = loss_pose

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            scales[idx:idx + num_images, :] = s[:, :]
            centers[idx:idx + num_images, :] = c[:, :]

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)
            save_hpe_results_to_lip_format(im_name_list=meta['filename'], pose_list=preds, save_path='pred_keypoints_lip.csv', eval_num=-1)
            parsing = interp(parsing).data.cpu().numpy()
            parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
            parsing_preds[idx:idx + num_images, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        mIoU = compute_mean_ioU(parsing_preds, scales, centers, 20, config.DATASET.ROOT, config.MODEL.IMAGE_SIZE)
        # miou = mIoU['Mean IU']
        # all.append([epoch,mIoU])
        # jsobj = json.dumps(all)
        # fileobject = open(results_save,'a+')
        # fileobject.write(jsobj)
        # fileobject.close()
        # with open(save_file,'a+') as f:
        #         f.writelines('epoch'+' '+ str(epoch)+' '+'loss'+' '+str(loss.data.cpu().numpy())+' '+'mean IoU'+' '+ str(miou)+'\n')

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc_parsing',
                mIoU['Mean IU'],
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    losses_val = 0.0
    ct = 0.0
    with torch.no_grad():
        end = time.time()
        for i, (input,label_parsing, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
        
            if isinstance(outputs, list):
                output = outputs[1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                #if config.TEST.SHIFT_HEATMAP:
                #    output_flipped[:, :, :, 1:] = \
                #        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)
            losses_val += loss.item()
            ct += 1

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)
            save_hpe_results_to_lip_format(im_name_list=meta['filename'], pose_list=preds, save_path='pred_keypoints_lip.csv', eval_num=-1)
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator, losses_val / ct


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def multi_scale_aug(image, base_size = 384,rand_scale=1):
        long_size = np.int(base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        return image



def multi_scale_testing(model, batch_input_im, crop_size=[384,384], flip=False, multi_scales=[1]):
    flip_pairs = [[14,15],[16,17],[18,19]]
    #(14,15,16,17,18,19)
    #flipped_idx = (15, 14, 17, 16, 19, 18)
    if len(batch_input_im.shape) > 4:
        batch_input_im = batch_input_im.squeeze()
    if len(batch_input_im.shape) == 3:
        batch_input_im = batch_input_im.unsqueeze(0)
    interp = torch.nn.Upsample(size=crop_size, mode='bilinear', align_corners=True)
    ms_outputs = []
    for s in multi_scales:
        if s !=1.0:
           interp_im = torch.nn.Upsample(scale_factor=s, mode='bilinear', align_corners=True)
           scaled_im = interp_im(batch_input_im)
        else:
            scaled_im = batch_input_im
        parsing_output = model(scaled_im.cuda())

        if isinstance(parsing_output[0], list):
                output = parsing_output[0][-1]
        else:
                output = parsing_output[0]

        output = interp(output)

        if flip:

                input_flipped = np.flip(scaled_im.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped[0], list):
                    flipped_output = outputs_flipped[0][-1]
                else:
                    flipped_output = outputs_flipped[0]

                
                flipped_output = flip_back(flipped_output.cpu().numpy(),flip_pairs)
                flipped_output = torch.from_numpy(flipped_output.copy()).cuda()
                output_rev = interp(flipped_output)

                output_mean = 0.5*(output+output_rev)
                ms_outputs.append(output_mean)

    
    ms_fused_parsing_output = torch.stack(ms_outputs)
    ms_fused_parsing_output = ms_fused_parsing_output.mean(0)
    return ms_fused_parsing_output

def save_hpe_results_to_lip_format(im_name_list, pose_list, save_path='pred_keypoints_lip.csv', eval_num=-1):
    
    if eval_num > 0:
        num_of_im = eval_num
    else:
        num_of_im = len(im_name_list)
    # im_name_list = eval_im_name_list
    result_list = []
    idx_map_to_lip = [10, 9, 8, 11, 12, 13, 15, 14, 1, 0, 4, 3, 2, 5, 6, 7]
    for ii in range(0, num_of_im):
        single_result = []
        single_result.append(im_name_list[ii])
        for ji in range(0, len(idx_map_to_lip)):
            single_result.append(str(int(pose_list[ii, idx_map_to_lip[ji], 0])))
            single_result.append(str(int(pose_list[ii, idx_map_to_lip[ji], 1])))
        result_list.append(single_result)
    with open(save_path, 'a+',newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in result_list:
            writer.writerow(line)
# if __name__ == '__main__':
   
