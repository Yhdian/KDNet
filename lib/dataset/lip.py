# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json_tricks as json
from collections import OrderedDict
import sys
sys.path.insert(0, '/home/tracy/data/scripts/HRNet-Human-Pose-Estimation/lib/')
sys.path.insert(0, '/home/tracy/data/scripts/HRNet-Human-Pose-Estimation/')
import numpy as np
from scipy.io import loadmat, savemat
from config import update_config
from dataset.JointsDataset import JointsDataset
import argparse
import csv
logger = logging.getLogger(__name__)


class LIPDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 16    
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)
        self.pixel_std = 200
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))
    
    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + (w-1) * 0.5
        center[1] = y + (h-1) * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        # if center[0] != -1:
        #     scale = scale * 1.25

        return center, scale

    def _get_db(self):
        # create train/val split
        file_name = os.path.join(
            self.root, 'annot', self.image_set+'.json'
        )

        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno['root']:
            image_name = a['im_name']
            w = a['img_width']
            h = a['img_height']
            # c = np.array(a['center'], dtype=np.float)
            # s = np.array([a['scale'], a['scale']], dtype=np.float)

            c, s = self._box2cs([0, 0, w, h])

            # # Adjust center/scale slightly to avoid cropping limbs
            # if c[0] != -1:
            #     c[1] = c[1] + 15 * s[1]
            #     s = s * 1.25

            # # MPII uses matlab format, index is based 1,
            # # we should first convert to 0-based index
            # c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)
            if self.image_set != 'test':
                joints = np.array(a['joint_self'])
                joints[:, 0:2] = joints[:, 0:2]
                #joints[:, 0:2] = joints[:, 0:2] - 1
                # get visibility of joints
                #joints_vis = np.array(a['joints_vis'])
                coord_sum = np.sum(joints, axis=1)
                joints_vis = coord_sum != 0
                
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]
            if self.image_set == 'train':
                image_dir = 'train_images'
            else:
                image_dir = 'val_images'
            if self.image_set == 'train':
                anno_dir = 'train_segmentations'
            else:
                anno_dir = 'val_segmentations'
            if self.image_set == 'test':
                image_dir = 'test_images'
  
            #image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            gt_db.append(
                {
                    'image': os.path.join(self.root, image_dir, image_name),
                    'anno': os.path.join(self.root, anno_dir, image_name[:-4]+'.png'),
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': image_name,
                    'imgnum': 0,
                }
            )

        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        SC_BIAS = 0.6
        threshold = 0.5

        # gt_file = '/home/tracy/Yanghong/scripts/data/images_labels/TrainVal_pose_annotations/gt_valid.mat'
        # gt_dict = loadmat(gt_file)
        # dataset_joints = gt_dict['dataset_joints']
        # jnt_missing = gt_dict['jnt_missing']
        # pos_gt_src = gt_dict['pos_gt_src']
        # headboxes_src = gt_dict['headboxes_src']

        path = cfg.DATASET.CSV_FILE
        additional_dim = True
        pos_gt_src,jnt_visible = read_data(path, additional_dim)
        headsizes = get_head_size(pos_gt_src)
        pck_th_range = [0.5]
        dist = get_norm_dist(preds, pos_gt_src, headsizes)
        #print(dist)
        pck = compute_pck(dist, pck_th_range)
        pck = pck[-1]
        
        # pos_gt_src = np.transpose(pos_gt_src, [1, 2, 0])
        # jnt_visible = np.transpose(jnt_visible, [1,0])

        

        # pos_pred_src = np.transpose(preds, [1, 2, 0])

        # head = np.where(dataset_joints == 'head')[1][0]
        # lsho = np.where(dataset_joints == 'lsho')[1][0]
        # lelb = np.where(dataset_joints == 'lelb')[1][0]
        # lwri = np.where(dataset_joints == 'lwri')[1][0]
        # lhip = np.where(dataset_joints == 'lhip')[1][0]
        # lkne = np.where(dataset_joints == 'lkne')[1][0]
        # lank = np.where(dataset_joints == 'lank')[1][0]

        # rsho = np.where(dataset_joints == 'rsho')[1][0]
        # relb = np.where(dataset_joints == 'relb')[1][0]
        # rwri = np.where(dataset_joints == 'rwri')[1][0]
        # rkne = np.where(dataset_joints == 'rkne')[1][0]
        # rank = np.where(dataset_joints == 'rank')[1][0]
        # rhip = np.where(dataset_joints == 'rhip')[1][0]

        # #jnt_visible = 1 - jnt_missing
        # uv_error = pos_pred_src - pos_gt_src
        # uv_err = np.linalg.norm(uv_error, axis=1)
        # # headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        # # headsizes = np.linalg.norm(headsizes, axis=0)
        # headsizes *= SC_BIAS
        # scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        # scaled_uv_err = np.divide(uv_err, scale)
        # scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        # jnt_count = np.sum(jnt_visible, axis=1)
        # less_than_threshold = np.multiply((scaled_uv_err <= threshold),
        #                                   jnt_visible)
        # PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        # # save
        # rng = np.arange(0, 0.5+0.01, 0.01)
        # pckAll = np.zeros((len(rng), 16))

        # for r in range(len(rng)):
        #     threshold = rng[r]
        #     less_than_threshold = np.multiply(scaled_uv_err <= threshold,
        #                                       jnt_visible)
        #     pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
        #                              jnt_count)

        # PCKh = np.ma.array(PCKh, mask=False)
        # PCKh.mask[6:8] = True

        # jnt_count = np.ma.array(jnt_count, mask=False)
        # jnt_count.mask[6:8] = True
        # jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', 0.5 * (pck[8] + pck[9])),
            ('Shoulder', 0.5 * (pck[12] + pck[13])),
            ('Elbow', 0.5 * (pck[11] + pck[14])),
            ('Wrist', 0.5 * (pck[10] + pck[15])),
            ('Hip', 0.5 * (pck[2]  + pck[3])),
            ('Knee', 0.5 * (pck[1]  + pck[4])),
            ('Ankle', 0.5 * (pck[0]  + pck[5])),
            ('Mean', pck[-2]),
            ('Mean@0.1', pck[-1])
        ]
        name_value = OrderedDict(name_value)

        return name_value , name_value['Mean@0.1']

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='/home/tracy/data/scripts/HRNet-Human-Pose-Estimation/experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args

def read_data(path, additional_dim):
    labels = []
    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        for row in reader:
            label = row[1:]
            for l in range(len(label)):
                if label[l] == 'nan':
                    label[l] = '-1'
                label[l] = float(label[l])
            labels.append(label)

    data = np.array(labels)
    dim =2
    if additional_dim:
        dim = 3
    data = np.reshape(data,[data.shape[0], int(data.shape[1] / dim), dim])

    vis_label = np.zeros((data.shape[0], data.shape[1]))

    if additional_dim:
        vis_label[:, :] = data[:, :, 2]
        data = data[:, :, 0:2]
    else:
        vis_label = vis_label + 1
        data[data<0] = 1

    return data, vis_label

def get_head_size(gt):
	head_size = np.linalg.norm(gt[:,9,:] - gt[:,8,:], axis=1)
	for n in range(gt.shape[0]):
		if gt[n,8,0] < 0 or gt[n,9,0] < 0:  
			head_size[n] = 0

	return head_size

def get_norm_dist(pred, gt, ref_dist):
	N = pred.shape[0]
	P = pred.shape[1]
	dist = np.zeros([N, P])
	for n in range(N):
		cur_ref_dist = ref_dist[n]
		if cur_ref_dist == 0:
			dist[n, :] = -1   
		else:
			dist[n, :] = np.linalg.norm(gt[n, :, :] - pred[n, :, :], axis=1) / cur_ref_dist
			for p in range(P):
				if gt[n, p, 0] < 0 or gt[n, p, 1] < 0 or (gt[n, p, 0] == 0 and gt[n, p, 1] == 0): dist[n, p] = -1
	return dist

def compute_pck(dist, pck_th_range):
	P = dist.shape[1]
	pck = np.zeros([len(pck_th_range), P + 2])

    # For individual joint
	for p in range(P):
		for thi in range(len(pck_th_range)):
			th = pck_th_range[thi]
			joint_dist = dist[:, p]
			pck[thi, p] = 100 * np.mean(joint_dist[np.where(joint_dist >= 0)] <= th)
        # For uppper body
	for thi in range(len(pck_th_range)):
		th = pck_th_range[thi]
		joint_dist = dist[:, 8:16]
		pck[thi, P] = 100 * np.mean(joint_dist[np.where(joint_dist >= 0)] <= th)

	# For all joints
	for thi in range(len(pck_th_range)):
		th = pck_th_range[thi]
		joints_index = list(range(0,6)) + list(range(8,16))
		joint_dist = dist[:, joints_index]
		pck[thi, P + 1] = 100 * np.mean(joint_dist[np.where(joint_dist >= 0)] <= th)

	

	return pck

def pck_table_output_lip_dataset(pck,epoch, save_file,method_name='ours'):
    str_template = '{0:10} & {1:6} & {2:6} & {3:6} & {4:6} & {5:6} & {6:6} & {7:6} & {8:6} & {9:6}'
    head_str = str_template.format('PCKh@0.5', 'Head', 'Sho.', 'Elb.', 'Wri.', 'Hip', 'Knee', 'Ank.', 'U.Body', 'Avg.')
    num_str = str_template.format(method_name, '%1.1f'%((pck[8]  + pck[9])  / 2.0),
                                               '%1.1f'%((pck[12] + pck[13]) / 2.0),
                                               '%1.1f'%((pck[11] + pck[14]) / 2.0),
                                               '%1.1f'%((pck[10] + pck[15]) / 2.0),
                                               '%1.1f'%((pck[2]  + pck[3])  / 2.0),
                                               '%1.1f'%((pck[1]  + pck[4])  / 2.0),
                                               '%1.1f'%((pck[0]  + pck[5])  / 2.0),
                                               '%1.1f'%(pck[-2]),
                                               '%1.1f'%(pck[-1]))
    
    print(head_str)
    print(num_str)
    if epoch==0:
        with open(save_file,'a+') as f:
                f.writelines('epoch'+' '+head_str+'\n')
    else:
        with open(save_file,'a+') as f:
                f.writelines('epoch'+' '+ str(epoch)+' '+num_str+'\n')

if __name__ == '__main__':
    # Data loading code
    import torchvision.transforms as transforms
    import torch
    import dataset
    from config import cfg

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    args = parse_args()
    update_config(cfg, args)

    # gt_file = '/home/tracy/data/scripts/data/images_labels/TrainVal_pose_annotations/gt_valid.mat'
    # gt_dict = loadmat(gt_file)
    # dataset_joints = gt_dict['dataset_joints']
    # jnt_missing = gt_dict['jnt_missing']
    # pos_gt_src = gt_dict['pos_gt_src']
    # headboxes_src = gt_dict['headboxes_src']
    # headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    # headsizes = np.linalg.norm(headsizes, axis=0)

    # file_name = os.path.join(
    #         self.root, 'annot', self.image_set+'.json'
    #     )
    path = cfg.DATASET.CSV_FILE
    additional_dim = True
    pose,vis_pose = read_data(path, additional_dim)
    head_size = get_head_size(pose)
    pose_test = np.transpose(pose, [1, 2, 0])
    vis_pose_test = np.transpose(vis_pose, [1,0])
    train_dataset = eval('dataset.'+'lip')(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        print(input)
