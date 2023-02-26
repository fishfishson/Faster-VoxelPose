# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import os.path as osp
import numpy as np
import json_tricks as json
import pickle
import logging
import cv2
import copy
from tqdm import tqdm

from dataset.JointsDataset import JointsDataset

from easymocap.mytools.camera_utils import read_cameras


logger = logging.getLogger(__name__)

DEBUG = False

body25topanoptic15 = [1,0,8,5,6,7,12,13,14,2,3,4,9,10,11]

panoptic_joints_def = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
}

panoptic_bones_def = [
    [0, 1], [0, 2],  # trunk
    [0, 3], [3, 4], [4, 5],  # left arm
    [0, 9], [9, 10], [10, 11],  # right arm
    [2, 6], [6, 7], [7, 8],  # left leg
    [2, 12], [12, 13], [13, 14],  # right leg
]

class DEMO(JointsDataset):
    def __init__(self, cfg, is_train=True, is_test=False, transform=None):
        super().__init__(cfg, is_train, transform)

        self.num_joints = len(panoptic_joints_def)
        self.num_views = cfg.DATASET.CAMERA_NUM
        self.root_id = cfg.DATASET.ROOTIDX

        self.has_evaluate_function = False
        self.transform = transform

        self.image_set = 'demo'
        self.sequence_list = ['demo']
        self._interval = 1
        if cfg.DATASET.CAM_LIST is not None:
            self.cam_list = cfg.DATASET.CAM_LIST.split(' ')
        else:
            self.cam_list = ['01','02','03','04']
    
        self.cameras = self._get_cam()
        self.db_file = 'faster_voxelpose_{}_cam{}.pkl'.format(self.image_set, self.num_views)
        self.db_file = osp.join(self.dataset_root, self.db_file)

        if osp.exists(self.db_file):
            info = pickle.load(open(self.db_file, 'rb'))
            assert info['sequence_list'] == self.sequence_list
            assert info['interval'] == self._interval
            self.db = info['db']
        else:
            self._get_db()
            info = {
                'sequence_list': self.sequence_list,
                'interval': self._interval,
                'db': self.db
            }
            pickle.dump(info, open(self.db_file, 'wb'))
        self.db_size = len(self.db)
    
    def _get_db(self):
        for seq in tqdm(self.sequence_list):
            curr_anno = osp.join(self.dataset_root, seq, 'images', self.cam_list[0])
            anno_files = sorted(glob.iglob('{:s}/*.png'.format(curr_anno)))
            anno_files += sorted(glob.iglob('{:s}/*.jpg'.format(curr_anno)))
            print(f'load sequence: {seq}', flush=True)

            cameras = self.cameras[seq]

            # save all image paths and 3d gt joints
            for i, anno_file in enumerate(anno_files):
                if i % self._interval == 0:
                    all_image_path = []
                    # for k in range(self.num_views):
                    for k, v in cameras.items():
                        # suffix = osp.basename(anno_file).replace("body3DScene", "")
                        # prefix = "{:02d}_{:02d}".format(self.cam_list[k][0], self.cam_list[k][1])
                        image_path = osp.join(self.dataset_root, seq, "images", k, osp.basename(anno_file))
                        all_image_path.append(image_path)
                    all_poses_3d = [np.random.rand(15, 3) * 1000.0]
                    all_poses_3d_vis = [np.ones(15)]

                    if len(all_poses_3d) > 0:
                        self.db.append({
                            'seq': seq,
                            'all_image_path': all_image_path,
                            'joints_3d': all_poses_3d,
                            'joints_3d_vis': all_poses_3d_vis,
                        })
            
        super()._rebuild_db()
        logger.info("=> {} images from {} views loaded".format(len(self.db), self.num_views))
        return
    
    def _get_cam(self):
        cameras = dict()

        for seq in self.sequence_list:
            cameras[seq] = dict()

            calib = read_cameras(osp.join(self.dataset_root, seq))

            for k, v in calib.items():
                if k not in self.cam_list: continue
                sel_cam = dict()
                sel_cam['K'] = np.array(v['K'])
                sel_cam['distCoef'] = np.array(v['dist']).flatten()
                sel_cam['R'] = np.array(v['R'])
                sel_cam['t'] = np.array(v['T']).reshape(3, 1)
                sel_cam['T'] = -np.dot(sel_cam['R'].T, sel_cam['t']) * 1000.0
                sel_cam['fx'] = sel_cam['K'][0, 0]
                sel_cam['fy'] = sel_cam['K'][1, 1]
                sel_cam['cx'] = sel_cam['K'][0, 2]
                sel_cam['cy'] = sel_cam['K'][1, 2]
                sel_cam['k'] = sel_cam['distCoef'][[0, 1, 4]].reshape(3, 1)
                sel_cam['p'] = sel_cam['distCoef'][[2, 3]].reshape(2, 1)
                cameras[seq][k] = sel_cam

        return cameras 
    
    def __getitem__(self, idx):
        input, target, meta, input_heatmap = super().__getitem__(idx)
        return input, target, meta, input_heatmap
    
    def __len__(self):
        return self.db_size

    def evaluate(self, preds):
        eval_list = []
        gt_num = self.db_size
        assert len(preds) == gt_num, 'number mismatch'

        total_gt = 0
        for i in range(len(preds)):
            db_rec = copy.deepcopy(self.db[i])
            num_person = db_rec['meta']['num_person']
            joints_3d = db_rec['meta']['joints_3d'][:num_person]
            joints_3d_vis = db_rec['meta']['joints_3d_vis'][:num_person]

            if num_person == 0:
                continue

            pred = preds[i].copy()
            pred = pred[pred[:, 0, 3] >= 0]
            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis > 0.1
                    mpjpe = np.mean(np.sqrt(np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4]
                eval_list.append({
                    "mpjpe": float(min_mpjpe),
                    "score": float(score),
                    "gt_id": int(total_gt + min_gt)
                })
            total_gt += len(joints_3d)

        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)

        mpjpe = self._eval_list_to_mpjpe(eval_list)
        recall = self._eval_list_to_recall(eval_list, total_gt)
        msg = 'Evaluation results on Panoptic dataset:\nap@25: {aps_25:.4f}\tap@50: {aps_50:.4f}\tap@75: {aps_75:.4f}\t' \
              'ap@100: {aps_100:.4f}\tap@125: {aps_125:.4f}\tap@150: {aps_150:.4f}\t' \
              'recall@500mm: {recall:.4f}\tmpjpe@500mm: {mpjpe:.3f}'.format(
                aps_25=aps[0], aps_50=aps[1], aps_75=aps[2], aps_100=aps[3],
                aps_125=aps[4], aps_150=aps[5], recall=recall, mpjpe=mpjpe
              )
        metric = np.mean(aps)

        return metric, msg

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.append(item["gt_id"])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                gt_det.append(item["gt_id"])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]

        return len(np.unique(gt_ids)) / total_gt