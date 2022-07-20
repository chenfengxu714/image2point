import glob
import os
import os.path
import random

import numpy as np

from torchsparse import SparseTensor
from torchsparse.utils import sparse_collate_fn, sparse_quantize

__all__ = ['S3DIS']


CLASSES = [
      'clutter', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column', 'door', 'floor', 'sofa',
      'stairs', 'table', 'wall', 'window'
  ]


class S3DIS(dict):
    def __init__(self, root, voxel_size, config_data, **kwargs):

        sample_stride = kwargs.get('sample_stride', 1)

        super(S3DIS, self).__init__({
            'train':
            S3DISInternal(root,
                            voxel_size,
                            sample_stride=1,
                            less_frame = config_data.less_frame,
                            split='train',
                            clip_bound=config_data.clip_bound,
                            color_std=config_data.color_std,
                            color_trans_ratio=config_data.color_trans_ratio
                            ),
            'test':
            S3DISInternal(root,
                            voxel_size,
                            clip_bound=config_data.clip_bound,
                            sample_stride=sample_stride,
                            split='val')
        })


class S3DISInternal:
    def __init__(self,
                 root,
                 voxel_size,
                 split,
                 clip_bound,
                 sample_stride=1,
                 less_frame=1,
                 color_std=0.005,
                 color_trans_ratio=0.05):

        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.sample_stride = sample_stride
        self.std = color_std
        self.trans_range_ratio = color_trans_ratio
        self.clip_bound = clip_bound

        if self.split == 'train':
            self.seqs = [
                '1', '2', '3', '4', '6',
            ]
        else:
            self.seqs = ['5']

        self.files = []
        for area in self.seqs:
            self.files.extend(glob.glob(os.path.join(root, 'Area_{}/*/*.txt'.format(area))))

        if self.sample_stride > 1:
            self.files = self.files[::self.sample_stride]

        if split == 'train' and less_frame < 1:
            print("original {} frames".format(len(self.files)))
            print("sample {} frames".format(int(len(self.files)*less_frame)))
            self.files = random.sample(self.files, int(len(self.files)*less_frame))


    def __len__(self):
        return len(self.files)


    def read_txt(self, txtfile):
        # Read txt file and parse its content.
        with open(txtfile) as f:
            pointcloud = [l.split() for l in f]
        # Load point cloud to named numpy array.
        pointcloud = np.array(pointcloud).astype(np.float32)
        assert pointcloud.shape[1] == 6
        xyz = pointcloud[:, :3].astype(np.float32)
        rgb = pointcloud[:, 3:].astype(np.uint8)
        return xyz, rgb

    def feature_aug(self, feat):
        # color jitter
        if random.random() < 0.95:
            noise = np.random.randn(feat.shape[0], 3)
            noise *= self.std * 255
            feat[:, :3] = np.clip(noise + feat[:, :3], 0, 255)

        # color translation
        if random.random() < 0.95:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.trans_range_ratio
            feat[:, :3] = np.clip(tr + feat[:, :3], 0, 255)

        # color auto contrast
        if random.random() < 0.2:
            lo = feat[:, :3].min(0, keepdims=True)
            hi = feat[:, :3].max(0, keepdims=True)
            assert hi.max() > 1, f"invalid color value. Color is supposed to be [0-255]"

            scale = 255 / (hi - lo)

            contrast_feat = (feat[:, :3] - lo) * scale

            blend_factor = 0.5
            feat[:, :3] = (1 - blend_factor) * feat + blend_factor * contrast_feat
        return feat

    def coord_aug(self, coord):
        # random horizonal filp
        if random.random() < 0.95:
            for curr_ax in set(range(3)) - set([2]):
                if random.random() < 0.5:
                    coord_max = np.max(coord[:, curr_ax])
                    coord[:, curr_ax] = coord_max - coord[:, curr_ax]
        return coord


    def clip(self, coords, center=None, trans_aug_ratio=None):
        bound_min = np.min(coords, 0).astype(float)
        bound_max = np.max(coords, 0).astype(float)
        bound_size = bound_max - bound_min
        if center is None:
            center = bound_min + bound_size * 0.5
        if trans_aug_ratio is not None:
            trans = np.multiply(trans_aug_ratio, bound_size)
            center += trans
        lim = self.clip_bound


        if isinstance(self.clip_bound, (int, float)):
            if bound_size.max() < self.clip_bound:
                return None
            else:
                clip_inds = ((coords[:, 0] >= (-lim + center[0])) & \
                (coords[:, 0] < (lim + center[0])) & \
                (coords[:, 1] >= (-lim + center[1])) & \
                (coords[:, 1] < (lim + center[1])) & \
                (coords[:, 2] >= (-lim + center[2])) & \
                (coords[:, 2] < (lim + center[2])))
            return clip_inds

        # Clip points outside the limit
        clip_inds = ((coords[:, 0] >= (lim[0][0] + center[0])) & \
            (coords[:, 0] < (lim[0][1] + center[0])) & \
            (coords[:, 1] >= (lim[1][0] + center[1])) & \
            (coords[:, 1] < (lim[1][1] + center[1])) & \
            (coords[:, 2] >= (lim[2][0] + center[2])) & \
            (coords[:, 2] < (lim[2][1] + center[2])))
        return clip_inds


    def __getitem__(self, index):

        annotation, _ = os.path.split(self.files[index])
        subclouds = glob.glob(os.path.join(annotation, 'Annotations/*.txt'))

        pc_, feat_, labels_ = [], [], []

        for inst, subcloud in enumerate(subclouds):
            xyz, rgb = self.read_txt(subcloud)
            _, annotation_subfile = os.path.split(subcloud)
            clsidx = CLASSES.index(annotation_subfile.split('_')[0])

            pc_.append(xyz)
            feat_.append(rgb)
            labels_.append(np.ones((len(xyz), 1), dtype=np.int32) * clsidx)


        if len(pc_) == 0:
            print(self.files[index], ' has 0 files.')
        else:
            pc_ = np.concatenate(pc_, 0)
            feat_ = np.concatenate(feat_, 0)
            labels_ = np.concatenate(labels_, 0).squeeze()

            if self.split == 'train':
                pc_ = self.coord_aug(pc_)
                feat_ = self.feature_aug(feat_)


            clip_inds = self.clip(pc_)
            if clip_inds is not None:
                pc_, feat_, labels_ = pc_[clip_inds], feat_[clip_inds], labels_[clip_inds]

            pc_ = np.round(pc_ / self.voxel_size)
            feat_ = feat_ / 255. - 0.5

            pc_ -= pc_.mean(0)
            feat_ = np.concatenate((pc_, feat_), 1)

            inds, labels, inverse_map = sparse_quantize(pc_,
                                                    feat_,
                                                    labels_,
                                                    return_index=True,
                                                    return_invs=True)

            pc = pc_[inds]
            feat = feat_[inds]
            labels = labels_.T[inds]
            lidar = SparseTensor(feat, pc)
            labels = SparseTensor(labels, pc)
            labels_ = SparseTensor(labels_, pc_)
            inverse_map = SparseTensor(inverse_map, pc_)

            return {
                'lidar': lidar,
                'targets': labels,
                'targets_mapped': labels_,
                'inverse_map': inverse_map,
                'file_name': self.files[index]
            }


    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)
