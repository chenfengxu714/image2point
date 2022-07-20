import os
import os.path

try:
    import h5py
except:
    print("Install h5py with `pip install h5py`")

import glob
import random
import numpy as np
from torchsparse import SparseTensor
from torchsparse.utils import sparse_collate_fn, sparse_quantize


class CoordinateTransformation:
    def __init__(self, scale_range=(0.9, 1.1), trans=0.25, jitter=0.025, clip=0.05):
        self.scale_range = scale_range
        self.trans = trans
        self.jitter = jitter
        self.clip = clip

    def __call__(self, coords):
        if random.random() < 0.9:
            coords *= np.random.uniform(
                low=self.scale_range[0], high=self.scale_range[1], size=[1, 3]
            )
        if random.random() < 0.9:
            coords += np.random.uniform(low=-self.trans, high=self.trans, size=[1, 3])
        if random.random() < 0.7:
            coords += np.clip(
                self.jitter * (np.random.rand(len(coords), 3) - 0.5),
                -self.clip,
                self.clip,
            )
        return coords

    def __repr__(self):
        return f"Transformation(scale={self.scale_range}, translation={self.trans}, jitter={self.jitter})"


class ModelNet(dict):
    def __init__(self, root, voxel_size, config_data, **kwargs):

        super(ModelNet, self).__init__({
            'train':
            ModelNetInternal(root,
                            voxel_size,
                            num_points = config_data.num_points,
                            few_shot= config_data.few_shot if ('few_shot' in config_data) and (config_data.few_shot != 'none') else None,
                            phase='train',
                            transform=CoordinateTransformation(trans=0.2)
                            ),
            'test':
            ModelNetInternal(root,
                            voxel_size,
                            num_points = config_data.num_points,
                            phase='val',
                            )
        })


def roate_and_scale(points, scale=None, angle=None):
    if angle:
        theta = angle
    else:
        theta = np.random.uniform(0, 2 * np.pi)
    if scale:
        scale_factor = scale
    else:
        scale_factor = np.random.uniform(0.95, 1.05)

    rot_mat = np.array([[np.cos(theta),
                                np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])

    points[:, :3] = np.dot(points[:, :3], rot_mat) * scale_factor
    return points


class ModelNetInternal:
    def __init__(
        self,
        data_root,
        voxel_size,
        num_points,
        phase,
        transform=None,
        few_shot=None
    ):
        phase = "test" if phase in ["val", "test"] else "train"
        self.few_shot = few_shot
        self.data, self.label = self.load_data(data_root, phase)
        self.transform = transform
        self.phase = phase
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.angle = 0.0

    def set_angle(self, angle):
        self.angle = angle

    def load_data(self, data_root, phase):
        data, labels = [], []
        assert os.path.exists(data_root), f"{data_root} does not exist"
        files = glob.glob(os.path.join(data_root, "ply_data_%s*.h5" % phase))
        assert len(files) > 0, "No files found"
        for h5_name in files:
            with h5py.File(h5_name) as f:
                data.extend(f["data"][:].astype("float32"))
                labels.extend(f["label"][:].astype("int64"))
        data = np.stack(data, axis=0)
        labels = np.stack(labels, axis=0)

        # Few shot setting and semi-supervised setting
        selected_data = []
        selected_labels = []
        if self.few_shot and phase == "train":
            for i in range(40):
                selected_index = labels.squeeze()==i
                sample_ind = np.random.choice(np.arange(np.sum(selected_index)), size=self.few_shot, replace=False)
                selected_data.append(data[selected_index][sample_ind])
                selected_labels.append(labels[selected_index][sample_ind])
            data = np.vstack(selected_data)
            labels = np.vstack(selected_labels)

        return data, labels

    def __getitem__(self, i: int) -> dict:
        xyz = self.data[i]
        label = self.label[i]
        if self.phase == "train":
            np.random.shuffle(xyz)
        if len(xyz) > self.num_points:
            xyz = xyz[: self.num_points]
        if self.transform is not None:
            xyz = self.transform(xyz)

        pc_ = np.round(xyz[:, :3] / self.voxel_size)
        pc_ -= pc_.min(0, keepdims=1)
        feat_ = xyz
        labels_ = label
        inds, inverse_map = sparse_quantize(pc_,
                                                    feat_,
                                                    labels=None,
                                                    return_index=True,
                                                    return_invs=True)

        if 'train' in self.phase:
            if len(inds) > self.num_points:
                inds = np.random.choice(inds, self.num_points, replace=False)

        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels_
        lidar = SparseTensor(feat, pc)
        inverse_map = SparseTensor(inverse_map, pc_)

        return {
            'lidar': lidar,
            'targets': labels,
            'inverse_map': inverse_map
        }

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        return f"ModelNet40H5(phase={self.phase}, length={len(self)}, transform={self.transform})"

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)
