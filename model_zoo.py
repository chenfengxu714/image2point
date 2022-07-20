import torch
import torchvision.models as models

from core.models.collection.spvcnn18 import SPVCNN18
from core.models.collection.spvcnn18_cls import SPVCNN18_cls
from core.models.collection.spvcnn34_cls import SPVCNN34_cls
from core.models.collection.spvcnn50_cls import SPVCNN50_cls
from core.models.collection.spvcnn101_cls import SPVCNN101_cls
from core.models.collection.spvcnn152_cls import SPVCNN152_cls


__all__ = ['spvcnn18', 'spvcnn34', 'spvcnn_cls']


def spvcnn18(configs):
    if 'cr' in configs.model:
        cr = configs.model.cr
    else:
        cr = 1.0

    model = SPVCNN18(
        num_classes=configs.dataset.num_classes,
        input_dim=configs.data.input_dim,
        cr=cr,
        pres=configs.dataset.voxel_size,
        vres=configs.dataset.voxel_size,
    )

    if configs.train.pretrained_dir != 'none':
        resnet = torch.load(configs.train.pretrained_dir)
        if 'state_dict' in resnet.keys():
            resnet = resnet['state_dict']
    else:
        resnet = models.resnet18(pretrained=True).state_dict()

    key_list = []
    for key in resnet.keys():
        if 'layer' in key:
            key_list.append(key)

    spv_key = []
    for key in model.state_dict().keys():
        if 'stage' in key:
            spv_key.append(key)

    assert len(key_list) == len(spv_key)

    weights = {}

    for idx, res_key in enumerate(key_list):
        pretrain_weight = resnet[res_key]
        if len(pretrain_weight.shape) != 4:
            weights[spv_key[idx]] = pretrain_weight
        else:
            weights[spv_key[idx]] = weight_transform(pretrain_weight, res_key, \
                trans_type=configs.train.transformation \
                    if 'transformation' in configs.train else 'z-axis')
        if weights[spv_key[idx]].shape != model.state_dict()[spv_key[idx]].shape:
            weights[spv_key[idx]] = weights[spv_key[idx]].view(27, 2, 64, 64).mean(1)

    model.load_state_dict(weights, strict=False)
    return model


def weight_transform(model, key, trans_type='z-axis'):
    if 'bn' in key:
        return model

    if 'conv' in key and model.shape[-1]==3:
        cout, cin = model.shape[:2]
        if trans_type == 'z-axis':
            model = model.permute(2, 3, 1, 0).contiguous()
            weights = model.clone().view(-1, cin, cout).repeat(3, 1, 1)
            weights[:9, :, :] = model[0, :, :, :].repeat(3, 1, 1)
            weights[9:18, :, :] = model[1, :, :, :].repeat(3, 1, 1)
            weights[18:, :, :] = model[2, :, :, :].repeat(3, 1, 1)

        elif trans_type == 'y-axis':
            trans = torch.eye(9).repeat(1, 3).contiguous()
            model = model.view(cout, cin, -1)
            weights = torch.matmul(model, trans).permute(2, 1, 0)

        elif trans_type == 'x-axis':
            trans = torch.zeros(9, 27)
            trans[0, :3] = 1
            trans[3, 3:6] = 1
            trans[6, 6:9] = 1
            trans[1, 9:12] = 1
            trans[4, 12:15] = 1
            trans[7, 15:18] = 1
            trans[2, 18:21] = 1
            trans[5, 21:24] = 1
            trans[8, 24:27] = 1
            trans = trans.contiguous()
            model = model.view(cout, cin, -1)
            weights = torch.matmul(model, trans).permute(2, 1, 0)

    elif model.shape[-1]==1 and 'downsample' in key:
        weights = model.clone().squeeze().permute(1, 0).contiguous().unsqueeze(0).repeat(8, 1, 1)

    elif model.shape[-1]==1:
        weights = model.clone().squeeze().permute(1, 0).contiguous()
    return weights


def spvcnn34(configs):
    if 'cr' in configs.model:
        cr = configs.model.cr
    else:
        cr = 1.0

    model = SPVCNN34_cls(
        num_classes=configs.dataset.num_classes,
        input_dim=configs.data.input_dim,
        cr=cr,
        pres=configs.dataset.voxel_size,
        vres=configs.dataset.voxel_size)

    if configs.train.pretrained_dir != 'none':
        resnet = torch.load(configs.train.pretrained_dir, map_location='cuda:0')
        if 'state_dict' in resnet.keys():
            resnet = resnet['state_dict']

    if "nce" in configs.train.pretrained_dir or "hard" in configs.train.pretrained_dir:
        key_list = []
        for key in resnet.keys():
            if 'block1' in key:
                key_list.append(key)
            if 'block2' in key:
                key_list.append(key)
            if 'block3' in key:
                key_list.append(key)
            if 'block4' in key:
                key_list.append(key)
            if 'conv2p2s2' in key:
                key_list.append(key)
            if 'bn2' in key:
                key_list.append(key)
            if 'conv3p4s2' in key:
                key_list.append(key)
            if 'bn3' in key:
                key_list.append(key)
            if 'conv4p8s2' in key:
                key_list.append(key)
            if 'bn4' in key:
                key_list.append(key)

        spv_key = []
        for key in model.state_dict().keys():
            if 'stage' in key or 'ds' in key:
                spv_key.append(key)

    else:
        key_list = []
        for key in resnet.keys():
            if 'layer' in key:
                key_list.append(key)
            if 'ds' in key:
                key_list.append(key)

        spv_key = []
        for key in model.state_dict().keys():
            if 'stage' in key:
                spv_key.append(key)
            if 'ds' in key:
                spv_key.append(key)

    assert len(key_list) == len(spv_key)

    weights = {}

    for idx, res_key in enumerate(key_list):
        pretrain_weight = resnet[res_key]
        if len(pretrain_weight.shape) != 4:
            weights[spv_key[idx]] = pretrain_weight
        else:
            weights[spv_key[idx]] = weight_transform_34(pretrain_weight, res_key, \
                trans_type=configs.train.transformation \
                    if 'transformation' in configs.train else 'z-axis')

    model.load_state_dict(weights, strict=False)
    return model


def weight_transform_34(model, key, trans_type='z-axis'):
    if 'bn' in key:
        return model
    if 'conv' in key and model.shape[-1]==3:
        cout, cin = model.shape[:2]

        if trans_type == 'z-axis':
            model = model.permute(2,3,1,0).contiguous()
            weights = model.clone().view(-1, cin, cout).repeat(3, 1, 1)
            weights[:9, :, :] = model[0, :, :, :].repeat(3, 1, 1)
            weights[9:18, :, :] = model[1, :, :, :].repeat(3, 1, 1)
            weights[18:, :, :] = model[2, :, :, :].repeat(3, 1, 1)

    elif 'net' in key and 'ds' in key and model.shape[-1]==2:
        cout, cin = model.shape[:2]
        model = model.permute(2,3,1,0).contiguous()
        weights = model.clone().view(-1, cin, cout).repeat(2, 1, 1)
        weights[:4 :, :] = model[0, :, :, :].repeat(2, 1, 1)
        weights[4:, :, :] = model[1, :, :, :].repeat(2, 1, 1)

    elif model.shape[-1]==1 and 'downsample' in key:
        weights = model.clone().squeeze().permute(1, 0).contiguous()

    elif model.shape[-1]==1:
        weights = model.clone().squeeze().permute(1, 0).contiguous()
    return weights


def spvcnn_cls(configs):
    if 'cr' in configs.model:
        cr = configs.model.cr
    else:
        cr = 1.0

    if configs.model.name == "spvcnn18_cls":
        model = SPVCNN18_cls(
            num_classes=configs.dataset.num_classes,
            input_dim=configs.data.input_dim,
            cr=cr,
            pres=configs.dataset.voxel_size,
            vres=configs.dataset.voxel_size)

    elif configs.model.name == "spvcnn50_cls":
        model = SPVCNN50_cls(
            num_classes=configs.dataset.num_classes,
            input_dim=configs.data.input_dim,
            cr=cr,
            pres=configs.dataset.voxel_size,
            vres=configs.dataset.voxel_size)

    elif configs.model.name == "spvcnn101_cls":
        model = SPVCNN101_cls(
            num_classes=configs.dataset.num_classes,
            input_dim=configs.data.input_dim,
            cr=cr,
            pres=configs.dataset.voxel_size,
            vres=configs.dataset.voxel_size)

    elif configs.model.name == "spvcnn152_cls":
        model = SPVCNN152_cls(
            num_classes=configs.dataset.num_classes,
            input_dim=configs.data.input_dim,
            cr=cr,
            pres=configs.dataset.voxel_size,
            vres=configs.dataset.voxel_size)

    else:
        raise NotImplementedError

    if configs.train.pretrained_dir != 'none':
        resnet = torch.load(configs.train.pretrained_dir, map_location='cuda:0')
        if 'state_dict' in resnet.keys():
            resnet = resnet['state_dict']
    elif configs.model.name == "spvcnn18_cls":
        resnet = models.resnet18(pretrained=True).state_dict()
    elif configs.model.name == "spvcnn50_cls":
        resnet = models.resnet50(pretrained=True).state_dict()
    elif configs.model.name == "spvcnn101_cls":
        resnet = models.resnet101(pretrained=True).state_dict()
    elif configs.model.name == "spvcnn152_cls":
        resnet = models.resnet152(pretrained=True).state_dict()

    key_list = []
    for key in resnet.keys():
        if 'layer' in key:
            if 'load_exclude' in configs.train and \
                all(['layer{}'.format(e) not in key for e in configs.train.load_exclude]):
                key_list.append(key)

    spv_key = []
    for key in model.state_dict().keys():
        if 'stage' in key:
            if 'load_exclude' in configs.train and \
                all(['stage{}'.format(e) not in key for e in configs.train.load_exclude]):
                spv_key.append(key)

    assert len(key_list) == len(spv_key)

    weights = {}

    for idx, res_key in enumerate(key_list):
        pretrain_weight = resnet[res_key]
        if len(pretrain_weight.shape) != 4:
            weights[spv_key[idx]] = pretrain_weight
        else:
            weights[spv_key[idx]] =  weight_transform(pretrain_weight, res_key, \
                trans_type=configs.train.transformation \
                    if 'transformation' in configs.train else 'z-axis')

    model.load_state_dict(weights, strict=False)
    return model