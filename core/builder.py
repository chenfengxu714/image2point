from typing import Callable

import torch
import torch.optim
from torch import nn
from torchpack.utils.config import configs
from torchpack.utils.typing import Dataset, Optimizer, Scheduler

__all__ = [
    'make_dataset', 'make_model', 'make_criterion', 'make_optimizer',
    'make_scheduler', 'freeze_parameter'
]


def make_dataset() -> Dataset:
    if configs.dataset.name == 'modelnet':
        from core.datasets import ModelNet
        dataset = ModelNet(root=configs.dataset.root,
                        voxel_size=configs.dataset.voxel_size,
                        config_data=configs.data)

    elif configs.dataset.name == 'semantic_kitti':
        from core.datasets import SemanticKITTI
        dataset = SemanticKITTI(root=configs.dataset.root,
                        voxel_size=configs.dataset.voxel_size,
                        config_data=configs.data)

    elif configs.dataset.name == 's3dis':
        from core.datasets import S3DIS
        dataset = S3DIS(root=configs.dataset.root,
                        voxel_size=configs.dataset.voxel_size,
                        config_data=configs.data)

    else:
        raise NotImplementedError(configs.dataset.name)
    return dataset


def make_model() -> nn.Module:
    if 'cr' in configs.model:
        cr = configs.model.cr
    else:
        cr = 1.0

    if configs.model.name == 'spvcnn18':
        from core.models.collection import SPVCNN18
        model = SPVCNN18(
                num_classes=configs.dataset.num_classes,
                input_dim=configs.data.input_dim,
                cr=cr,
                pres=configs.dataset.voxel_size,
                vres=configs.dataset.voxel_size)

    elif configs.model.name == 'spvcnn34':
        from core.models.collection import SPVCNN34
        model = SPVCNN34(
                num_classes=configs.dataset.num_classes,
                input_dim=configs.data.input_dim,
                cr=cr,
                pres=configs.dataset.voxel_size,
                vres=configs.dataset.voxel_size)

    elif configs.model.name == 'spvcnn18_cls':
        from core.models.collection import SPVCNN18_cls
        model = SPVCNN18_cls(
                num_classes=configs.dataset.num_classes,
                input_dim=configs.data.input_dim,
                cr=cr,
                pres=configs.dataset.voxel_size,
                vres=configs.dataset.voxel_size)

    elif configs.model.name == 'spvcnn34_cls':
        from core.models.collection import SPVCNN34_cls
        model = SPVCNN34_cls(
                num_classes=configs.dataset.num_classes,
                input_dim=configs.data.input_dim,
                cr=cr,
                pres=configs.dataset.voxel_size,
                vres=configs.dataset.voxel_size)

    elif configs.model.name == 'spvcnn50_cls':
        from core.models.collection import SPVCNN50_cls
        model = SPVCNN50_cls(
                num_classes=configs.dataset.num_classes,
                input_dim=configs.data.input_dim,
                cr=cr,
                pres=configs.dataset.voxel_size,
                vres=configs.dataset.voxel_size)

    elif configs.model.name == 'spvcnn101_cls':
        from core.models.collection import SPVCNN101_cls
        model = SPVCNN101_cls(
                num_classes=configs.dataset.num_classes,
                input_dim=configs.data.input_dim,
                cr=cr,
                pres=configs.dataset.voxel_size,
                vres=configs.dataset.voxel_size)
    
    elif configs.model.name == 'spvcnn152_cls':
        from core.models.collection import SPVCNN152_cls
        model = SPVCNN152_cls(
                num_classes=configs.dataset.num_classes,
                input_dim=configs.data.input_dim,
                cr=cr,
                pres=configs.dataset.voxel_size,
                vres=configs.dataset.voxel_size)

    else:
        raise NotImplementedError(configs.model.name)
    return model


def make_criterion() -> Callable:
    if configs.criterion.name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=configs.criterion.ignore_index)

    elif configs.criterion.name == 'softmax_mse':
        from core.criterions import softmax_mse_loss
        criterion = softmax_mse_loss

    else:
        raise NotImplementedError(configs.criterion.name)
    return criterion


def make_optimizer(model: nn.Module) -> Optimizer:
    if configs.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=configs.optimizer.lr,
            momentum=configs.optimizer.momentum,
            weight_decay=configs.optimizer.weight_decay,
            nesterov=configs.optimizer.nesterov)

    elif configs.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)

    elif configs.optimizer.name == 'adamw':
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)

    else:
        raise NotImplementedError(configs.optimizer.name)
    return optimizer


def make_scheduler(optimizer: Optimizer) -> Scheduler:
    if configs.scheduler.name == 'none':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1)

    elif configs.scheduler.name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configs.num_epochs)

    elif configs.scheduler.name == 'cosine_warmup':
        from core.schedulers import cosine_schedule_with_warmup
        from functools import partial
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=partial(
                cosine_schedule_with_warmup,
                num_epochs=configs.num_epochs,
                batch_size=configs.batch_size,
                dataset_size=configs.dataset.training_size))

    elif configs.scheduler.name == 'polyLR':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda s: \
                (1 - s / (configs.scheduler.max_iter + 1))**configs.scheduler.power)

    else:
        raise NotImplementedError(configs.scheduler.name)
    return scheduler


def freeze_parameter(model: nn.Module):
    if ('trainable_params' in configs.train) and (configs.train.trainable_params != 'all'):
        if configs.train.trainable_params == 'io' or \
            configs.train.trainable_params == 'ioms':
            for name, p in model.named_parameters():
                if 'stage' in name:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

        elif configs.train.trainable_params == 'iobn':
            for name, p in model.named_parameters():
                if 'stage' in name and 'kernel' in name:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

        else:
            raise NotImplementedError(configs.train.trainable_params)
