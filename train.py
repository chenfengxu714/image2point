import argparse
import sys
import random
import numpy as np

import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchpack import distributed as dist
from torchpack.callbacks import (InferenceRunner, MaxSaver,TopKCategoricalAccuracy,
                                 Saver)
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core.builder import make_dataset, make_model, make_criterion, \
    make_optimizer, make_scheduler, freeze_parameter

from core.trainers import GeneralTrainer
from core.distill import DistillTrainer
from core.callbacks import MeanIoU

from model_zoo import spvcnn18, spvcnn34, spvcnn_cls


def main() -> None:
    dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--seed', default=None, type=int, help='change seed')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)

    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')

    # Set tensorboard
    writer = SummaryWriter(log_dir="tensorboard/"+args.run_dir)

    # Seed
    if ('seed' not in configs.train) or (configs.train.seed is None):
        configs.train.seed = torch.initial_seed() % (2**32 - 1)
    seed = configs.train.seed + dist.rank() * configs.workers_per_gpu * configs.num_epochs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset = make_dataset()
    dataflow = {}

    for split in dataset:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.batch_size,
            sampler=sampler,
            num_workers=configs.workers_per_gpu,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn)

    # Determine which kind of model should we build
    if not configs.train.pretrained:
        model = make_model()
    else:
        if configs.model.name == 'spvcnn18':
            model = spvcnn18(configs=configs)
        elif configs.model.name == 'spvcnn34_cls':
            model = spvcnn34(configs=configs)
        elif configs.model.name in \
            ['spvcnn18_cls', 'spvcnn50_cls', 'spvcnn101_cls', 'spvcnn152_cls']:
            model = spvcnn_cls(configs=configs)
        else:
            raise NotImplementedError

    # Determine whether to load the checkpoint
    if ('resume_path' in configs.data) and (configs.data.resume_path != 'none'):
        old_state_dict = torch.load(configs.data.resume_path + \
            '/checkpoints/max-acc-top1.pt')['model']
        new_state_dict = {}
        for key, value in old_state_dict.items():
            new_state_dict[key[7:]] = value
        model.load_state_dict(new_state_dict)
    elif ('distill_path' in configs.data) and (configs.data.resume_path != 'none'):
        teacher_model = make_model()
        old_state_dict = torch.load(configs.data.distill_path + \
            '/checkpoints/max-acc-top1.pt')['model']
        new_state_dict = {}
        for key, value in old_state_dict.items():
            new_state_dict[key[7:]] = value
        teacher_model.load_state_dict(new_state_dict)
        model.load_state_dict(new_state_dict)
        for param in teacher_model.parameters():
            param.detach_()
        teacher_model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model.cuda(),
        device_ids=[dist.local_rank()],
        find_unused_parameters=True)

    criterion = make_criterion()
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer)
    freeze_parameter(model)

    if 'distill_path' in configs.data:
        trainer = DistillTrainer(model=model,
                            teacher_model=teacher_model,
                            criterion=criterion,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            num_workers=configs.workers_per_gpu,
                            seed=seed)

    else:
        trainer = GeneralTrainer(model=model,
                            criterion=criterion,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            fix_mean_var=(configs.train.trainable_params == 'io'),
                            num_workers=configs.workers_per_gpu,
                            seed=seed)

    trainer.train_with_defaults(
        dataflow['train'],
        num_epochs=configs.num_epochs,
        callbacks=[
            InferenceRunner(
                dataflow[split],
                callbacks=[MeanIoU(
                    name=f'iou/{split}',
                    num_classes=configs.dataset.num_classes,
                    ignore_label=configs.data.ignore_label,
                    writer=writer
                    ) if configs.dataset.name != 'modelnet' else
                    TopKCategoricalAccuracy(k=1, name='acc/top1')])
            for split in ['test']
        ] + [
            MaxSaver('iou/test' if configs.dataset.name != 'modelnet' else 'acc/top1'),
            Saver(max_to_keep=10),
        ])


if __name__ == '__main__':
    main()
