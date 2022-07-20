from typing import Any, Dict
import numpy as np
import torch
from torch import nn
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler

__all__ = ['DistillTrainer']


class DistillTrainer(Trainer):
    def __init__(self, model: nn.Module, teacher_model: nn.Module,
                 optimizer: Optimizer, scheduler: Scheduler,
                 num_workers: int, seed: int, criterion=None) -> None:
        self.model = model
        self.teacher_model = teacher_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.epoch_num = 1
        self.batch_size = 32

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        _inputs = dict()
        for key, value in feed_dict.items():
            if not 'name' in key:
                _inputs[key] = value.cuda()
        inputs = _inputs['lidar']
        targets = None
        if isinstance(inputs, torch.Tensor):
            B, C, H, W = inputs.shape
            targets = feed_dict['targets'].long().cuda(non_blocking=True)
            psuedo_label = self.teacher_model(inputs)
            outputs = self.model(inputs)
            if outputs.requires_grad:
                loss = self.criterion(outputs, psuedo_label.detach().data.long().argmax(1))
                self.summary.add_scalar('loss', loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            else:
                targets = feed_dict['targets'].long().cuda(non_blocking=True)
                outputs = outputs.reshape(B, -1, H*W).argmax(1)
                targets = targets.reshape(B, -1)

            return {'outputs': outputs, 'targets': targets}

        else:
            inputs = _inputs['lidar'] #[32, Class 3]
            targets = feed_dict['targets']
            if type(targets) == torch.Tensor:
                targets = targets.long().cuda(non_blocking=True).squeeze(-1)
            else:
                targets = targets.F.long().cuda(non_blocking=True)
            outputs = self.model(inputs) # [32, Channel 40]
            psuedo_label = self.teacher_model(inputs) # [32, 40]

            if outputs.requires_grad:
                loss = self.criterion(outputs, psuedo_label.detach().data.long().argmax(1))
                self.summary.add_scalar('loss', loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            else:
                targets = feed_dict['targets']
                if isinstance(targets, torch.Tensor):
                    targets = targets.long().cuda(non_blocking=True).squeeze(-1)
                else:
                    targets = targets.F.long().cuda(non_blocking=True)
                if isinstance(feed_dict['targets'], torch.Tensor):
                    return {'outputs': outputs, 'targets': targets}
                invs = feed_dict['inverse_map']
                all_labels = feed_dict['targets_mapped']
                _outputs = []
                _targets = []
                for idx in range(invs.C[:, -1].max()+1):
                    cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
                    cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                    cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                    outputs_mapped = outputs[cur_scene_pts][cur_inv].argmax(1)
                    targets_mapped = all_labels.F[cur_label]
                    _outputs.append(outputs_mapped)
                    _targets.append(targets_mapped)
                outputs = torch.cat(_outputs, 0)
                targets = torch.cat(_targets, 0)

            return {'outputs': outputs, 'targets': targets}

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num-1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
                self.seed + (self.epoch_num-1) * self.num_workers + worker_id)

    def _after_epoch(self) -> None:
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = dict()
        state_dict['model'] = self.model.state_dict()
        state_dict['teacher_model'] = self.teacher_model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.model.load_state_dict(state_dict['teacher_model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass
