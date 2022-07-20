from typing import Any, Callable, Dict
import numpy as np
import torch
from torch import nn
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler

__all__ = ['GeneralTrainer']


class GeneralTrainer(Trainer):
    def __init__(self, model: nn.Module, criterion: Callable,
                 optimizer: Optimizer, scheduler: Scheduler,
                 fix_mean_var: bool,
                 num_workers: int, seed: int) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.fix_mean_var = fix_mean_var
        self.seed = seed
        self.epoch_num = 1

    def fix_bn_mean_var(self):
        for name, m in self.model.named_modules():
            if m.__class__.__name__.find('BatchNorm') != -1 and 'stage' in name:
                m.eval()

    def _before_epoch(self) -> None:
        self.model.train()
        if self.fix_mean_var:
            self.fix_bn_mean_var()
        self.dataflow.sampler.set_epoch(self.epoch_num-1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
                self.seed + (self.epoch_num-1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        _inputs = dict()
        for key, value in feed_dict.items():
            if not 'name' in key:
                _inputs[key] = value.cuda()
        inputs = _inputs['lidar'] if 'lidar' in _inputs.keys() else None
        if isinstance(inputs, torch.Tensor):
            B, C, H, W = inputs.shape
            targets = feed_dict['targets'].long().cuda(non_blocking=True)
            outputs = self.model(inputs)
            if outputs.requires_grad:
                loss = self.criterion(outputs, targets)
                self.summary.add_scalar('loss', loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            else:
                outputs = outputs.reshape(B, -1, H*W).argmax(1)
                targets = targets.reshape(B, -1)

            return {'outputs': outputs, 'targets': targets}

        else:
            if 'lidar' in _inputs.keys():
                inputs = _inputs['lidar']
            else:
                inputs = [_inputs['lidar0'], _inputs['lidar1'], _inputs['lidar2'], _inputs['lidar3']]
            targets = feed_dict['targets']
            if isinstance(targets, torch.Tensor):
                targets = targets.long().cuda(non_blocking=True).squeeze(-1)
            else:
                targets = targets.F.long().cuda(non_blocking=True)
            outputs = self.model(inputs)

            if outputs.requires_grad:
                loss = self.criterion(outputs, targets.squeeze(-1))
                self.summary.add_scalar('loss', loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            else:
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
                    outputs_mapped = outputs[cur_scene_pts][
                        cur_inv].argmax(1)
                    targets_mapped = all_labels.F[cur_label]
                    _outputs.append(outputs_mapped)
                    _targets.append(targets_mapped)
                outputs = torch.cat(_outputs, 0)
                targets = torch.cat(_targets, 0)

            return {'outputs': outputs, 'targets': targets}

    def _after_epoch(self) -> None:
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = dict()
        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass
