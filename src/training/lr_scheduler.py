# copied from https://github.com/kuleshov-group/caduceus/blob/main/src/utils/optim/schedulers.py
import math
import warnings
import torch

from timm.scheduler import CosineLRScheduler


# https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
class CosineWarmup(torch.optim.lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, T_max, eta_min=0, warmup_step=0, **kwargs):
        self.warmup_step = warmup_step
        super().__init__(optimizer, T_max - warmup_step, eta_min, *kwargs)

    # Copied from CosineAnnealingLR, but adding warmup and changing self.last_epoch to
    # self.last_epoch - self.warmup_step.
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == self.warmup_step:  # also covers the case where both are 0
            return self.base_lrs
        elif self.last_epoch < self.warmup_step:
            return [base_lr * (self.last_epoch + 1) / self.warmup_step for base_lr in self.base_lrs]
        elif (self.last_epoch - self.warmup_step - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.warmup_step) / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - self.warmup_step - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    _get_closed_form_lr = None


def InvSqrt(optimizer, warmup_step):
    """ Originally used for Transformer (in Attention is all you need)"""

    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        if step == warmup_step:  # also covers the case where both are 0
            return 1.
        else:
            return 1. / (step ** 0.5) if step > warmup_step else (step + 1) / (warmup_step ** 1.5)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def constant(optimizer, warmup_step):
    def lr_lambda(step):
        if step == warmup_step:  # also covers the case where both are 0
            return 1.
        else:
            return 1. if step > warmup_step else (step + 1) / warmup_step

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class TimmCosineLRScheduler(CosineLRScheduler, torch.optim.lr_scheduler._LRScheduler):
    """ Wrap timm.scheduler.CosineLRScheduler so we can call scheduler.step() without passing in epoch.
    It supports resuming as well.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_epoch = -1
        self.step(epoch=0)

    def step(self, epoch=None):
        if epoch is None:
            self._last_epoch += 1
        else:
            self._last_epoch = epoch
        # We call either step or step_update, depending on whether we're using the scheduler every
        # epoch or every step.
        # Otherwise, lightning will always call step (i.e., meant for each epoch), and if we set
        # scheduler interval to "step", then the learning rate update will be wrong.
        if self.t_in_epochs:
            super().step(epoch=self._last_epoch)
        else:
            super().step_update(num_updates=self._last_epoch)


# copied from https://github.com/huangwenze/TISnet/blob/main/tisnet/model/utils.py#L33
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
