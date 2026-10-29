import torch
import math
from math import ceil
from itertools import permutations
from torchaudio.transforms import MelScale
from dataclasses import dataclass, field, fields
from typing import List, Type, Any, Callable, Optional, Union
from loguru import logger
from ..decorators import *


class ValidationLossReduceLROnPlateau:
    """Reduce LR after an exact number of non-improving metric evaluations."""

    def __init__(
        self,
        optimizer,
        start_epoch=1,
        patience=3,
        factor=0.8,
        min_lr=0.0,
        mode="min",
        threshold=0.0,
    ):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'.")
        if not 0.0 < float(factor) < 1.0:
            raise ValueError("factor must be between 0 and 1.")
        if int(patience) < 1:
            raise ValueError("patience must be at least 1.")

        self.optimizer = optimizer
        self.start_epoch = int(start_epoch)
        self.patience = int(patience)
        self.factor = float(factor)
        self.min_lr = float(min_lr)
        self.mode = str(mode)
        self.threshold = float(threshold)
        self.best = None
        self.num_bad_epochs = 0
        self.last_epoch = -1

    def step(self, *args, **kwargs):
        self.last_epoch += 1

    def step_metric(self, metric, epoch):
        self.last_epoch = int(epoch)
        if int(epoch) < self.start_epoch or metric is None:
            return False

        metric = float(metric)
        if not math.isfinite(metric):
            return False

        improved = self.best is None
        if self.best is not None:
            if self.mode == "min":
                improved = metric < self.best - self.threshold
            else:
                improved = metric > self.best + self.threshold

        if improved:
            self.best = metric
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        if self.num_bad_epochs < self.patience:
            return False

        changed = False
        for group in self.optimizer.param_groups:
            old_lr = float(group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if new_lr < old_lr:
                group["lr"] = new_lr
                changed = True
        self.num_bad_epochs = 0
        return changed

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "start_epoch": self.start_epoch,
            "patience": self.patience,
            "factor": self.factor,
            "min_lr": self.min_lr,
            "mode": self.mode,
            "threshold": self.threshold,
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state_dict):
        self.best = state_dict.get("best", self.best)
        self.num_bad_epochs = int(state_dict.get("num_bad_epochs", 0))
        self.last_epoch = int(state_dict.get("last_epoch", -1))


class FixedEpochStepLR(torch.optim.lr_scheduler.LRScheduler):
    """Hold LR, then decay it at fixed epoch boundaries with a lower bound."""

    def __init__(
        self,
        optimizer,
        hold_epochs=60,
        step_size=10,
        gamma=0.8,
        min_lr=1.0e-5,
        last_epoch=-1,
    ):
        if int(hold_epochs) < 1:
            raise ValueError("hold_epochs must be at least 1.")
        if int(step_size) < 1:
            raise ValueError("step_size must be at least 1.")
        if not 0.0 < float(gamma) < 1.0:
            raise ValueError("gamma must be between 0 and 1.")

        self.hold_epochs = int(hold_epochs)
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch=last_epoch)

    def _decay_count(self, completed_epochs):
        if int(completed_epochs) < self.hold_epochs:
            return 0
        return 1 + (int(completed_epochs) - self.hold_epochs) // self.step_size

    def _lr_at(self, base_lr, completed_epochs):
        decay_count = self._decay_count(completed_epochs)
        return max(float(base_lr) * self.gamma ** decay_count, self.min_lr)

    def get_lr(self):
        return [self._lr_at(base_lr, self.last_epoch) for base_lr in self.base_lrs]

    def will_decay_after(self, completed_epoch):
        return any(
            self._lr_at(base_lr, completed_epoch)
            < self._lr_at(base_lr, int(completed_epoch) - 1)
            for base_lr in self.base_lrs
        )


@logger_wraps()
@dataclass(slots=False)
class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
    optimizer: torch.optim.Optimizer
    warmup_steps: int
    last_epoch: int = field(default=-1)
    lr_lambda: callable = field(init=False)

    def __post_init__(self):
        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(self.warmup_steps)
            return 1.0

        self.lr_lambda = lr_lambda
        super().__init__(self.optimizer, self.lr_lambda, last_epoch=self.last_epoch)
        
    def __repr__(self):
        # __init__
        class_name = self.__class__.__name__
        init_fields = [f for f in fields(self) if f.init]
        field_strs = [f"{field.name}={getattr(self, field.name)!r}" for field in init_fields]

        # __post_init__
        lr_lambda_repr = f"lr_lambda = {self.lr_lambda}"
        post_init_reprs = [lr_lambda_repr]

        return f"<{class_name}({', '.join(field_strs + post_init_reprs)})>"
