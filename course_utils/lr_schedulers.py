"""Two-phase fine-tuning LR schedule: linear warm-up, then offset polynomial decay.

Teaching copy of the schedulers used by the PANTHER / TBI nnU-Net fine-tuning
trainers. Behaviour is identical to the lab originals; only the docstrings are
expanded for teaching.

The schedule the course centres on (with ``max_lr = 1e-3``, ``warmup = 50``,
``total = 1000``):

* **Phase 1 (epochs 0..49) — warm-up.** ``lr = max_lr / warmup * (epoch + 1)``.
  So epoch 0 -> ``2e-5`` (NOT zero) and epoch 49 -> ``1e-3``. The whole network
  trains during this phase.
* **Phase 2 (epochs 50..999) — decay.** offset polynomial:
  ``lr = init_lr * (1 - (epoch - warmup) / (total - warmup)) ** 0.9``.
  Epoch 50 -> ``1e-3`` and it decays *towards* (not exactly to) zero.

There is no parameter-selection / freezing stage; this is purely an LR technique.
"""

from __future__ import annotations

try:  # torch >= 2.0 exposes the public name
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:  # pragma: no cover - very old torch
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class Lin_incr_LRScheduler(LRScheduler):
    """Linearly increase LR from ``max_lr / max_steps`` up to ``max_lr``.

    At step ``s`` the LR is ``max_lr / max_steps * (1 + s)``. With
    ``max_lr=1e-3`` and ``max_steps=50`` that is ``2e-5`` at step 0 and ``1e-3``
    at step 49 -- the warm-up does not start at literal zero.
    """

    def __init__(
        self,
        optimizer,
        max_lr: float,
        max_steps: int,
        current_step: int | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.max_steps = max_steps
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step: int | None = None) -> None:
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.max_lr / self.max_steps * (1 + current_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class PolyLRScheduler_offset(LRScheduler):
    """Polynomial LR decay that only begins after a warm-up offset.

    Steps before ``start_step`` are clamped to the start (so the decay phase
    "sees" step 0 at epoch ``start_step``), then
    ``lr = initial_lr * (1 - t / (max_steps - start_step)) ** exponent`` where
    ``t = step - start_step``.
    """

    def __init__(
        self,
        optimizer,
        initial_lr: float,
        max_steps: int,
        start_step: int,
        exponent: float = 0.9,
        current_step: int | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps - start_step
        self.start_step = start_step
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step: int | None = None) -> None:
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        current_step = max(current_step - self.start_step, 0)
        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


def two_phase_lr(
    epoch: int,
    *,
    max_lr: float = 1e-3,
    warmup_epochs: int = 50,
    total_epochs: int = 1000,
    exponent: float = 0.9,
) -> float:
    """Pure-function LR for a given epoch -- handy for plotting the curve.

    Matches :class:`Lin_incr_LRScheduler` (warm-up) and
    :class:`PolyLRScheduler_offset` (decay) without needing a torch optimizer.

    >>> round(two_phase_lr(0), 8)
    2e-05
    >>> two_phase_lr(49)
    0.001
    >>> two_phase_lr(50)
    0.001
    >>> 0.0 < two_phase_lr(999) < 1e-4
    True
    """
    if not 0 <= epoch < total_epochs:
        raise ValueError(f"epoch {epoch} out of range [0, {total_epochs})")
    if epoch < warmup_epochs:
        return max_lr / warmup_epochs * (epoch + 1)
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return max_lr * (1.0 - progress) ** exponent
