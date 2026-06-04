"""Correctness checks for the two-phase LR schedule.

Pins the load-bearing numbers the course teaches: the warm-up does NOT start at
zero (epoch 0 -> 2e-5), it peaks at 1e-3 at epoch 49, the decay phase re-starts
at 1e-3 at epoch 50, and the LR decays toward (but not exactly to) zero.
"""

from __future__ import annotations

import pytest
import torch

from course_utils.lr_schedulers import (
    Lin_incr_LRScheduler,
    PolyLRScheduler_offset,
    two_phase_lr,
)

MAX_LR = 1e-3
WARMUP = 50
TOTAL = 1000


def _make_optimizer():
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([param], lr=MAX_LR)


def test_warmup_does_not_start_at_zero():
    opt = _make_optimizer()
    warm = Lin_incr_LRScheduler(opt, max_lr=MAX_LR, max_steps=WARMUP)
    warm.step(0)
    assert opt.param_groups[0]["lr"] == pytest.approx(2e-5)


def test_warmup_peaks_at_max_lr():
    opt = _make_optimizer()
    warm = Lin_incr_LRScheduler(opt, max_lr=MAX_LR, max_steps=WARMUP)
    warm.step(WARMUP - 1)
    assert opt.param_groups[0]["lr"] == pytest.approx(MAX_LR)


def test_decay_restarts_at_max_lr_at_switch():
    opt = _make_optimizer()
    poly = PolyLRScheduler_offset(opt, initial_lr=MAX_LR, max_steps=TOTAL, start_step=WARMUP)
    poly.step(WARMUP)
    assert opt.param_groups[0]["lr"] == pytest.approx(MAX_LR)


def test_decay_approaches_zero_but_stays_positive():
    opt = _make_optimizer()
    poly = PolyLRScheduler_offset(opt, initial_lr=MAX_LR, max_steps=TOTAL, start_step=WARMUP)
    poly.step(TOTAL - 1)
    lr = opt.param_groups[0]["lr"]
    assert 0.0 < lr < 1e-5


@pytest.mark.parametrize(
    "epoch,expected",
    [(0, 2e-5), (49, 1e-3), (50, 1e-3)],
)
def test_two_phase_lr_pure_function(epoch, expected):
    assert two_phase_lr(epoch) == pytest.approx(expected)


def test_two_phase_lr_decays_toward_zero():
    assert 0.0 < two_phase_lr(999) < 1e-4


def test_two_phase_lr_rejects_out_of_range():
    with pytest.raises(ValueError):
        two_phase_lr(TOTAL)
    with pytest.raises(ValueError):
        two_phase_lr(-1)
