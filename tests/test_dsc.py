"""Correctness checks for the Dice helpers, incl. absent-class NaN semantics."""

from __future__ import annotations

import numpy as np
import pytest

from course_utils.dsc import dice_for_label, dice_per_label, nanmean_dice


def test_perfect_overlap_is_one():
    a = np.array([1, 1, 0, 0])
    assert dice_for_label(a, a, 1) == pytest.approx(1.0)


def test_no_overlap_is_zero():
    pred = np.array([1, 0])
    ref = np.array([0, 1])
    assert dice_for_label(pred, ref, 1) == pytest.approx(0.0)


def test_half_overlap_is_point_five():
    # pred has labels {1,1}, ref has {1,0}: intersection 1, denom 3 -> 2*1/3? No:
    # |A|=2 (positions 0,1), |B|=1 (position 0), intersection=1 -> 2*1/(2+1)=0.667.
    # Construct an exact 0.5 case instead: |A|=2, |B|=2, intersection=1 -> 2/4=0.5.
    pred = np.array([1, 1, 0, 0])
    ref = np.array([1, 0, 1, 0])
    assert dice_for_label(pred, ref, 1) == pytest.approx(0.5)


def test_absent_label_is_nan():
    zeros = np.zeros(4, dtype=int)
    assert np.isnan(dice_for_label(zeros, zeros, 1))


def test_nanmean_drops_absent_classes():
    pred = np.array([1, 1, 0, 0])
    ref = np.array([1, 0, 1, 0])
    scores = dice_per_label(pred, ref, labels=[1, 2])  # label 2 absent -> NaN
    assert np.isnan(scores[2])
    # nanmean over {1,2} should equal the label-1 score (0.5), ignoring the NaN.
    assert nanmean_dice(scores, labels=[1, 2]) == pytest.approx(0.5)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        dice_per_label(np.zeros(3), np.zeros(4), labels=[1])
