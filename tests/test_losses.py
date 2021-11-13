import itertools as it
from typing import List, Tuple

import jax.numpy as jnp
import numpy as np
import pytest as pt

from jdetr import losses
from jdetr._typing import JaxArray


@pt.fixture(scope="module")
def some_boxes() -> Tuple[np.ndarray, np.ndarray]:
    boxes = np.array(
        [
            [0.43910939, 0.79495835, 0.7267344, 0.9483542],
            [0.08571875, 0.31514582, 0.6987031, 0.9939375],
            [0.344125, 0.0, 0.63554686, 0.32766667],
            [0.54446876, 0.80670834, 0.5626875, 0.8576458],
            [0, 0, 1, 1],
        ]
    )
    labels = np.array([1, 2, 3, 4, 5])
    return boxes, labels


def to_logits(labels: np.ndarray) -> np.ndarray:
    logits = np.zeros((5, 1 + np.max(labels)), dtype=jnp.float32)
    logits[np.arange(len(logits)), labels] = 1.0
    return logits


@pt.mark.parametrize("permutation", list(it.permutations(range(5)))[::10])
@pt.mark.parametrize("weights", [(1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
def test_set_criterion_matching(
    some_boxes: Tuple[np.ndarray, np.ndarray],
    permutation: List[int],
    weights: Tuple[float, float, float],
):
    boxes, labels = some_boxes
    logits = to_logits(labels)
    kwargs = {
        name: val
        for name, val in zip(
            ["matcher_l1_weight", "matcher_label_weight", "matcher_giou_weight"],
            weights,
        )
    }
    criterion = losses.SetCriterion(**kwargs)
    permutation_arr = np.array(permutation)

    idx = criterion.match_predictions(
        boxes[permutation_arr], boxes, logits[permutation_arr], labels  # type: ignore
    )
    idx_pred, idx_true = np.array(idx).T

    np.testing.assert_array_equal(permutation_arr[idx_pred], idx_true)


@pt.mark.parametrize("permutation", list(it.permutations(range(5)))[::5])
def test_set_criterion_incomplete_matching(
    some_boxes: Tuple[np.ndarray, np.ndarray], permutation: List[int]
):
    boxes, labels = some_boxes
    logits = to_logits(labels)
    criterion = losses.SetCriterion()
    permutation_arr = np.array(permutation)

    idx = criterion.match_predictions(
        boxes[permutation_arr],
        boxes[::2],
        logits[permutation_arr],
        labels[::2],
    )
    idx_pred, idx_true = np.array(idx).T

    np.testing.assert_array_equal(permutation_arr[idx_pred], 2 * idx_true)
