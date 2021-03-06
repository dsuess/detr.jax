import itertools as it
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest as pt

from jdetr._typing import JaxArray
from jdetr.losses import SetCriterion


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
            ["l1_weight", "matcher_label_weight", "giou_weight"],
            weights,
        )
    }
    criterion = SetCriterion(**kwargs)
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
    criterion = SetCriterion()
    permutation_arr = np.array(permutation)

    idx = criterion.match_predictions(
        boxes[permutation_arr],
        boxes[::2],
        logits[permutation_arr],
        labels[::2],
    )
    idx_pred, idx_true = np.array(idx).T

    np.testing.assert_array_equal(permutation_arr[idx_pred], 2 * idx_true)


def test_set_criterion_matching_batched(
    some_boxes: Tuple[np.ndarray, np.ndarray],
):
    boxes, labels = some_boxes
    permutations = np.array(list(it.permutations(range(5)))[::10])
    boxes = jnp.array([boxes] * len(permutations))
    boxes_permuted = jnp.array([b[idx] for b, idx in zip(boxes, permutations)])
    labels = np.array([labels] * len(permutations))
    logits = np.array([to_logits(l[perm]) for l, perm in zip(labels, permutations)])
    logits = jnp.array(10000 * logits - 5000)

    match = jax.vmap(SetCriterion().match_predictions)
    indices = match(boxes_permuted, boxes, logits, jnp.array(labels))

    for idx, perm in zip(indices, permutations):
        idx_pred, idx_true = np.array(idx).T
        np.testing.assert_array_equal(perm[idx_pred], idx_true)


def test_set_criterion_matching_with_invalid_boxes():
    # Create perfect matching with box that's invalid -> even with
    # label-weight disabled it should be ignored
    boxes_gt = jnp.array([[0, 0, 1, 1], [0.25, 0.25, 0.75, 0.75]])
    labels_gt = jnp.array([1, 0])
    boxes_pred = jnp.array([[0.25, 0.25, 0.75, 0.75]])
    labels_pred = jnp.array([[-1000, 1000]])

    criterion = SetCriterion(matcher_label_weight=0)
    (idx,) = criterion.match_predictions(boxes_pred, boxes_gt, labels_pred, labels_gt)
    assert idx.tolist() == [0, 0]


@pt.mark.parametrize("permutation", list(it.permutations(range(5)))[::10])
def test_set_criterion_loss(
    some_boxes: Tuple[np.ndarray, np.ndarray],
    permutation: List[int],
):
    boxes, labels = some_boxes
    logits = 10000 * to_logits(labels) - 5000
    criterion = SetCriterion()
    permutation_arr = np.array(permutation)

    loss, losses = criterion(
        jnp.array(boxes[permutation_arr]),
        jnp.array(boxes),
        jnp.array(logits[permutation_arr]),
        jnp.array(labels),
    )

    np.testing.assert_array_less(loss, 1e-7)


def test_batched_set_criterion_loss(
    some_boxes: Tuple[np.ndarray, np.ndarray],
):
    boxes, labels = some_boxes
    permutations = np.array(list(it.permutations(range(5)))[::10])
    boxes = jnp.array([boxes] * len(permutations))
    boxes_permuted = jnp.array([b[idx] for b, idx in zip(boxes, permutations)])
    labels = np.array([labels] * len(permutations))
    logits = np.array([to_logits(l[perm]) for l, perm in zip(labels, permutations)])
    logits = jnp.array(10000 * logits - 5000)
    criterion = jax.vmap(SetCriterion())

    loss, losses = criterion(boxes_permuted, boxes, logits, jnp.array(labels))

    np.testing.assert_array_less(loss, 1e-7)


@pt.mark.parametrize("permutation", list(it.permutations(range(5)))[::10])
def test_set_criterion_loss_with_invalid_boxes(
    some_boxes: Tuple[np.ndarray, np.ndarray],
    permutation: List[int],
):
    boxes, labels = some_boxes
    boxes_true = np.concatenate((boxes[:-2], np.zeros((2, 4))))
    labels_true = np.concatenate((labels[:-2], np.zeros((2,), dtype=np.int32)))

    logits = 10000 * to_logits(labels) - 5000
    criterion = SetCriterion()
    permutation_arr = np.array(permutation)

    loss, losses = criterion(
        jnp.array(boxes[permutation_arr]),
        jnp.array(boxes_true),
        jnp.array(logits[permutation_arr]),
        jnp.array(labels_true),
    )

    assert losses.l1_loss < 1e-7
    assert losses.giou_loss < 1e-7
    assert losses.labels_loss > 10.0


@pt.mark.parametrize("permutation", list(it.permutations(range(5)))[::10])
def test_set_criterion_loss_with_invalid_boxes_correctly_predicted(
    some_boxes: Tuple[np.ndarray, np.ndarray],
    permutation: List[int],
):
    boxes, labels = some_boxes
    boxes_true = np.concatenate((boxes[:-2], np.zeros((2, 4))))
    labels_true = np.concatenate((labels[:-2], np.zeros((2,), dtype=np.int32))).copy()
    labels[-2:] = 0

    logits = 10000 * to_logits(labels) - 5000
    criterion = SetCriterion()
    permutation_arr = np.array(permutation)

    loss, losses = criterion(
        jnp.array(boxes[permutation_arr]),
        jnp.array(boxes_true),
        jnp.array(logits[permutation_arr]),
        jnp.array(labels_true),
    )

    assert losses.l1_loss < 1e-7
    assert losses.giou_loss < 1e-7
    assert losses.labels_loss < 1e-7


@pt.mark.skip(reason="TODO: SetCriterion only implemented for square cost matrices")
@pt.mark.parametrize("permutation", list(it.permutations(range(5)))[::10])
def test_set_criterion_loss_missing_groundtruth(
    some_boxes: Tuple[np.ndarray, np.ndarray],
    permutation: List[int],
):
    boxes, labels = some_boxes
    boxes_true = np.concatenate((boxes[:-2], np.zeros((2, 4))))
    labels_true = np.concatenate((labels[:-2], np.zeros((2,), dtype=np.int32)))

    logits = 10000 * to_logits(labels) - 5000
    criterion = SetCriterion()
    permutation_arr = np.array(permutation)

    _, losses_with = criterion(
        jnp.array(boxes[permutation_arr]),
        jnp.array(boxes_true),
        jnp.array(logits[permutation_arr]),
        jnp.array(labels_true),
    )
    _, losses_without = criterion(
        jnp.array(boxes[permutation_arr]),
        jnp.array(boxes_true[:-2]),
        jnp.array(logits[permutation_arr]),
        jnp.array(labels_true[:-2]),
    )

    assert losses_with.l1_loss == losses_without.l1_loss
    assert losses_with.giou_loss == losses_without.giou_loss
    assert losses_with.labels_loss == losses_without.labels_loss
