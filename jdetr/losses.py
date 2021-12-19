from typing import Callable, NamedTuple, Tuple

import jax
from jax import numpy as jnp
from scenic.model_lib.matchers.hungarian_jax import hungarian_single

from jdetr._typing import JaxArray


def box_area(boxes: JaxArray) -> JaxArray:
    width = boxes[..., 2] - boxes[..., 0]
    height = boxes[..., 3] - boxes[..., 1]
    return width * height


def box_intersection_and_union(
    boxes1: JaxArray, boxes2: JaxArray
) -> Tuple[JaxArray, JaxArray]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    x1 = jnp.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = jnp.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = jnp.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = jnp.minimum(boxes1[..., 3], boxes2[..., 3])

    width = jnp.maximum(x2 - x1, 0)
    height = jnp.maximum(y2 - y1, 0)

    intersection = width * height
    union = area1 + area2 - intersection
    return intersection, union


def box_giou(boxes1: JaxArray, boxes2: JaxArray):
    intersection, union = box_intersection_and_union(boxes1, boxes2)

    x1 = jnp.minimum(boxes1[..., 0], boxes2[..., 0])
    y1 = jnp.minimum(boxes1[..., 1], boxes2[..., 1])
    x2 = jnp.maximum(boxes1[..., 2], boxes2[..., 2])
    y2 = jnp.maximum(boxes1[..., 3], boxes2[..., 3])

    width = jnp.maximum(x2 - x1, 0)
    height = jnp.maximum(y2 - y1, 0)
    area = width * height

    return intersection / union - (area - union) / area


class SetCriterionLosses(NamedTuple):
    l1_loss: JaxArray
    giou_loss: JaxArray
    labels_loss: JaxArray


class SetCriterion:
    def __init__(
        self,
        l1_weight: float = 1.0,
        giou_weight: float = 1.0,
        matcher_label_weight: float = 1.0,
        negative_weight: float = 0.1,
        matcher: Callable[[JaxArray], JaxArray] = hungarian_single,
    ):
        self.matcher = matcher
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight
        self.matcher_label_weight = matcher_label_weight
        self.negative_weight = negative_weight

    def match_predictions(
        self,
        box_pred: JaxArray,
        box_true: JaxArray,
        labels_pred: JaxArray,
        labels_true: JaxArray,
    ) -> JaxArray:

        box_loss = jnp.linalg.norm(
            box_pred[:, None] - box_true[None, :], axis=-1, ord=1
        )

        labels_pred = jax.nn.softmax(labels_pred, axis=-1)
        labels_loss = -labels_pred[:, labels_true]
        giou_loss = 1 - box_giou(box_pred[:, None], box_true[None, :])

        cost_matrix = (
            self.l1_weight * box_loss
            + self.giou_weight * giou_loss
            + self.matcher_label_weight * labels_loss
        )
        is_valid = (labels_true > 0).astype(cost_matrix.dtype)[None, :]
        cost_upper_bound = 4 * self.l1_weight + self.giou_weight
        cost_matrix = cost_matrix * is_valid + cost_upper_bound * (1 - is_valid)

        idx = self.matcher(cost_matrix)
        return jnp.moveaxis(idx, 0, -1)

    def __call__(
        self,
        box_pred: JaxArray,
        box_true: JaxArray,
        labels_pred: JaxArray,
        labels_true: JaxArray,
    ) -> Tuple[JaxArray, SetCriterionLosses]:
        # pylint: disable=too-many-locals

        assert (
            box_pred.shape == box_true.shape
        ), "SetCriterion only implemented for square cost matrices"
        assignment = self.match_predictions(
            box_pred, box_true, labels_pred, labels_true
        )
        idx_pred, idx_true = assignment[..., 0], assignment[..., 1]

        # loss for matched boxes
        box_p_valid, labels_p_valid = box_pred[idx_pred], labels_pred[idx_pred]
        box_t_valid, labels_t_valid = box_true[idx_true], labels_true[idx_true]
        is_valid = (labels_t_valid > 0).astype(jnp.float32)

        l1_loss = jnp.sum(
            is_valid * jnp.linalg.norm(box_p_valid - box_t_valid, axis=-1, ord=1)
        )
        giou_loss = jnp.sum(is_valid * (1 - box_giou(box_p_valid, box_t_valid)))

        label_weight = is_valid + (1 - is_valid) * self.negative_weight
        labels_loss = jnp.sum(
            -label_weight
            * jax.nn.log_softmax(labels_p_valid)[
                jnp.arange(labels_p_valid.shape[0]), labels_t_valid
            ]
        )

        loss = self.l1_weight * l1_loss + self.giou_weight * giou_loss + labels_loss

        return loss, SetCriterionLosses(l1_loss, giou_loss, labels_loss)
