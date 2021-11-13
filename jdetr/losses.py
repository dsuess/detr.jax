from typing import Callable, Tuple

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


class SetCriterion:
    def __init__(
        self,
        matcher_l1_weight: float = 1.0,
        matcher_giou_weight: float = 1.0,
        matcher_label_weight: float = 1.0,
        matcher: Callable[[JaxArray], JaxArray] = hungarian_single,
    ):
        self.matcher = matcher
        self.matcher_l1_weight = matcher_l1_weight
        self.matcher_giou_weight = matcher_giou_weight
        self.matcher_label_weight = matcher_label_weight

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
            self.matcher_l1_weight * box_loss
            + self.matcher_giou_weight * giou_loss
            + self.matcher_label_weight * labels_loss
        )
        idx = self.matcher(cost_matrix)
        return jnp.moveaxis(idx, 0, -1)

    def __call__(
        self,
        box_pred: JaxArray,
        box_true: JaxArray,
        labels_pred: JaxArray,
        labels_true: JaxArray,
    ) -> JaxArray:
        ...
