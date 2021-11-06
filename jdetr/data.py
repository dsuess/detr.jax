# pylint: disable=no-value-for-parameter,unexpected-keyword-arg
from typing import Callable, Tuple

import tensorflow as tf

DataTuple = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]


def resize_square(
    new_size: int,
) -> Callable[[DataTuple], DataTuple]:
    new_size = tf.constant(new_size, dtype=tf.float32)

    def inner(data: DataTuple) -> DataTuple:
        image, boxes, labels = data
        shape = tf.cast(tf.shape(image), tf.float32)
        box_scale_factor = tf.concat((shape[:2], shape[:2]), axis=0)
        boxes = boxes * box_scale_factor[None]

        # resize
        scale_factor = new_size / tf.reduce_max(shape[:2])
        new_shape = tf.round(shape[:2] * scale_factor)
        image = tf.image.resize(image, tf.cast(new_shape, tf.int32))
        boxes = boxes * scale_factor

        # pad
        pad_lt = tf.math.floordiv(new_size - new_shape, 2)
        pad_rb = new_size - new_shape - pad_lt
        image = tf.pad(image, [[pad_lt[0], pad_rb[0]], [pad_lt[1], pad_rb[1]], [0, 0]])
        boxes += tf.concat((pad_lt, pad_lt), axis=0)[None]
        boxes /= new_size

        return image, boxes, labels

    return inner


def pad_boxes(max_boxes: int) -> Callable[[DataTuple], DataTuple]:
    def inner(data: DataTuple) -> DataTuple:
        image, boxes, labels = data
        n_boxes = tf.shape(boxes)[0]
        num_pad_boxes = tf.maximum(max_boxes - n_boxes, 0)

        boxes = tf.concat(
            (
                boxes[:max_boxes],
                tf.zeros((num_pad_boxes, 4), dtype=boxes.dtype),
            ),
            axis=0,
        )
        labels = tf.concat(
            (
                labels[:max_boxes] + 1,
                tf.zeros((num_pad_boxes,), dtype=labels.dtype),
            ),
            axis=0,
        )
        return image, boxes, labels

    return inner
