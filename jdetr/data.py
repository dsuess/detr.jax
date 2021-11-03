from typing import Callable, Tuple

import tensorflow as tf

DataTuple = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]


def resize_square(
    new_size: int,
) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], DataTuple]:
    """
    >>> import numpy as np
    >>> image = np.zeros((640, 480, 3), dtype=np.uint8)
    >>> boxes = np.array([[0.43910939, 0.79495835, 0.7267344 , 0.9483542 ],
    ...                  [0.08571875, 0.31514582, 0.6987031 , 0.9939375 ],
    ...                  [0.344125  , 0.        , 0.63554686, 0.32766667],
    ...                  [0.54446876, 0.80670834, 0.5626875 , 0.8576458 ]])
    >>> labels = np.array([0, 1, 2, 3])
    >>> image_, boxes_, labels_ = resize_square(1000)(image, boxes, labels)
    >>> tuple(image_.shape)
    (1000, 1000, 3)
    >>> assert tuple(labels) == tuple(labels_)
    >>> assert np.all(boxes_ <= 1) and np.all(boxes_ >= 0)

    """
    # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
    new_size = tf.constant(new_size, dtype=tf.float32)

    def inner(image: tf.Tensor, boxes: tf.Tensor, labels: tf.Tensor) -> DataTuple:
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
