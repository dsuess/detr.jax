import numpy as np

from jdetr.data import pad_boxes, resize_square


def test_resize_square():
    image = np.zeros((640, 480, 3), dtype=np.uint8)
    boxes = np.array(
        [
            [0.43910939, 0.79495835, 0.7267344, 0.9483542],
            [0.08571875, 0.31514582, 0.6987031, 0.9939375],
            [0.344125, 0.0, 0.63554686, 0.32766667],
            [0.54446876, 0.80670834, 0.5626875, 0.8576458],
        ]
    )
    labels = np.array([0, 1, 2, 3])

    image_, boxes_, labels_ = resize_square(1000)(image, boxes, labels)
    assert tuple(image_.shape) == (1000, 1000, 3)
    assert tuple(labels) == tuple(labels_)
    assert np.all(boxes_.numpy() <= 1) and np.all(boxes_.numpy() >= 0)


def test_pad_boxes():
    image = np.zeros((640, 480, 3), dtype=np.uint8)
    boxes = np.array(
        [
            [0.43910939, 0.79495835, 0.7267344, 0.9483542],
            [0.08571875, 0.31514582, 0.6987031, 0.9939375],
            [0.344125, 0.0, 0.63554686, 0.32766667],
        ]
    )
    labels = np.array([1, 2, 3])

    _, boxes_, labels_ = pad_boxes(10)(image, boxes, labels)
    boxes_, labels_ = boxes_.numpy(), labels_.numpy()

    assert np.all(labels_[:3] > 0) and np.all(labels_[3:] == 0)
    assert np.all(boxes == boxes_[:3]) and np.all(labels == labels_[:3] - 1)
