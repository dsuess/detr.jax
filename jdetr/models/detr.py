import haiku as hk
import jax
from jax import numpy as jnp

from jdetr._typing import JaxArray

from .backbone import ResNet50
from .positional_encoding import sinusoidal_encoding

__all__ = ["DETR"]


class DETR(hk.Module):
    """
    >>> from jdetr.utils import Init
    >>> x = jnp.zeros((1, 224, 224, 3))
    >>> y = Init(DETR, 2).__call__(x, is_training=True)

    """

    def __init__(self, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = hk.Sequential(
            [ResNet50(0, resnet_v2=True), hk.Conv2D(hidden_dim, kernel_shape=1)]
        )
        self.num_pos_features = 64

    def __call__(self, image: JaxArray, is_training: bool):
        # FIXME Make work in channel_first
        assert image.shape[1] == image.shape[2]

        cnn_features = self.backbone(image, is_training=is_training)
        _, grid_size, _, _ = cnn_features.shape
        cnn_features = jnp.reshape(
            cnn_features, (cnn_features.shape[0], -1, cnn_features.shape[-1])
        )
        posititional_encoding = sinusoidal_encoding(grid_size, self.num_pos_features)
        posititional_encoding = jnp.tile(
            posititional_encoding.reshape(1, -1, posititional_encoding.shape[-1]),
            (cnn_features.shape[0], 1, 1),
        )
        features = jnp.concatenate([cnn_features, posititional_encoding], axis=-1)
