from typing import Tuple

import haiku as hk
import jax
from jax import numpy as jnp

from jdetr._typing import JaxArray

from .backbone import HeadlessResnet
from .positional_encoding import sinusoidal_encoding
from .transformer import Transformer

__all__ = ["DETR"]


class DETR(hk.Module):
    def __init__(
        self,
        backbone: HeadlessResnet,
        transformer: Transformer,
        num_classes: int,
        num_queries: int,
    ):
        super().__init__()
        self.backbone = hk.Sequential(
            [backbone, hk.Conv2D(transformer.feature_dim, kernel_shape=1)]
        )
        self.transformer = transformer
        self.num_classes = num_classes
        self.num_queries = num_queries

    def __call__(
        self, image: JaxArray, *, is_training: bool
    ) -> Tuple[JaxArray, JaxArray]:
        # FIXME Make work in channel_first
        assert image.shape[1] == image.shape[2]

        cnn_features = self.backbone(image, is_training=is_training)
        _, grid_size, _, _ = cnn_features.shape
        cnn_features = jnp.reshape(
            cnn_features, (cnn_features.shape[0], -1, cnn_features.shape[-1])
        )

        posititional_encoding = sinusoidal_encoding(
            grid_size, self.transformer.feature_dim // 2
        )
        # Tile the positional encoding to match batch size
        posititional_encoding = jnp.tile(
            posititional_encoding.reshape(1, -1, posititional_encoding.shape[-1]),
            (cnn_features.shape[0], 1, 1),
        )

        query_encoding = hk.get_parameter(
            "query_encoding",
            [1, self.num_queries, self.transformer.feature_dim],
            init=hk.initializers.TruncatedNormal(),
        )
        query_encoding = jnp.tile(query_encoding, (cnn_features.shape[0], 1, 1))
        features = self.transformer(
            cnn_features, posititional_encoding, query_encoding, is_training
        )

        box_regressor = hk.nets.MLP(
            output_sizes=[self.transformer.feature_dim] * 3 + [4]
        )
        box_coords = jax.nn.sigmoid(box_regressor(features))
        class_logits = hk.Linear(self.num_classes + 1)(features)
        return box_coords, class_logits
