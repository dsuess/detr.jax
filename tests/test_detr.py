from typing import Tuple

import haiku as hk
import jax
from jax import numpy as jnp

from jdetr import models
from jdetr._typing import JaxArray


def predict_detr(x: JaxArray) -> Tuple[JaxArray, JaxArray]:
    backbone = models.ResNet50(0, resnet_v2=True)
    transformer = models.Transformer(
        feature_dim=16,
        num_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        feedforward_dim=32,
    )
    detr = models.DETR(backbone, transformer, num_classes=6, num_queries=4)
    return detr(x, is_training=True)


def test_detr_output_shape():
    func = hk.transform_with_state(predict_detr)
    x = jnp.zeros((1, 96, 96, 3))
    params, state = func.init(jax.random.PRNGKey(0), x)
    (box_coords, class_logits), _ = func.apply(params, state, jax.random.PRNGKey(0), x)

    assert box_coords.shape == (1, 4, 4)
    assert class_logits.shape == (1, 4, 7)
