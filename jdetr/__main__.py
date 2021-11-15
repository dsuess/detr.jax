from typing import Tuple

import elegy as eg
import haiku as hk
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import typer

from jdetr import data, losses, models
from jdetr._typing import JaxArray


def detr_forward(x: JaxArray, training: bool):
    backbone = models.ResNet50(num_classes=0, resnet_v2=True)
    transformer = models.Transformer(
        feature_dim=256,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        feedforward_dim=2048,
        dropout_rate=0.1,
    )
    detr = models.DETR(backbone, transformer, num_classes=80, num_queries=100)
    return detr(x, is_training=training)


def loss_fn(y_pred: Tuple[JaxArray, JaxArray], y_true: Tuple[JaxArray, JaxArray]):
    box_pred, labels_pred = y_pred
    box_true, labels_true = y_true
    loss, _ = losses.SetCriterion()(box_pred, box_true, labels_pred, labels_true)
    return loss


def train(batch_size: int = 4, data_dir: str = "__pycache__/coco"):
    # pylint: disable=unused-argument,invalid-name,unused-variable
    ds = tfds.load(
        "coco/2017",
        split="train",
        download=True,
        data_dir=data_dir,
    )

    ds = (
        ds.map(data.get_elems, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(data.resize_square(600), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(data.pad_boxes(100), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .shuffle(1024)
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    (x,) = ds.take(1)
    __import__("pdb").set_trace()  # FIXME PDB

    model = eg.Model(
        hk.transform_with_state(detr_forward),
        loss=loss_fn,
        optimizer=optax.adamw(learning_rate=1e-3),
    )

    model.fit(
        ds,
        epochs=100,
        callbacks=[eg.callbacks.TensorBoard("logdir", update_freq="batch")],
    )


if __name__ == "__main__":
    app = typer.Typer()
    app.command()(train)
    app()
