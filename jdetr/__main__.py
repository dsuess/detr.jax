from typing import Any, Mapping, NamedTuple, Tuple

import haiku as hk
import jax
import optax
import typer
from haiku._src.data_structures import (  # pylint: disable=no-name-in-module
    FlatMap,
)

from jdetr import data, losses, models
from jdetr._typing import JaxArray, Number, PRNGKey


def build_forward_fn(num_classes: int, max_boxes: int = 100):
    def detr_forward(
        x: JaxArray, is_training: bool = True
    ) -> Tuple[JaxArray, JaxArray]:
        backbone = models.ResNet50(num_classes=0, resnet_v2=True)
        transformer = models.Transformer(
            feature_dim=256,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            feedforward_dim=2048,
            dropout_rate=0.1,
        )
        detr = models.DETR(
            backbone, transformer, num_classes=num_classes, num_queries=max_boxes
        )
        return detr(x, is_training=is_training)

    return detr_forward


class State(NamedTuple):
    step: int
    trainable_params: FlatMap
    stateful_params: FlatMap
    opt_state: FlatMap
    rng: PRNGKey


class Trainer:
    def __init__(self, forward_fn, loss_fn, optimizer):
        self.forward_fn = forward_fn
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    # @ft.partial(jax.jit, static_argnums=(0,))
    def init_states(self, rng: PRNGKey, batch: data.DataTuple) -> State:
        out_rng, init_rng = jax.random.split(rng)
        images, _, _ = batch
        trainable_params, stateful_params = self.forward_fn.init(init_rng, images)
        opt_state = self.optimizer.init(trainable_params)
        return State(
            step=0,
            trainable_params=trainable_params,
            stateful_params=stateful_params,
            opt_state=opt_state,
            rng=out_rng,
        )

    def forward(
        self,
        trainable_params: FlatMap,
        stateful_params: FlatMap,
        rng: PRNGKey,
        batch: data.DataTuple,
    ) -> Tuple[JaxArray, Mapping[str, Any]]:
        images, boxes, labels = batch
        (boxes_pred, labels_pred), stateful_params = self.forward_fn.apply(
            trainable_params, stateful_params, rng, images
        )
        # TODO Add pmap-averaging

        batch_num_valid_boxes = (labels > 0).sum()
        loss = (
            self.loss_fn(boxes_pred, boxes, labels_pred, labels) / batch_num_valid_boxes
        )
        return loss, {
            "loss": loss,
            "boxes_pred": boxes_pred,
            "labels_pred": labels_pred,
            "stateful_params": stateful_params,
            "num_valid_boxes": batch_num_valid_boxes,
        }

    def update_state(
        self, state: State, batch: data.DataTuple
    ) -> Tuple[State, Mapping[str, Number]]:
        new_rng_state, rng_state = jax.random.split(state.rng)
        grads, logs = jax.grad(self.forward, has_aux=True)(
            state.trainable_params, state.stateful_params, rng_state, batch
        )
        # TODO Add pmap grad-averaging
        updates, opt_state = self.optimizer.update(grads, state.opt_state)
        trainable_params = optax.apply_updates(state.trainable_params, updates)

        state = State(
            state.step + 1,
            trainable_params,
            logs.pop("stateful_params"),
            opt_state,
            new_rng_state,
        )

        return state, logs


def train(
    batch_size_per_device: int = 4,
    data_dir: str = "__pycache__/coco",
    max_boxes: int = 100,
    learning_rate: float = 3e-4,
):
    batch_size = batch_size_per_device * 1

    # FIXME Do sth with validation data
    train_data, _ = data.get_coco_datagen(batch_size, data_dir, max_boxes=max_boxes)
    forward_fn = hk.transform_with_state(
        build_forward_fn(train_data.n_classes, max_boxes=max_boxes)
    )
    loss_fn = jax.vmap(losses.SetCriterion())
    optimizer = optax.chain(
        optax.clip_by_global_norm(1), optax.adamw(learning_rate, b1=0.9, b2=0.99)
    )

    rng = jax.random.PRNGKey(428)
    (batch,) = train_data.data.take(1).as_numpy_iterator()
    trainer = Trainer(forward_fn, loss_fn, optimizer)
    state = trainer.init_states(rng, batch)
    for batch in train_data.data.as_numpy_iterator():
        state, logs = trainer.update_state(state, batch)
        print(logs)


if __name__ == "__main__":
    app = typer.Typer()
    app.command()(train)
    app()
