import haiku as hk
import jax

from jdetr._typing import JaxArray

__all__ = ["ResNet50"]


class HeadlessResnet(hk.nets.ResNet):
    def __call__(
        self, inputs: JaxArray, is_training: bool, test_local_stats: bool = False
    ) -> JaxArray:
        out = inputs
        out = self.initial_conv(out)
        if not self.resnet_v2:
            out = self.initial_batchnorm(out, is_training, test_local_stats)
            out = jax.nn.relu(out)

        out = hk.max_pool(
            out, window_shape=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding="SAME"
        )

        for block_group in self.block_groups:
            out = block_group(out, is_training, test_local_stats)

        return out


class ResNet50(HeadlessResnet, hk.nets.ResNet50):
    """
    >>> from jdetr.utils import Init
    >>> x = jnp.zeros((1, 224, 224, 3))
    >>> y = Init(ResNet50, 0).__call__(x, is_training=True)
    >>> y.shape
    (1, 7, 7, 2048)
    """

    ...
