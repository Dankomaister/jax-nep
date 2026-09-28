"""One compact species batch; select the local scalar output."""

import jax
import numpy as np
import jax.numpy as jnp


def local_network(model, descriptors, types, accumulation_dtype=None):
    present, inverse = np.unique(np.asarray(types), return_inverse=True)
    if accumulation_dtype is None:
        pre = jnp.einsum(
            "id,sdh->ish",
            descriptors,
            jnp.asarray(model.input_weights[present]),
            precision="highest",
        )
    else:
        with jax.enable_x64():
            pre = jnp.sum(
                descriptors.astype(accumulation_dtype)[:, None, :, None]
                * jnp.asarray(model.input_weights[present], accumulation_dtype)[None],
                axis=2,
            ).astype(descriptors.dtype)
    hidden = jnp.tanh(pre - jnp.asarray(model.hidden_bias[present]))
    local = jnp.sum(hidden * jnp.asarray(model.output_weights[present]), axis=-1)
    return local[jnp.arange(len(types)), jnp.asarray(inverse)] - model.output_bias
