from typing import TypeAlias, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    # Only during type‐checking do we import jax.numpy
    import jax.numpy as jnp
    ArrayLike: TypeAlias = Union[np.ndarray, jnp.ndarray]
else:
    # At runtime, we only need to know that ArrayLike is at least a numpy.ndarray
    ArrayLike: TypeAlias = np.ndarray
