"""Utility for rl module."""

import numpy as np

from gymnasium import spaces
from gymnasium.spaces.utils import flatdim, flatten, unflatten
from numpy.typing import NDArray


@flatdim.register(spaces.Discrete)
def _flatdim_discrete(space: spaces.Discrete) -> int:
    # Number of bits required to represent the space
    return int(np.log2(space.n)) + 1


@flatten.register(spaces.Discrete)
def _flatten_discrete(space: spaces.Discrete, x: np.int64) -> NDArray[np.int64]:
    # Convert the integer into a binary array
    x -= space.start
    width = flatdim(space)
    result = np.zeros(width, dtype=np.int64)
    for i in range(width):
        result[i] = x & 1
        x >>= 1
    return result[::-1]


@unflatten.register(spaces.Discrete)
def _unflatten_discrete(space: spaces.Discrete, x: NDArray[np.int64]) -> np.int64:
    # Convert the binary array into the original integer
    result = 0
    for bit in x:
        result = (result << 1) | bit
    return result + space.start
