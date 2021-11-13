import numpy as np
import pytest as pt


@pt.fixture(scope="session")
def rgen():
    return np.random.Generator(np.random.PCG64(seed=12345))
