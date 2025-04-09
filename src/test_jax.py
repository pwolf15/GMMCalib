import jax.numpy as jnp
from jax import random

key = random.PRNGKey(0)
x = random.normal(key, (1000, 1000))
y = x @ x.T
