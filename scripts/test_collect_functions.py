import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt

def collect_hessian(pytree_hessian):

    # base case
    if isinstance(pytree_hessian, ArrayLike):

        return pytree_hessian
    
    # bottom recursive case - concatenate horizontally
    elif isinstance(pytree_hessian[list(pytree_hessian.keys())[0]], ArrayLike):

        return jnp.concatenate([collect_hessian(pytree_hessian[key]) for key in pytree_hessian.keys()], axis=1)


    # intermediate recursive case - concatenate vertically
    else:

        return jnp.concatenate([collect_hessian(pytree_hessian[key]) for key in pytree_hessian.keys()], axis=0)
    

def collect_gradient(pytree_gradient):

    # base case
    if isinstance(pytree_gradient, ArrayLike):

        return pytree_gradient
    
    # intermediate recursive case - concatenate vertically
    else:

        return jnp.concatenate([collect_hessian(pytree_gradient[key]) for key in pytree_gradient.keys()], axis=0)

fake_hessian_tree = {'w1': {'w1': jnp.array([[1]]), 'w2': jnp.array([[2]])}, 'w2': {'w1': jnp.array([[3]]), 'w2': jnp.array([[4]])}}
fake_hessian = collect_hessian(fake_hessian_tree)

fake_gradient_tree = {'w1': jnp.arange(5, 8), 'w2': jnp.arange(5)}
fake_gradient = collect_gradient(fake_gradient_tree)

print('')