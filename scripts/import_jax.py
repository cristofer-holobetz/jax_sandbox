import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap, jacfwd, jacrev
from jax import random


@jit
def two_layer_network_loss(W, x, target_multiple):
    
  return jnp.mean((W[0] * W[1] * x - target_multiple * x)**2)

rng = np.random.default_rng()

init_weights_list = []
final_weights_list = []

for i in np.arange(100):

  TARGET_MULTIPLE = 2
  W0 = rng.uniform(-4, 4, size=2) + 1e-3
  W = W0

  for epoch in np.arange(1, 10):
    
    input = jnp.array(rng.poisson(4, size=1) + 1e-3)

    def two_layer_network_loss(W, x, target_multiple):
        
      return jnp.mean((W[0] * W[1] * x - target_multiple * x)**2)

    grad_network = grad(two_layer_network_loss, argnums=(0))
    gradient = grad_network(W, input, TARGET_MULTIPLE)

    def hessian(f):
        return jacfwd(jacrev(f))
    
    loss = two_layer_network_loss(W, input, TARGET_MULTIPLE)

    H = hessian(two_layer_network_loss)(W, input, TARGET_MULTIPLE)
    H_inv = jnp.linalg.inv(H)
    print(f'H: {H}')
    print(f'H_inv: ')
    print(f'loss: {loss}\n \n')
    print('Epoch {0}'.format(epoch))

    update_vector = jnp.linalg.matmul(H_inv, gradient)
    
    if loss > 0:

      W = W - jnp.linalg.matmul(H_inv, gradient)

    #W = W - 1e-3 * gradient

  init_weights_list.append(W0)
  final_weights_list.append(W)

init_weights_arr = np.array(init_weights_list)
final_weights_arr = np.array(final_weights_list)

fig, ax = plt.subplots()

ax.scatter(init_weights_arr[:, 0], init_weights_arr[:, 1], color='tab:blue', label='initial weights')
ax.scatter(final_weights_arr[:, 0], final_weights_arr[:, 1], color='tab:orange', label='final weights')

target_curve_x_left = np.linspace(-4, 0, 500)
target_curve_x_right = np.linspace(0, 4, 500)

target_curve_y_left = TARGET_MULTIPLE / target_curve_x_left
target_curve_y_right = TARGET_MULTIPLE / target_curve_x_right

ax.plot(target_curve_x_left, target_curve_y_left, color='tab:green', label='loss = 0 curve')
ax.plot(target_curve_x_right, target_curve_y_right, color='tab:green')

for i in np.arange(init_weights_arr.shape[0]):

  ax.plot([init_weights_arr[i, 0], final_weights_arr[i, 0]], [init_weights_arr[i, 1], final_weights_arr[i, 1]])

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)

ax.set_title('Learned weights under Newton\'s Method')
ax.set_xlabel('weight 1 value')
ax.set_ylabel('weight 2 value')

ax.legend()

plt.show()

print('')