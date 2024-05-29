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
grad_final_weights_list = []
newton_final_weights_list = []

for initial_condition in np.arange(1, 101):

  TARGET_MULTIPLE = 2
  W0 = rng.uniform(-4, 4, size=2) + 1e-3
  W_grad = W0
  W_newton = W0

  for epoch in np.arange(1, 100):

    print(f'Initial condition {initial_condition}')
    print('Epoch {0}'.format(epoch))

    input = jnp.array(rng.poisson(4, size=1) + 1e-3)

    def two_layer_network_loss(W, x, target_multiple):
      
      return jnp.mean((W[0] * W[1] * x - target_multiple * x)**2)

    grad_network = grad(two_layer_network_loss, argnums=(0))
    gradient1 = grad_network(W_grad, input, TARGET_MULTIPLE)

    def hessian(f):
        return jacfwd(jacrev(f))
    
    loss = two_layer_network_loss(W_newton, input, TARGET_MULTIPLE)

    if loss > 0:

      gradient2 = grad_network(W_newton, input, TARGET_MULTIPLE)

      H = hessian(two_layer_network_loss)(W_newton, input, TARGET_MULTIPLE)
      H_inv = jnp.linalg.inv(H)
      print(f'H: {H}')
      print(f'H_inv: ')
      print(f'loss: {loss}\n \n')

      newton_update_vector = jnp.linalg.matmul(H_inv, gradient2)

      W_newton = W_newton - newton_update_vector

    W_grad = W_grad - 1e-3 * gradient1

  init_weights_list.append(W0)
  grad_final_weights_list.append(W_grad)
  newton_final_weights_list.append(W_newton)

init_weights_arr = np.array(init_weights_list)
grad_final_weights_arr = np.array(grad_final_weights_list)
newton_final_weights_arr = np.array(newton_final_weights_list)

fig, ax = plt.subplots()

ax.scatter(init_weights_arr[:, 0], init_weights_arr[:, 1], color='tab:blue', label='initial weights')
ax.scatter(grad_final_weights_arr[:, 0], grad_final_weights_arr[:, 1], color='tab:orange', label='final weights under GD')
ax.scatter(newton_final_weights_arr[:, 0], newton_final_weights_arr[:, 1], color='tab:purple', label='final weights under NM')

target_curve_x_left = np.linspace(-4, 0, 500)
target_curve_x_right = np.linspace(0, 4, 500)

target_curve_y_left = TARGET_MULTIPLE / target_curve_x_left
target_curve_y_right = TARGET_MULTIPLE / target_curve_x_right

ax.plot(target_curve_x_left, target_curve_y_left, color='tab:green', label='loss = 0 curve')
ax.plot(target_curve_x_right, target_curve_y_right, color='tab:green')

for i in np.arange(init_weights_arr.shape[0]):

  ax.plot([init_weights_arr[i, 0], grad_final_weights_arr[i, 0]], [init_weights_arr[i, 1], grad_final_weights_arr[i, 1]], color='tab:orange')
  ax.plot([init_weights_arr[i, 0], newton_final_weights_arr[i, 0]], [init_weights_arr[i, 1], newton_final_weights_arr[i, 1]], color='tab:purple')


ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)

ax.set_title('Learned weights under different update rules')
ax.set_xlabel('weight 1 value')
ax.set_ylabel('weight 2 value')

ax.legend()

plt.show()

print('')