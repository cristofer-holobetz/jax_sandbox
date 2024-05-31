import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt

TARGET_MULTIPLE = 2
root_seed = 1
np.random.seed(root_seed)
in_dim = 1
hidden_dim = 1
out_dim = 1
items_n = 5
lr = 0.03
steps = 1000
inits = 1000

ws_init = {
    "w1": np.random.uniform(-4, 4, (inits, hidden_dim, in_dim)),
    "w2": np.random.uniform(-4, 4, (inits, out_dim, hidden_dim))
}

def collect_hessian(pytree_hessian):

    # base case
    if isinstance(pytree_hessian, ArrayLike):

        return pytree_hessian
    
    # bottom recursive case - concatenate horizontally
    elif isinstance(pytree_hessian[list(pytree_hessian.keys())[0]], ArrayLike):
        
        collected_hessian = [collect_hessian(pytree_hessian[key]) for key in pytree_hessian.keys()]

        return jnp.concatenate(collected_hessian, axis=1)

    # intermediate recursive case - concatenate vertically
    else:

        collected_hessian = [collect_hessian(pytree_hessian[key]) for key in pytree_hessian.keys()]

        return jnp.concatenate(collected_hessian, axis=0)
    
def collect_gradient(pytree_gradient):

    # base case
    if isinstance(pytree_gradient, ArrayLike):

        return pytree_gradient
    
    # intermediate recursive case - concatenate vertically
    else:

        return jnp.concatenate([collect_hessian(pytree_gradient[key]) for key in pytree_gradient.keys()], axis=0)

def flatten_ws_leaves(ws):

    ws_treedef = jax.tree_util.tree_structure(ws)

    ws_leaves = jax.tree.leaves(ws)
    flattened_leaves = [jnp.ravel(leaf) for leaf in ws_leaves]

    ws_flattened = jax.tree_util.tree_unflatten(ws_treedef, flattened_leaves)

    return ws_flattened

def unflatten_ws_leaves(ws, dimensions=((hidden_dim, in_dim), (out_dim, hidden_dim))):

    ws_treedef = jax.tree_util.tree_structure(ws)

    ws_leaves = jax.tree.leaves(ws)
    flattened_leaves = [jnp.reshape(leaf, dimensions[i]) for i, leaf in enumerate(ws_leaves)]

    ws_flattened = jax.tree_util.tree_unflatten(ws_treedef, flattened_leaves)

    return ws_flattened

# loss is mean squared error given predictions and labels
def loss_l2(y_hat, y, ws=None):
    
    return 0.5 * jnp.linalg.norm(y_hat - y, ord=2)**2

# This function creates a trainer given a particular set of weights,
# a specific loss function and a learning rate
# The trainer 
def create_trainer(ws, lr, loss, jit=True):
    
    def forward(ws, x, y, loss):
        
        def network(ws, x):
            
            ws = unflatten_ws_leaves(ws, dimensions=((hidden_dim, in_dim), (out_dim, hidden_dim)))
            
            for i in range(len(ws)):
                
                x = ws[f"w{i + 1}"] @ x

            return x
        
        return loss(network(ws, x), y, ws)
    
    def train(ws, x, y, v_and_g, compute_hessian=None, optimizer_type='sgd'):
        
        ws = flatten_ws_leaves(ws)

        loss, grads = v_and_g(ws, x, y)
        
        if optimizer_type == 'sgd':

            def sgd(w, grad):
                
                return w - lr * grad
            
            ws = jax.tree_util.tree_map(sgd, ws, grads)

        else:

            def nm(w, update_vector):

                return w - update_vector

            
            # To appropriately send the correct hessian row x grad vector, we need
            # to make a tree_map with hessian rows (i think)

            # I want the hessian to be a pytree where each leaf is the hessian
            # wrt only the weights in that layer
            pytree_hessian = compute_hessian(ws, x, y)
            hessian = collect_hessian(pytree_hessian).squeeze()
            #inverse_hessian = jnp.linalg.inv(hessian)
            print('Loss')
            print(loss)


            #print(inverse_hessian)

            gradient = collect_gradient(grads)
            #ws = flatten_ws_leaves(ws)

            ws_treedef = jax.tree_util.tree_structure(ws)
            
            update_vector = jax.scipy.linalg.solve(hessian, gradient)

            # Only set to be nothing if loss is 0, otherwise we use the computed update_vector
            #update_vector = jnp.heaviside(loss, 0) * update_vector

            update_vector = jnp.nan_to_num(update_vector, nan=0, posinf=0, neginf=0)

            print('Solving for update_vector')
            print(update_vector)

            num_layers = len(list(ws.keys()))
            num_weights_per_layer = [0] + [np.prod(ws[key].shape) for key in ws.keys()]
            separated_update_vector = [update_vector[num_weights_per_layer[i]:num_weights_per_layer[i] + num_weights_per_layer[i+1]] for i in np.arange(num_layers)]
            pytree_update_vector = jax.tree_util.tree_unflatten(ws_treedef, separated_update_vector)

            #loss_tree = jax.tree_util.tree_unflatten(ws_treedef, [loss] * num_layers)

            ws = jax.tree_util.tree_map(nm, ws, pytree_update_vector)
            ws = unflatten_ws_leaves(ws, ((hidden_dim, in_dim), (out_dim, hidden_dim)))

        return loss, ws
    
    def training_step(ws, xs, ys, train):
        
        loss, ws = train(ws, xs, ys)

        return ws, loss

    forward = jax.tree_util.Partial(forward, loss=loss)
    v_and_g = jax.value_and_grad(forward)
    compute_hessian = jax.hessian(forward, argnums=0)
    
    train = jax.tree_util.Partial(train, v_and_g=v_and_g, compute_hessian=compute_hessian, optimizer_type='nm')
    training_step = jax.tree_util.Partial(training_step, train=train)

    ws_map = {k: 0 for k in ws.keys()}
    training_step = jax.vmap(training_step, (ws_map, None, None))
    return (ws, jax.jit(training_step)) if jit else (ws, training_step)


"""
root_seed = 1
np.random.seed(root_seed)
in_dim = 3
hidden_dim = 5
out_dim = 12
items_n = 5
lr = 0.03
steps = 700
inits = 5
stds = np.linspace(0.01, 0.5, 25)

ws = {
    "w1": jnp.concatenate([np.random.normal(0., std, (inits, hidden_dim, in_dim)) for std in stds], axis=0),
    "w2": jnp.concatenate([np.random.normal(0., std, (inits, out_dim, hidden_dim)) for std in stds], axis=0)
}
"""
xs = np.random.poisson(lam=1, size=(in_dim, items_n)) + 10
ys = TARGET_MULTIPLE * xs
#ys =  np.random.normal(0., 1., (out_dim, items_n))

losses = []
weights = []
ws, trainer = create_trainer(ws_init, lr, loss_l2, jit=True)
for step in range(steps):
    
    i =  np.random.randint(items_n)
    ws, loss = trainer(ws, xs[:, [i]], ys[:, [i]])
    losses.append(loss)

    weights.append(ws)

fig, ax = plt.subplots()

init_w1 = ws_init['w1'].squeeze()
init_w2 = ws_init['w2'].squeeze()

final_w1 = ws['w1'].squeeze()
final_w2 = ws['w2'].squeeze()

# Plot the initial weights
ax.scatter(init_w1, init_w2, label='initializations', color='tab:blue')

# Plot the final weights
ax.scatter(final_w1, final_w2, label='final weights', color='tab:orange')

target_curve_x_left = np.linspace(-4, 0, 500)[:-1]
target_curve_x_right = np.linspace(0, 4, 500)[1:]

target_curve_y_left = TARGET_MULTIPLE / target_curve_x_left
target_curve_y_right = TARGET_MULTIPLE / target_curve_x_right

ax.plot(target_curve_x_left, target_curve_y_left, color='tab:green', label='loss = 0 curve')
ax.plot(target_curve_x_right, target_curve_y_right, color='tab:green')

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)

# Plot the init - final weight pairs
for init_i in np.arange(inits):

    ax.plot([init_w1[init_i], final_w1[init_i]], [init_w2[init_i], final_w2[init_i]], color='tab:purple')

plt.show()
print()