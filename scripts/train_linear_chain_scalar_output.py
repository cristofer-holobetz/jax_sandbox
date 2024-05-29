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

def distribute_inverse_hessian(inverse_hessian, treedef):
    """
    Take in an inverse hessian in matrix form and a tree structure definition. Unpack
    the inverse hessian so it's stored with the same structure as the treedef 

    """


# loss is mean squared error given predictions and labels
def loss_l2(y_hat, y, ws=None):
    
    return 0.5 * jnp.linalg.norm(y_hat - y, ord=2)**2

# This function creates a trainer given a particular set of weights,
# a specific loss function and a learning rate
# The trainer 
def create_trainer(ws, lr, loss, jit=True):
    
    def forward(ws, x, y, loss):
        
        def network(ws, x):
            
            for i in range(len(ws)):
                
                x = ws[f"w{i + 1}"] @ x

            return x
        
        return loss(network(ws, x), y, ws)
    
    def train(ws, x, y, v_and_g, compute_hessian=None, optimizer_type='sgd'):
        
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
            hessian = collect_hessian(pytree_hessian)
            inverse_hessian = jnp.linalg.inv(hessian)

            gradient = collect_gradient(ws)

            ws_treedef = jax.tree_util.tree_structure(ws)

            update_vector = jnp.linalg.matmul(inverse_hessian, gradient)
            pytree_update_vector = jax.tree_util.tree_unflatten(ws_treedef, update_vector)

            ws1 = jax.tree_util.tree_map(nm, ws, pytree_update_vector)

        return loss, ws1
    
    def training_step(ws, xs, ys, train):
        
        loss, ws = train(ws, xs, ys)

        return ws, loss

    forward = jax.tree_util.Partial(forward, loss=loss)
    v_and_g = jax.value_and_grad(forward)
    compute_hessian = jax.hessian(forward)
    
    train = jax.tree_util.Partial(train, v_and_g=v_and_g, compute_hessian=compute_hessian, optimizer_type='nm')
    training_step = jax.tree_util.Partial(training_step, train=train)

    ws_map = {k: 0 for k in ws.keys()}
    training_step = jax.vmap(training_step, (ws_map, None, None))
    return (ws, jax.jit(training_step)) if jit else (ws, training_step)

root_seed = 1
np.random.seed(root_seed)
in_dim = 1
hidden_dim = 1
out_dim = 1
items_n = 5
lr = 0.03
steps = 700
inits = 5
std = 0.2

#ws = {
#    "w1": jnp.concatenate([np.random.normal(0., std, (inits, hidden_dim, in_dim)) for std in stds], axis=0),
#    "w2": jnp.concatenate([np.random.normal(0., std, (inits, out_dim, hidden_dim)) for std in stds], axis=0)
#}

ws = {
    "w1": np.random.uniform(-3, 3, (hidden_dim, in_dim)),
    "w2": np.random.uniform(-3, 3, (hidden_dim, in_dim))
}

xs =  np.random.normal(0., 1., (in_dim, items_n))
ys =  np.random.normal(0., 1., (out_dim, items_n))

losses = []
weights = []
ws, trainer = create_trainer(ws, lr, loss_l2, jit=False)
for step in range(steps):
    
    i =  np.random.randint(items_n)
    ws, loss = trainer(ws, xs[:, [i]], ys[:, [i]])
    losses.append(loss)

weights.append(ws)

# Loss for 700 Training steps for 5 initialisations across 25 initial stds
print(np.asarray(losses).shape)
print()

fig, ax = plt.subplots()

ax.scatter()