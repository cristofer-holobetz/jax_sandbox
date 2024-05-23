import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
def loss_l2(y_hat, y, ws=None):
    return 0.5 * jnp.linalg.norm(y_hat - y, ord=2)**2
def create_trainer(ws, lr, loss, jit=True):
    def forward(ws, x, y, loss):
        def network(ws, x):
            for i in range(len(ws)):
                x = ws[f"w{i + 1}"] @ x
            return x
        return loss(network(ws, x), y, ws)
    def train(ws, x, y, v_and_g):
        def sgd(w, grad):
            return w - lr * grad
        loss, grads = v_and_g(ws, x, y)
        ws = jax.tree_util.tree_map(sgd, ws, grads)
        return loss, ws
    def training_step(ws, xs, ys, train):
        loss, ws = train(ws, xs, ys)
        return ws, loss
    forward = jax.tree_util.Partial(forward, loss=loss)
    v_and_g = jax.value_and_grad(forward)
    train = jax.tree_util.Partial(train, v_and_g=v_and_g)
    training_step = jax.tree_util.Partial(training_step, train=train)
    ws_map = {k: 0 for k in ws.keys()}
    training_step = jax.vmap(training_step, (ws_map, None, None))
    return (ws, jax.jit(training_step)) if jit else (ws, training_step)
root_seed = 1
np.random.seed(root_seed)
in_dim = 3
hidden_dim = 5
out_dim = 12
items_n = 5
lr = 0.03
steps = 700
inits = 5
stds = jnp.linspace(0.01, 0.5, 25)
ws = {
    "w1": jnp.concatenate([np.random.normal(0., std, (inits, hidden_dim, in_dim)) for std in stds], axis=0),
    "w2": jnp.concatenate([np.random.normal(0., std, (inits, out_dim, hidden_dim)) for std in stds], axis=0)
}
xs =  np.random.normal(0., 1., (in_dim, items_n))
ys =  np.random.normal(0., 1., (out_dim, items_n))
losses = []
ws, trainer = create_trainer(ws, lr, loss_l2)
for step in range(steps):
    i =  np.random.randint(items_n)
    ws, loss = trainer(ws, xs[:, [i]], ys[:, [i]])
    losses.append(loss)
# Loss for 700 Training steps for 5 initialisations across 25 initial stds
print(np.asarray(losses).shape)
print()