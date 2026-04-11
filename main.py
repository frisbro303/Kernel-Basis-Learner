import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RKHS Sheaf
    """)
    return


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    from jax import vmap, jit, random, tree_util
    import optax
    import matplotlib.pyplot as plt


    # --- 1. Model Definition ---
    def init_phi_params(rng, layers=[1, 64, 64, 1]):
        params = []
        for in_dim, out_dim in zip(layers[:-1], layers[1:]):
            k1, k2 = random.split(rng)
            params.append({
                'w': random.normal(k1, (in_dim, out_dim)) * jnp.sqrt(2/in_dim),
                'b': jnp.zeros((out_dim,))
            })
            rng = k2
        return params

    def phi_forward(params, x):
        # Standard MLP forward pass
        activation = x
        for layer in params[:-1]:
            activation = jax.nn.tanh(jnp.dot(activation, layer['w']) + layer['b'])
        return jnp.dot(activation, params[-1]['w']) + params[-1]['b']

    # Vectorization Strategy:
    # 1. Map a single model over a batch of X inputs
    batch_phi = vmap(phi_forward, in_axes=(None, 0))
    # 2. Map the entire collection of models over the same batch of X
    # in_axes=(0, None) -> map over the 'n_eigen' dimension of stacked params
    multi_phi = vmap(batch_phi, in_axes=(0, None))

    # --- 2. Loss Function ---
    def loss_fn(all_params, c, x_obs, y_obs, x_domain):
        # phis shape: (n_eigen, batch_size)
        phis_obs = multi_phi(all_params, x_obs).squeeze()
        y_pred = jnp.dot(c, phis_obs)
        data_loss = jnp.mean((y_obs - y_pred)**2)

        # Orthogonality on the global domain RR via sampling
        phis_dom = multi_phi(all_params, x_domain).squeeze()
        n_samples = x_domain.shape[0]
        # gram[i, j] = average(phi_i * phi_j)
        gram = jnp.dot(phis_dom, phis_dom.T) / n_samples
        ortho_loss = jnp.mean((gram - jnp.eye(n_eigen))**2)

        return data_loss + 2.0 * ortho_loss

    # --- 3. Initialization and Training ---
    n_eigen = 8
    key = random.PRNGKey(42)
    k_params, k_data, k_train = random.split(key, 3)

    # FIX: Stack independent parameters into a single PyTree of Arrays
    list_of_params = [init_phi_params(k) for k in random.split(k_params, n_eigen)]
    all_params = tree_util.tree_map(lambda *args: jnp.stack(args), *list_of_params)
    c = jnp.ones(n_eigen) / n_eigen

    # Synthetic Data
    x_obs = jnp.linspace(-3, 3, 50).reshape(-1, 1)
    y_obs = (jnp.sin(x_obs) + 0.5 * jnp.cos(2.5 * x_obs)).flatten()

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init((all_params, c))

    @jit
    def train_step(all_params, c, opt_state, x_obs, y_obs, x_domain):
        loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(all_params, c, x_obs, y_obs, x_domain)
        updates, opt_state = optimizer.update(grads, opt_state)
        all_params, c = optax.apply_updates((all_params, c), updates)
        return all_params, c, opt_state, loss

    for i in range(3001):
        k_train, subkey = random.split(k_train)
        # Global sampling across RR to learn the continuous basis
        x_domain = random.normal(subkey, (400, 1)) * 5.0
    
        all_params, c, opt_state, loss = train_step(all_params, c, opt_state, x_obs, y_obs, x_domain)
        if i % 1000 == 0: print(f"Step {i}, Loss: {loss:.6f}")

    # --- 4. Plotting Results ---
    x_test = jnp.linspace(-10, 10, 600).reshape(-1, 1)
    phis_test = multi_phi(all_params, x_test).squeeze()
    y_final = jnp.dot(c, phis_test)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    for k in range(n_eigen):
        plt.plot(x_test, phis_test[k], label=f'$\phi_{k}(x)$')
    plt.axvspan(-3, 3, color='gray', alpha=0.1, label='Data Region')
    plt.title("Independent Eigenfunctions over RR")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(x_obs, y_obs, color='black', s=10, label='Data')
    plt.plot(x_test, y_final, color='red', lw=2, label='Global Reconstruction')
    plt.title("Reconstruction in the Learned RKHS")
    plt.legend()
    plt.show()

    return


if __name__ == "__main__":
    app.run()
