import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Train
    """)
    return


@app.cell
def _():
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax import jit, random, value_and_grad
    import optax
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_olivetti_faces

    return (
        fetch_olivetti_faces,
        jax,
        jit,
        jnp,
        np,
        optax,
        plt,
        random,
        value_and_grad,
    )


@app.cell
def _(
    fetch_olivetti_faces,
    jax,
    jit,
    jnp,
    np,
    optax,
    plt,
    random,
    value_and_grad,
):


    # --- 1. Parameters ---
    N_BASIS = 256        # Number of basis functions
    M_ANCHORS = 512      # Shared landmarks
    BATCH_SIZE = 400     # Using full batch for smoother convergence
    STEPS = 15001
    LEARNING_RATE = 5e-3 

    # --- 2. Data Preparation ---
    faces_data = fetch_olivetti_faces()
    Y_obs = jnp.array(faces_data.images).reshape(400, -1)
    Y_obs = (Y_obs - Y_obs.min()) / (Y_obs.max() - Y_obs.min())

    x_range = jnp.linspace(-1, 1, 64)
    coords = jnp.stack(jnp.meshgrid(x_range, x_range, indexing='ij'), axis=-1).reshape(-1, 2)

    @jit
    def compute_basis(coords, anchors, mixing_weights, L_params):
        """
        Anisotropic Basis: exp(-0.5 * (x-z).T @ (L @ L.T) @ (x-z))
        L_params: (M, 3) representing [l11, l21, l22] of lower-triangular L
        """
        l11, l21, l22 = L_params[:, 0], L_params[:, 1], L_params[:, 2]

        diff = coords[:, None, :] - anchors[None, :, :]  # (P, M, 2)
        dx, dy = diff[..., 0], diff[..., 1]

        # Transform coordinates by L^T
        v1 = l11[None, :] * dx + l21[None, :] * dy
        v2 = l22[None, :] * dy
        dist_sq = v1**2 + v2**2

        # Gaussian Kernel + Linear Mixing
        K_pixels_anchors = jnp.exp(-0.5 * dist_sq)
        phis = K_pixels_anchors @ mixing_weights

        # Unit norm normalization
        return phis / (jnp.linalg.norm(phis, axis=0, keepdims=True) + 1e-6)

    @jit
    def fast_solve(Phi, Y_batch, reg=1e-5):
        n_basis = Phi.shape[1]
        lhs = Phi.T @ Phi + reg * jnp.eye(n_basis)
        rhs = Phi.T @ Y_batch.T
        L = jnp.linalg.cholesky(lhs)
        m_n = jax.scipy.linalg.cho_solve((L, True), rhs).T
        return m_n

    def init_params(rng):
        k1, k2, k3, k4 = random.split(rng, 4)
        l11_init = random.uniform(k3, (M_ANCHORS, 1), minval=15.0, maxval=25.0)
        l22_init = random.uniform(k4, (M_ANCHORS, 1), minval=15.0, maxval=25.0)
        l21_init = jnp.zeros((M_ANCHORS, 1)) 

        init_l = jnp.concatenate([l11_init, l21_init, l22_init], axis=1)

        return {
            'anchors': random.uniform(k1, (M_ANCHORS, 2), minval=-0.8, maxval=0.8),
            'mixing_weights': random.normal(k2, (M_ANCHORS, N_BASIS)) * 0.1,
            'L_params': init_l
        }


    @jit
    def total_variation(y_img):
        """Calculates the Total Variation of a 2D image."""
        diff_h = jnp.abs(y_img[1:, :] - y_img[:-1, :])
        diff_v = jnp.abs(y_img[:, 1:] - y_img[:, :-1])
        return jnp.mean(diff_h) + jnp.mean(diff_v)

    @jit
    def train_step(params, opt_state, y_batch, coords):
        def loss_fn(p):
            phis = compute_basis(coords, p['anchors'], p['mixing_weights'], p['L_params'])
            m_n = fast_solve(phis, y_batch)
            y_pred = m_n @ phis.T

            # L2 Reconstruction Loss
            recon_loss = jnp.mean((y_batch - y_pred)**2)

            # Total Variation Loss (reshaped to image dimensions)
            y_pred_imgs = y_pred.reshape(-1, 64, 64)
            tv_loss = jnp.mean(jax.vmap(total_variation)(y_pred_imgs))

            # Orthogonality constraint
            ortho_loss = jnp.mean((phis.T @ phis - jnp.eye(N_BASIS))**2)

            # Weighted Total Loss
            # lambda_tv usually ranges from 1e-4 to 1e-2
            return recon_loss + 0.01 * ortho_loss + 1e-2 * tv_loss

        loss, grads = value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss


    # --- 4. Execution ---
    rng = random.PRNGKey(42)
    params = init_params(rng)

    # 1. Define your separate optimizers
    # We give L_params a 10x higher learning rate to encourage stretching
    optimizers = {
        'fast': optax.adam(LEARNING_RATE * 5),
        'standard': optax.adam(LEARNING_RATE)
    }

    # 2. Create a "mask" that matches your params structure
    # This tells Optax which optimizer to use for which key
    param_labels = {
        'anchors': 'standard',
        'mixing_weights': 'standard',
        'L_params': 'fast'  # Use the high learning rate here
    }

    # 3. Combine them
    optimizer = optax.multi_transform(optimizers, param_labels)

    opt_state = optimizer.init(params)


    print("Training Model...")
    for i in range(STEPS):
        rng, subk = random.split(rng)
        idx = random.choice(subk, 400, shape=(BATCH_SIZE,))
        params, opt_state, loss = train_step(params, opt_state, Y_obs[idx], coords)

        if i % 500 == 0:
            print(f"Step {i:4d} | Loss: {loss:.6f}")

    np.savez(
        "model.npz", 
        anchors=np.array(params['anchors']),
        mixing_weights=np.array(params['mixing_weights']),
        L_params=np.array(params['L_params'])
    )
    print(f"--- Training complete. Model saved to model.npz ---")

    # Final inference
    phis = compute_basis(coords, params['anchors'], params['mixing_weights'], params['L_params'])
    m_all = fast_solve(phis, Y_obs)
    Y_reconstructed = m_all @ phis.T

    # --- 5. Visualization ---
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(phis[:, i].reshape(64, 64), cmap='RdBu_r')
        axes[i].set_title(f"Basis {i}")
        axes[i].axis('off')
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    indices = [25, 125, 225]
    for i, idx in enumerate(indices):
        axes[0, i].imshow(Y_obs[idx].reshape(64, 64), cmap='gray')
        axes[1, i].imshow(Y_reconstructed[idx].reshape(64, 64), cmap='gray')
        axes[0, i].set_title("Original")
        axes[1, i].set_title("Reconstructed")
        for ax in axes[:, i]: ax.axis('off')
    plt.tight_layout()
    plt.show()

    import matplotlib.patches as patches

    # --- 5. Visualization with Ellipses ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    indices = [25, 125, 225]

    # Extract geometry for the landmarks
    anc = params['anchors']
    l11, l21, l22 = params['L_params'][:, 0], params['L_params'][:, 1], params['L_params'][:, 2]

    for i, idx in enumerate(indices):
        # Plot Original + Landmarks
        axes[0, i].imshow(Y_obs[idx].reshape(64, 64), cmap='gray', extent=[-1, 1, 1, -1])

        # Draw each anisotropic landmark as an ellipse
        for j in range(M_ANCHORS):
            # The precision matrix is P = L @ L.T
            # To get the ellipse shape, we look at the Covariance Sigma = P^-1
            L = jnp.array([[l11[j], 0], [l21[j], l22[j]]])
            P = L @ L.T
            Sigma = jnp.linalg.inv(P)

            # Calculate eigenvalues for width/height and eigenvectors for rotation
            vals, vecs = jnp.linalg.eigh(Sigma)
            angle = jnp.degrees(jnp.arctan2(vecs[1, 1], vecs[0, 1]))

            # We use 2*sqrt(val) to represent ~2 standard deviations
            width, height = 2 * jnp.sqrt(vals[1]), 2 * jnp.sqrt(vals[0])

            ellipse = patches.Ellipse(
                xy=(anc[j, 1], anc[j, 0]), 
                width=width, height=height, angle=angle,
                edgecolor='red', facecolor='none', alpha=0.3, lw=0.5
            )
            axes[0, i].add_patch(ellipse)

        axes[0, i].set_title(f"Original {idx} + Landmarks")

        # Plot Reconstructed
        axes[1, i].imshow(Y_reconstructed[idx].reshape(64, 64), cmap='gray')
        axes[1, i].set_title("Reconstructed")

        for ax in axes[:, i]: 
            ax.axis('off')

    plt.tight_layout()
    plt.show()
    return Y_obs, compute_basis, m_all, params


@app.cell
def _(Y_obs, compute_basis, jnp, m_all, params, plt):
    # --- 6. Upscaling to 256x256 ---
    upscale_res = 512
    x_range_high = jnp.linspace(-1, 1, upscale_res)
    # Create a much denser grid
    coords_high = jnp.stack(jnp.meshgrid(x_range_high, x_range_high, indexing='ij'), axis=-1).reshape(-1, 2)



    phis_high = compute_basis(coords_high, params['anchors'], params['mixing_weights'], params['L_params'])


    # Use the weights (m_all) we already solved for the 64x64 version
    # This "projects" the low-res weights onto the high-res basis 
    idx_to_plot = 20 #Let's pick one face (e.g., the person with gxlasses)
    Y_upscaled = (m_all[idx_to_plot] @ phis_high.T).reshape(upscale_res, upscale_res)

    # Compare Original (64x64) vs Upscaled (256x256)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 6))
    _axes[0].imshow(Y_obs[idx_to_plot].reshape(64, 64), cmap='gray', interpolation='nearest')
    _axes[0].set_title("Original (64x64 Pixels)")
    _axes[1].imshow(Y_upscaled, cmap='gray')
    _axes[1].set_title(f"Neural Upscale ({upscale_res}x{upscale_res})")
    for _ax in _axes: _ax.axis('off')
    plt.show()
    return


@app.cell
def _(mo):
    t_slider = mo.ui.slider(start=0, stop=1, step=0.01, value=0.5, label="Morphing Factor")
    t_slider
    return (t_slider,)


@app.cell(hide_code=True)
def _(h, jnp, m_n_all, phis_final, plt, t_slider, w):
    # 2. In a new cell, perform the interpolation and plot
    # Let's morph between Face 0 and Face 1
    person_a = m_n_all[45]
    person_b = m_n_all[100]

    # Interpolate the coefficients
    mixed_coeffs = (1 - t_slider.value) * person_a + t_slider.value * person_b

    # Reconstruct using the shared basis
    morphed_face = jnp.dot(mixed_coeffs, phis_final.T).reshape(h, w)

    plt.imshow(morphed_face, cmap='gray')
    plt.axis('off')
    plt.title(f"Morphed Face (t={t_slider.value})")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
