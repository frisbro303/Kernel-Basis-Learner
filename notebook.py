# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.1",
#     "matplotlib==3.10.8",
#     "numpy==2.4.4",
# ]
# ///
import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches


   
    data_path = mo.notebook_location() / "public" / "model.npz"
    data = np.load(data_path)

    params = {
        'anchors': data['anchors'],
        'mixing_weights': data['mixing_weights'],
        'L_params': data['L_params']
    }

    def compute_basis_np(coords, anchors, mixing_weights, L_params):
        l11, l21, l22 = L_params[:, 0], L_params[:, 1], L_params[:, 2]
    
        diff = coords[:, np.newaxis, :] - anchors[np.newaxis, :, :] # (P, M, 2)
        dx, dy = diff[..., 0], diff[..., 1]

        v1 = l11[np.newaxis, :] * dx + l21[np.newaxis, :] * dy
        v2 = l22[np.newaxis, :] * dy
        dist_sq = v1**2 + v2**2

        phis = np.exp(-0.5 * dist_sq) @ mixing_weights
        return phis / (np.linalg.norm(phis, axis=0, keepdims=True) + 1e-6)

    # 3. Setup coordinates and generate Phis
    x_range = np.linspace(-1, 1, 64)
    coords = np.stack(np.meshgrid(x_range, x_range, indexing='ij'), axis=-1).reshape(-1, 2)
    phis = compute_basis_np(coords, params['anchors'], params['mixing_weights'], params['L_params'])

    # 4. Visualization: Basis Functions
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(phis[:, i].reshape(64, 64), cmap='RdBu_r')
        axes[i].set_title(f"Basis {i}")
        axes[i].axis('off')
    plt.show()

    # 5. Visualization: Landmarks/Ellipses
    # Assuming Y_obs and Y_reconstructed are available as numpy arrays
    def plot_landmarks(Y_obs, Y_reconstructed, params, indices=[25, 125, 225]):
        anc = params['anchors']
        l11, l21, l22 = params['L_params'].T
    
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for i, idx in enumerate(indices):
            # Top Row: Original + Landmark Ellipses
            axes[0, i].imshow(Y_obs[idx].reshape(64, 64), cmap='gray', extent=[-1, 1, 1, -1])
        
            for j in range(len(anc)):
                # Sigma = (L L^T)^-1
                L = np.array([[l11[j], 0], [l21[j], l22[j]]])
                Sigma = np.linalg.inv(L @ L.T)
                vals, vecs = np.linalg.eigh(Sigma)
            
                # Width/Height are 2 * std_dev; Angle from eigenvector
                angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
                w, h = 2 * np.sqrt(vals[1]), 2 * np.sqrt(vals[0])

                ellipse = patches.Ellipse(
                    xy=(anc[j, 1], anc[j, 0]), width=w, height=h, angle=angle,
                    edgecolor='red', facecolor='none', alpha=0.2, lw=0.5
                )
                axes[0, i].add_patch(ellipse)
        
            axes[1, i].imshow(Y_reconstructed[idx].reshape(64, 64), cmap='gray')
            axes[0, i].set_title(f"Original {idx} + Ellipses")
            axes[1, i].set_title("Reconstructed")
            for ax in axes[:, i]: ax.axis('off')
        plt.tight_layout()
        plt.show()


    return


if __name__ == "__main__":
    app.run()
