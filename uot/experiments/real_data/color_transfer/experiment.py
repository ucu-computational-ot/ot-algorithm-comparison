import os
import datetime
import pandas as pd
from PIL import Image
import jax
import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt

from uot.experiments.experiment import Experiment
from uot.utils.logging import logger
from uot.experiments.real_data.color_transfer.color_transfer_problem import ColorTransferProblem
from uot.experiments.real_data.color_transfer.measurement import measure_color_transfer_metrics


from uot.data.dataset_loader import load_matrix_as_color_grid
from uot.solvers.back_and_forth.pushforward import _forward_pushforward_nd
from mpl_toolkits.mplot3d import Axes3D  # For 3D plots


class ColorTransferExperiment(Experiment):
    """Color transfer experiment configuration"""

    def __init__(
            self,
            name: str,
            # solve_fn: callable[ColorTransferProblem, BaseSolver, list[BaseMeasure], list[ArrayLike], dict],
            output_dir: str = "output/color_transfer",
            drop_columns: list[str] = [],
    ):
        super().__init__(name=name, solve_fn=measure_color_transfer_metrics)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.drop_columns = drop_columns
        logger.debug(f"Initialized ColorTransferExperiment with output directory: {self.output_dir}")

    def run_on_problems(self, problems, solver, progress_callback=None, use_cost_matrix: bool = True, **solver_kwargs) -> pd.DataFrame:
        results = pd.DataFrame()
        for prob in problems:
            marginals = prob.get_marginals()

            # NOTE: for debugging I am saving the measures
            # Function to plot and save 2D projections for a given measure
            def plot_2d_projections(axs, mu_nd, title_prefix, file_prefix):
                # Convert to NumPy
                mu_np = np.array(mu_nd)
                axs_np = [np.array(ax) for ax in axs]
                
                # Compute 2D projections (sum over one axis)
                proj_rg = np.sum(mu_np, axis=2)  # RG: sum over B
                proj_rb = np.sum(mu_np, axis=1)  # RB: sum over G
                proj_gb = np.sum(mu_np, axis=0)  # GB: sum over R
                
                # Create figure with 3 subplots
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # RG projection
                axes[0].imshow(proj_rg.T, origin='lower', extent=[axs_np[0][0], axs_np[0][-1], axs_np[1][0], axs_np[1][-1]],
                            aspect='equal', cmap='viridis')
                axes[0].set_title(f'{title_prefix} RG Projection')
                axes[0].set_xlabel('Red')
                axes[0].set_ylabel('Green')
                axes[0].grid(False)
                
                # RB projection
                axes[1].imshow(proj_rb.T, origin='lower', extent=[axs_np[0][0], axs_np[0][-1], axs_np[2][0], axs_np[2][-1]],
                            aspect='equal', cmap='viridis')
                axes[1].set_title(f'{title_prefix} RB Projection')
                axes[1].set_xlabel('Red')
                axes[1].set_ylabel('Blue')
                axes[1].grid(False)
                
                # GB projection
                axes[2].imshow(proj_gb.T, origin='lower', extent=[axs_np[1][0], axs_np[1][-1], axs_np[2][0], axs_np[2][-1]],
                            aspect='equal', cmap='viridis')
                axes[2].set_title(f'{title_prefix} GB Projection')
                axes[2].set_xlabel('Green')
                axes[2].set_ylabel('Blue')
                axes[2].grid(False)
                
                # Add colorbar for density
                fig.colorbar(axes[0].images[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.05, label='Density')
                
                # Save to file
                fig.savefig(f'{file_prefix}_2d_projections.png', dpi=300, bbox_inches='tight')
                plt.close(fig)

            # NOTE: for debugging, saving the initial measures as 2D projections
            axs_source, mu_source_nd = marginals[0].for_grid_solver(backend="jax", dtype=jnp.float64)
            print(f"{mu_source_nd.sum()=}")
            plot_2d_projections(axs_source, mu_source_nd, title_prefix='Source', file_prefix='source')

            if len(marginals) > 1:
                axs_target, mu_target_nd = marginals[1].for_grid_solver(backend="jax", dtype=jnp.float64)
                print(f"{mu_target_nd.sum()=}")
                plot_2d_projections(axs_target, mu_target_nd, title_prefix='Target', file_prefix='target')

            costs = prob.get_costs() if use_cost_matrix else []
            logger.info(f"Running experiment {solver.__name__} with {solver_kwargs} on problem: {prob}")

            metrics = self.solve_fn(prob, solver, marginals, costs, **solver_kwargs)
            metrics["status"] = "success"
            logger.info(f"Successfully finished {solver.__name__} with {solver_kwargs}")

            # try:
            #     metrics = self.solve_fn(prob, solver, marginals, costs, **solver_kwargs)
            #     metrics["status"] = "success"
            #     logger.info(f"Successfully finished {solver.__name__} with {solver_kwargs}")
            # except Exception as e:
            #     logger.error(f"Error processing problem {prob.name}: {e}")
            #     print(e)
            #     metrics = {
            #         "status": "failed",
            #         "exception": str(e),
            #     }

            metrics.update(prob.to_dict())
            logger.debug(f"Metrics keys for problem {prob.name}: {metrics.keys()}")


            if solver.__name__ == 'BackNForthSqEuclideanSolver':
                jax.debug.print("monge_map shape {}", metrics['monge_map'].shape)
                plot_monge_map_3d(metrics['monge_map'], axs=axs_source)  # Defaults to mid-blue slice (dim=2)
                # Call multiple times for other slices, e.g.:
                plot_monge_map_3d(metrics['monge_map'], axs=axs_source, slice_dim=2, slice_idx=0, file_prefix='monge_map_lowb')  # Low blue
                # plot_monge_map_3d(metrics['monge_map'], axs=axs_source, slice_dim=0, slice_idx=32, file_prefix='monge_map_midr')  # Mid red, adjusts labels accordingly

                # Compute and save pushforward measures as 2D projections
                # Assuming axs_source and axs_target are still in scope; if not, recompute from marginals[0] and marginals[1]
                rho_mu_nd, _ = _forward_pushforward_nd(mu_source_nd, -metrics.get('v_final'))
                print(f"{type(rho_mu_nd)=}")
                print(f"{rho_mu_nd.shape=}")
                plot_2d_projections(axs_target, rho_mu_nd, title_prefix='Pushforward Source', file_prefix='pushforward_source')

                # save also the resulted rgb histogram as a measure
                data = metrics['transported_image'].copy()
                jax.debug.print("transported image shape {}", data.shape)
                jax.debug.print("transported image bounds max/min {}/{}", data.max(), data.min())
                if data.ndim == 2:
                    data = data[:, :, None]

                num_channels = data.shape[2]
                pixels = data.reshape(-1, num_channels).astype(jnp.float64)
                # pixels = data.reshape(-1, num_channels).astype(jnp.float64) / 255.0

                transported_grid_measure_nd = load_matrix_as_color_grid(
                    pixels=pixels,
                    num_channels=num_channels,
                    # bins_per_channel=6,
                    bins_per_channel=64,
                    # use_jax=True,
                ).for_grid_solver(backend="jax", dtype=jnp.float64)[1]
                plot_2d_projections(axs_source, transported_grid_measure_nd, title_prefix='Measure from Transported Image', file_prefix='transported_measure')

                # compute pushforward of nu
                rho_nu_nd, _ = _forward_pushforward_nd(mu_target_nd, -metrics.get('u_final'))
                print(f"{type(rho_nu_nd)=}")
                print(f"{rho_nu_nd.shape=}")
                plot_2d_projections(axs_source, rho_nu_nd, title_prefix='Pushforward Target', file_prefix='pushforward_target')


            if 'transported_image' in metrics:
                self._save_image(metrics['transported_image'], prob, solver, solver_kwargs)
                metrics.pop('transported_image', None)

            metrics.pop('source_image', None)
            metrics.pop('target_image', None)

            results = pd.concat([
                results,
                pd.DataFrame([metrics])
                ], ignore_index=True)

            if self.drop_columns:
                results = results.drop(columns=self.drop_columns, errors='ignore')

            if progress_callback:
                progress_callback(1)
        return results


    def _save_image(self, image, prob, solver, params):
        """Save transported image with metadata using PIL"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        solver_name = solver.__name__
        param_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))

        filename = f"{prob.name}_{solver_name}_{param_str}_{timestamp}.png"
        save_path = os.path.join(self.output_dir, "images")
        os.makedirs(save_path, exist_ok=True)

        try:
            if isinstance(image, jax.Array):
                image = np.asarray(image)

            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

            pil_image = Image.fromarray(image)
            pil_image.save(os.path.join(save_path, filename))

            logger.info(f"Saved transported image to {filename}")

        except Exception as e:
            logger.error(f"Error saving image {filename}: {e}")


def plot_monge_map_3d(monge_map, axs, slice_dim=2, slice_idx=None, downsample_3d=8, title_prefix='Monge Map', file_prefix='monge_map'):
    monge_np = np.array(monge_map)
    
    if axs is None:
        axs = [np.arange(64), np.arange(64), np.arange(64)]  # Fallback to indices
    
    # Default to mid-slice if not specified
    if slice_idx is None:
        slice_idx = monge_np.shape[slice_dim] // 2
    
    # Extract 2D slice: e.g., if slice_dim=2 (blue), slice[:,:,slice_idx,:]
    slice_map = np.take(monge_np, slice_idx, axis=slice_dim)  # Shape (64,64,3) for dim=2
    
    # Plot 2D slice components, RGB, and quiver (adapted from previous)
    # Create figure for components
    fig_comp, axes_comp = plt.subplots(1, 3, figsize=(18, 6))
    labels = ['Red', 'Green', 'Blue']
    for i, ax in enumerate(axes_comp):
        im = ax.imshow(slice_map[..., i], origin='lower', extent=[axs[0][0], axs[0][-1], axs[1][0], axs[1][-1]],
                       aspect='equal', cmap='viridis')
        ax.set_title(f'{title_prefix} Slice (fixed {labels[slice_dim]}={axs[slice_dim][slice_idx]:.2f}) - {labels[i]} Component')
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.grid(False)
    fig_comp.colorbar(im, ax=axes_comp, orientation='vertical', fraction=0.02, pad=0.05, label='Mapped Value')
    fig_comp.savefig(f'{file_prefix}_slice{slice_dim}_{slice_idx}_components.png', dpi=300, bbox_inches='tight')
    plt.close(fig_comp)
    
    # RGB image view of slice (if values in [0,1])
    if np.all((slice_map >= 0) & (slice_map <= 1)):
        fig_rgb, ax_rgb = plt.subplots(1, 1, figsize=(10, 10))
        ax_rgb.imshow(np.clip(slice_map, 0, 1))  # Clip for safety
        ax_rgb.set_title(f'{title_prefix} Slice as RGB Image (fixed {labels[slice_dim]}={axs[slice_dim][slice_idx]:.2f})')
        ax_rgb.axis('off')
        fig_rgb.savefig(f'{file_prefix}_slice{slice_dim}_{slice_idx}_rgb.png', dpi=300, bbox_inches='tight')
        plt.close(fig_rgb)
    
    # 2D quiver for displacement in slice (RG plane, assuming slice_dim=2)
    # downsample_2d = 4
    # r, g = np.meshgrid(axs[0][::downsample_2d], axs[1][::downsample_2d], indexing='ij')
    # mapped_r = slice_map[::downsample_2d, ::downsample_2d, 0]
    # mapped_g = slice_map[::downsample_2d, ::downsample_2d, 1]
    # dr = mapped_r - r
    # dg = mapped_g - g
    # fig_quiv2d, ax_quiv2d = plt.subplots(1, 1, figsize=(10, 10))
    # ax_quiv2d.quiver(r, g, dr, dg, scale=1, scale_units='xy', angles='xy')
    # ax_quiv2d.set_title(f'{title_prefix} 2D Displacement (RG plane, fixed {labels[slice_dim]}={axs[slice_dim][slice_idx]:.2f})')
    # ax_quiv2d.set_xlabel(labels[0])
    # ax_quiv2d.set_ylabel(labels[1])
    # ax_quiv2d.set_xlim([axs[0][0], axs[0][-1]])
    # ax_quiv2d.set_ylim([axs[1][0], axs[1][-1]])
    # fig_quiv2d.savefig(f'{file_prefix}_slice{slice_dim}_{slice_idx}_quiver2d.png', dpi=300, bbox_inches='tight')
    # plt.close(fig_quiv2d)
    
    # 3D quiver: Downsample full 3D grid
    # step = downsample_3d
    # r3, g3, b3 = np.meshgrid(axs[0][::step], axs[1][::step], axs[2][::step], indexing='ij')
    # mapped_r3 = monge_np[::step, ::step, ::step, 0]
    # mapped_g3 = monge_np[::step, ::step, ::step, 1]
    # mapped_b3 = monge_np[::step, ::step, ::step, 2]
    # dr3 = mapped_r3 - r3
    # dg3 = mapped_g3 - g3
    # db3 = mapped_b3 - b3
    
    # fig_quiv3d = plt.figure(figsize=(10, 10))
    # ax_quiv3d = fig_quiv3d.add_subplot(111, projection='3d')
    # ax_quiv3d.quiver(r3, g3, b3, dr3, dg3, db3, length=0.1, normalize=True)  # Adjust length for visibility
    # ax_quiv3d.set_title(f'{title_prefix} 3D Displacement (downsampled {step}x)')
    # ax_quiv3d.set_xlabel(labels[0])
    # ax_quiv3d.set_ylabel(labels[1])
    # ax_quiv3d.set_zlabel(labels[2])
    # fig_quiv3d.savefig(f'{file_prefix}_quiver3d.png', dpi=300, bbox_inches='tight')
    # plt.close(fig_quiv3d)