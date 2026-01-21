import os
import datetime
import hashlib
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
from uot.solvers.back_and_forth.forward_pushforward import cic_pushforward_nd


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
        self.soft_extension_modes: list[bool] | None = [False]
        self.displacement_alphas: list[float] = [1.0]
        logger.debug(f"Initialized ColorTransferExperiment with output directory: {self.output_dir}")

    def set_soft_extension_modes(self, modes: list[bool] | None):
        if modes is None:
            self.soft_extension_modes = None
            return
        if not modes:
            modes = [False]
        self.soft_extension_modes = [bool(m) for m in modes]

    def set_displacement_alphas(self, alphas: list[float]):
        if not alphas:
            alphas = [1.0]
        self.displacement_alphas = [float(a) for a in alphas]

    def run_on_problems(self, problems, solver, progress_callback=None, use_cost_matrix: bool = True, **solver_kwargs) -> pd.DataFrame:
        results = pd.DataFrame()
        problem_list = list(problems)
        total = len(problem_list)
        for idx, prob in enumerate(problem_list, start=1):
            marginals = prob.get_marginals()

            costs = prob.get_costs() if use_cost_matrix else []
            logger.info(
                "Running experiment %s (%d/%d) on problem: %s",
                solver.__name__,
                idx,
                total,
                prob.name,
            )

            metrics_entries = self.solve_fn(
                prob,
                solver,
                marginals,
                costs,
                soft_extension_modes=self.soft_extension_modes,
                displacement_alphas=self.displacement_alphas,
                **solver_kwargs,
            )
            if not isinstance(metrics_entries, list):
                metrics_entries = [metrics_entries]

            for metrics in metrics_entries:
                metrics["status"] = "success"
                logger.info(f"Successfully finished {solver.__name__} with {solver_kwargs}")

                prob_info = prob.to_dict()
                if 'cost' in prob_info:
                    prob_info['cost_fn'] = prob_info.pop('cost')
                metrics.update(prob_info)
                logger.debug(f"Metrics keys for problem {prob.name}: {metrics.keys()}")

                if 'transported_image' in metrics:
                    image_params = dict(solver_kwargs)
                    if 'soft_extension' in metrics:
                        image_params['soft_extension'] = 'yes' if metrics.get('soft_extension') else 'no'
                    image_params['displacement_alpha'] = f"{metrics.get('displacement_alpha', 1.0):.3f}"
                    image = metrics['transported_image']
                    if hasattr(prob, "to_rgb_image"):
                        image = prob.to_rgb_image(image)
                    filename = self._save_image(image, prob, solver, image_params)
                    metrics["result_image_filename"] = filename
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
            if hasattr(prob, "free_memory"):
                prob.free_memory()
        return results


    def _save_image(self, image, prob, solver, params):
        """Save transported image with metadata using PIL"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        solver_name = solver.__name__
        param_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
        filename = self._build_safe_filename(prob.name, solver_name, param_str, timestamp)
        save_path = os.path.join(self.output_dir, "images")
        os.makedirs(save_path, exist_ok=True)

        try:
            if isinstance(image, jax.Array):
                image = np.asarray(image)

            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

            pil_image = Image.fromarray(image)
            pil_image.save(os.path.join(save_path, filename))

            logger.info(f"Saved transported image to {filename}")
            return filename

        except Exception as e:
            logger.error(f"Error saving image {filename}: {e}")

    @staticmethod
    def _sanitize_component(text: str, max_length: int) -> str:
        safe = "".join(ch if ch.isalnum() else "_" for ch in text)
        return safe[:max_length] if len(safe) > max_length else safe

    def _build_safe_filename(self, problem_name, solver_name, param_str, timestamp) -> str:
        problem_component = self._sanitize_component(problem_name, 60)
        solver_component = self._sanitize_component(solver_name, 40)
        safe_params = self._sanitize_component(param_str, 120)
        if len(safe_params) < len(param_str):
            digest = hashlib.md5(param_str.encode("utf-8")).hexdigest()[:10]
            safe_params = f"params_{digest}"
        base = f"{problem_component}_{solver_component}_{safe_params}_{timestamp}.png"
        max_len = 200
        if len(base) <= max_len:
            return base
        digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:10]
        return f"{problem_component[:50]}_{solver_component[:30]}_{digest}_{timestamp}.png"
