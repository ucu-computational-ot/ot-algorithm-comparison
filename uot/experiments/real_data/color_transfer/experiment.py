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
            costs = prob.get_costs() if use_cost_matrix else []
            logger.info(f"Running experiment {solver.__name__} with {solver_kwargs} on problem: {prob}")
            try:
                metrics = self.solve_fn(prob, solver, marginals, costs, **solver_kwargs)
                metrics["status"] = "success"
                logger.info(f"Successfully finished {solver.__name__} with {solver_kwargs}")
            except Exception as e:
                logger.error(f"Error processing problem {prob.name}: {e}")
                print(e)
                metrics = {
                    "status": "failed",
                    "exception": str(e),
                }

            metrics.update(prob.to_dict())
            logger.debug(f"Metrics keys for problem {prob.name}: {metrics.keys()}")

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
