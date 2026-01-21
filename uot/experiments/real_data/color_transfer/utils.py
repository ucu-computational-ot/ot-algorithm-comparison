import os

import numpy as np
from jax import numpy as jnp
import cv2
import PIL
import pandas as pd
from pandas import DataFrame

from uot.experiments.real_data.color_transfer.image_data import ImageData
from uot.problems.two_marginal import TwoMarginalProblem
from uot.utils.costs import cost_euclid
from uot.experiments.real_data.color_transfer.color_transfer_problem import ColorTransferProblem

from uot.utils.logging import logger

def _coerce_bin_numbers(bin_value):
    if isinstance(bin_value, int):
        return [bin_value]
    if isinstance(bin_value, (list, tuple)):
        if not bin_value:
            raise ValueError("bin-number must not be empty")
        for val in bin_value:
            if not isinstance(val, int):
                raise TypeError("bin-number entries must be integers")
        return list(bin_value)
    raise TypeError("bin-number must be an integer or a list of integers")

def _coerce_soft_extension_list(option_value):
    def _to_bool(entry):
        if isinstance(entry, bool):
            return entry
        if isinstance(entry, str):
            lowered = entry.strip().lower()
            if lowered in {"yes", "true", "1"}:
                return True
            if lowered in {"no", "false", "0"}:
                return False
        raise TypeError("soft-extension entries must be booleans or yes/no strings")

    if option_value is None:
        return None
    if isinstance(option_value, (list, tuple)):
        if not option_value:
            raise ValueError("soft-extension list must not be empty")
        return [_to_bool(v) for v in option_value]
    return [_to_bool(option_value)]


def _coerce_displacement_alphas(option_value):
    def _to_float(entry):
        try:
            value = float(entry)
        except (TypeError, ValueError):
            raise TypeError("displacement-interpolation entries must be numeric") from None
        if not 0.0 <= value <= 1.0:
            raise ValueError("displacement-interpolation values must lie in [0, 1]")
        return value

    if option_value is None:
        return [1.0]
    if isinstance(option_value, (list, tuple)):
        if not option_value:
            raise ValueError("displacement-interpolation list must not be empty")
        return [_to_float(v) for v in option_value]
    return [_to_float(option_value)]

def load_config_info(config: dict)-> tuple:
    """Get miscellaneous info from the config"""
    bin_numbers = _coerce_bin_numbers(config['bin-number'])
    soft_extension = (
        _coerce_soft_extension_list(config['soft-extension'])
        if 'soft-extension' in config else None
    )
    displacement_alphas = _coerce_displacement_alphas(config.get('displacement-interpolation', [1.0]))
    return (
        bin_numbers,
        config['batch-size'],
        config['pair-number'],
        config['images-dir'],
        config['rng-seed'],
        config.get('drop-columns', []),
        soft_extension,
        displacement_alphas,
    )

def get_image_problems(
    data: dict[str, ImageData],
    image_pairs: set[tuple],
    bins_per_channel: int,
) -> list[TwoMarginalProblem]:
    """Get the image pairs as problems for the experiment"""
    return [
        ColorTransferProblem(
            name=f"{source_name} -> {target_name}",
            mu=data[source_name].get_grid(),
            nu=data[target_name].get_grid(),
            cost_fn=cost_euclid,
            source_image=data[source_name],
            target_image=data[target_name],
            source_image_name=source_name,
            target_image_name=target_name,
            bins_per_channel=bins_per_channel,
        )
        for source_name, target_name in image_pairs
    ]


def sample_image_pairs(images_dir: str, bin_num: int, pair_num: int, seed: int = 42) -> tuple[dict[str, ImageData], set[tuple[str, str]]]:

    rng = np.random.default_rng(seed)
    all_images = os.listdir(images_dir)

    if len(all_images) * (len(all_images) - 1) < pair_num:
        raise ValueError("Not enough images to sample the required number of pairs.")
    
    pairs = set()
    data = {}

    while len(pairs) < pair_num:
        logger.debug(f"[sampler] sampling {len(pairs)} up to {pair_num=}")

        source_name, target_name = rng.choice(all_images, size=2, replace=False)

        if (source_name, target_name) not in pairs:
            pairs.add((source_name, target_name))

            if source_name not in data:
                data[source_name] = ImageData(source_name, bin_num)

            if target_name not in data:
                data[target_name] = ImageData(target_name, bin_num)
    
    return data, pairs
    

def name_to_tuple(name: str) -> tuple:
    """Convert problem name to a tuple of source and target image names"""
    parts = name.split(" -> ")
    if len(parts) != 2:
        raise ValueError(f"Invalid problem name format: {name}")
    return tuple(parts)


def get_param_columns(output: DataFrame) -> list[str]:
    """Get the parameter columns from the output DataFrame"""
    cols = output.columns.tolist()
    param_cols = [col for col in cols[cols.index('name') + 1:]]
    return param_cols

def im2mat(img):
    """Converts and image to matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))

def mat2im(X, shape): 
    """Converts back a matrix to an image"""
    return X.reshape(shape)

def read_image(image_path: str):
    im = PIL.Image.open(image_path)
    return jnp.asarray(im, dtype=jnp.float64) / 255.0
    # image = plt.imread(image_path)
    # if image.dtype == np.uint8:
    #     image = image.astype(np.float64) / 255.0
    # elif image.dtype in [np.float32, np.float64]:
    #     image = np.clip(image, 0, 1)
    # return image

def match_shape(target_img: np.ndarray, src_img: np.ndarray):
    src_h, src_w = src_img.shape[:2]
    tgt_h, tgt_w = target_img.shape[:2]

    if tgt_h > src_h or tgt_w > src_w:
        interp = cv2.INTER_AREA 
    else:
        interp = cv2.INTER_CUBIC

    return cv2.resize(target_img, (src_w, src_h), interpolation=interp)
