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

def load_config_info(config: dict)-> tuple:
    """Get miscellaneous info from the config"""
    return (
        config['bin-number'], 
        config['batch-size'],
        config['pair-number'],
        config['images-dir'],
        config['rng-seed'],
        config.get('drop-columns', [])
    )

def get_image_problems(data: dict[str, ImageData], image_pairs: set[tuple]) -> list[TwoMarginalProblem]:
    """Get the image pairs as problems for the experiment"""
    return [
        ColorTransferProblem(
            name=f"{source_name} -> {target_name}",
            mu=data[source_name].get_grid(),
            nu=data[target_name].get_grid(),
            cost_fn=cost_euclid,
            source_image=data[source_name].get_image(),
            target_image=data[target_name].get_image(),
        )
        for source_name, target_name in image_pairs
    ]
    return [
        TwoMarginalProblem(
            name=f"{source_name} -> {target_name}",
            mu=data[source_name].get_grid(),
            nu=data[target_name].get_grid(),
            cost_fn=cost_euclid
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
