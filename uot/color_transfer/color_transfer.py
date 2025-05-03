import jax
jax.config.update("jax_enable_x64", True)

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ot.utils import dist

from uot.algorithms.sinkhorn import sinkhorn
from uot.algorithms.gradient_ascent import gradient_ascent_two_marginal
from uot.algorithms.lbfgs import dual_lbfs_potentials
from uot.algorithms.lp import pot_lp
from uot.algorithms.col_gen import col_gen


rng = np.random.RandomState(42)

def im2mat(img):
    """Converts and image to matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))

def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)

def minmax(img):
    return np.clip(img, 0, 1)

def get_samples(X1, X2, nb=500):
    idx1 = rng.randint(X1.shape[0], size=(nb,))
    idx2 = rng.randint(X2.shape[0], size=(nb,))

    Xs = X1[idx1, :]
    Xt = X2[idx2, :]
    return Xs, Xt

def read_image(image_path: str):
    image = plt.imread(image_path).astype(np.float64) / 256
    return image[
            (image.shape[0] - 640) // 2 : (image.shape[0] + 640) // 2,
            (image.shape[1] - 640) // 2 : (image.shape[1] + 640) // 2
        ]

def show_images(result):
    _, axes = plt.subplots(len(images_pairs), len(algorithms) + 2, figsize=(15, 6))

    for image_ind in range(len(results)):

        axes[image_ind, 0].imshow(images_pairs[image_ind][0])
        axes[image_ind, 0].axis("off")
        if image_ind == 0:
            axes[image_ind, 0].set_title("Source")

        for i, (name, img) in enumerate(results[image_ind].items()):
            axes[image_ind, i + 1].imshow(img)
            axes[image_ind, i + 1].axis("off")
            if image_ind == 0:
                axes[image_ind, i + 1].set_title(name)

        axes[image_ind, -1].imshow(images_pairs[image_ind][1])
        axes[image_ind, -1].axis("off")

        if image_ind == 0:
            axes[image_ind, -1].set_title("Target")

    plt.tight_layout()
    plt.show()

def export_images(image_pairs, results: dict, save_path: str):
    for (source_name, source_image), (target_name, target_image) in image_pairs:
        _, axes = plt.subplots(1, len(algorithms) + 2, figsize=(15, 5))

        axes[0].imshow(source_image)
        axes[0].axis("off")

        for i, (name, img) in enumerate(results[(source_name, target_name)].items()):
            axes[i + 1].imshow(img)
            axes[i + 1].set_title(name)
            axes[i + 1].axis("off")

        axes[-1].imshow(target_image)
        axes[-1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{source_name}-{target_name}.png"))


def get_cost_matrix(X: np.ndarray, Y: np.ndarray):
    return np.linalg.norm((X[:,None] - Y), axis=-1)


def entropic_transport(v: np.ndarray, C: np.ndarray, source_image: np.ndarray,
                       target_sample: np.ndarray, batch_size=128, epsilon=1e-2) -> np.ndarray:
    indices = np.arange(source_image.shape[0])
    batch_indices = [indices[i: i + batch_size] for i in range(0, len(indices), batch_size)]

    transported = []

    for i, batch_index in enumerate(batch_indices):
        C = get_cost_matrix(source_image[batch_index], target_sample)
        K = np.exp(-C/epsilon + v)
        transported_X = np.dot(K, target_sample) / np.sum(K, axis=1)[:, None]
        transported.append(transported_X)
    return np.concatenate(transported, axis=0)


def emd_transport(coupling: np.ndarray, source_image: np.ndarray, source_sample: np.ndarray,
                  target_sample: np.ndarray, batch_size=128):
    indices = np.arange(source_image.shape[0])
    batch_indices = [
        indices[i : i + batch_size]
        for i in range(0, len(indices), batch_size)
    ]

    transp_Xs = []
    for batch_index in batch_indices:
        D0 = dist(source_image[batch_index], source_sample)
        idx = np.argmin(D0, axis=1)

        transp = coupling / np.sum(coupling, axis=1)[:, None]
        transp = np.nan_to_num(transp, nan=0, posinf=0, neginf=0)
        transp_Xs_ = np.dot(transp, target_sample)

        transp_Xs_ = transp_Xs_[idx, :] + source_image[batch_index] - source_sample[idx, :]

        transp_Xs.append(transp_Xs_)

    transp_Xs = np.concatenate(transp_Xs, axis=0)

    return transp_Xs


SAMPLE_SIZE = 128
BATCH_SIZE = 20000
IMAGES_DIR = os.path.join("datasets", "color_images")
EPSILON = 1e-1
images = [(image_file, read_image(os.path.join(IMAGES_DIR, image_file)))
          for image_file in os.listdir(IMAGES_DIR)]

ENTROPIC_TRANSPORT = 'entropic'
EMD_TRANSPORT = 'emd'

algorithms = {
    'sinkhorn': (sinkhorn, ENTROPIC_TRANSPORT),
    'grad-ascent': (gradient_ascent_two_marginal, ENTROPIC_TRANSPORT),
    'dual-lbfs': (dual_lbfs_potentials, ENTROPIC_TRANSPORT),
    'lp': (pot_lp, EMD_TRANSPORT),
    'col_gen': (col_gen, EMD_TRANSPORT)
}

images_pairs = list(itertools.permutations(images, 2))

results = {}


with tqdm(total=len(images_pairs) * len(algorithms), desc="Running color transfer") as pbar:
    for source_image, target_image in images_pairs:
        source_name, source_image = source_image
        target_name, target_image = target_image

        results[(source_name, target_name)] = {}

        shape = source_image.shape
        source_image, target_image = im2mat(source_image), im2mat(target_image)
        source_sample, target_sample = get_samples(source_image, target_image, SAMPLE_SIZE)
        source_measure, target_measure = np.ones(SAMPLE_SIZE) / SAMPLE_SIZE, np.ones(SAMPLE_SIZE) / SAMPLE_SIZE
        C = get_cost_matrix(source_sample, target_sample)

        for solver_name, (solver, transport_name) in algorithms.items():
            pbar.update(1)

            if transport_name == ENTROPIC_TRANSPORT:
                _, v = solver(source_measure, target_measure, C, epsilon=EPSILON)
                transported_image = entropic_transport(v, C, source_image, target_sample, batch_size=BATCH_SIZE, epsilon=EPSILON)
            elif transport_name == EMD_TRANSPORT:
                P, _ = solver(source_measure, target_measure, C)
                transported_image = emd_transport(coupling=P, source_image=source_image,
                                                  source_sample=source_sample, target_sample=target_sample,
                                                  batch_size=BATCH_SIZE)
            
            transported_image = minmax(mat2im(transported_image, shape))
            results[(source_name, target_name)][solver_name] = transported_image


export_images(images_pairs, results, "color_transfer_results/")