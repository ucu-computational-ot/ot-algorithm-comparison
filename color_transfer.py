import jax
jax.config.update("jax_enable_x64", True)


import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from algorithms.sinkhorn import sinkhorn
from algorithms.gradient_ascent import gradient_ascent_two_marginal
from algorithms.lbfgs import dual_lbfs_potentials

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


SAMPLE_SIZE = 128
BATCH_SIZE = 20000
IMAGES_DIR = os.path.join("datasets", "color_images")
EPSILON = 1e-3
images = [(image_file, read_image(os.path.join(IMAGES_DIR, image_file)))
          for image_file in os.listdir(IMAGES_DIR)]

algorithms = {
    'sinkhorn': sinkhorn,
    'grad-ascent': gradient_ascent_two_marginal,
    'dual-lbfs': dual_lbfs_potentials
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

        for solver_name, solver in algorithms.items():
            pbar.update(1)
            _, v = solver(source_measure, target_measure, C, epsilon=EPSILON)
            transported_image = entropic_transport(v, C, source_image, target_sample, batch_size=BATCH_SIZE, epsilon=EPSILON)
            transported_image = minmax(mat2im(transported_image, shape))
            results[(source_name, target_name)][solver_name] = transported_image


export_images(images_pairs, results, "color_transfer_results/moderate_regularization")