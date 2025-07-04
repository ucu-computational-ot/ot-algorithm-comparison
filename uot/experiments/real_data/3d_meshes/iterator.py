import os
import random
import open3d as o3d
import numpy as np
from typing import Iterator, Optional, Tuple

from uot.data.measure import DiscreteMeasure
from uot.problems.two_marginal import TwoMarginalProblem
from uot.problems.problem_generator import ProblemGenerator
from uot.utils.types import ArrayLike


def _load_and_sample_mesh(
    file_path: str,
    n_points: Optional[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a mesh from file and sample points uniformly.

    Returns:
        points: np.ndarray of shape (N, 3)
        colors: np.ndarray of shape (N, 3) or ones if no color
    """
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertex_colors():
        # default to white
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.ones((np.asarray(mesh.vertices).shape[0], 3))
        )
    verts = np.asarray(mesh.vertices)
    total = verts.shape[0]
    if n_points is None or n_points >= total:
        sampled = mesh
    else:
        sampled = mesh.sample_points_uniformly(number_of_points=n_points)
    pts = np.asarray(sampled.points)
    cols = np.asarray(sampled.colors)
    return pts, cols


class MeshProblemIterator(Iterator[TwoMarginalProblem]):
    """
    Generator & iterator for 3D mesh-based OT problems.

    Each dataset is a pair of meshes sampled at n_points, producing
    TwoMarginalProblem instances one at a time, lazily.
    """

    def __init__(
        self,
        name: str,
        mesh_folder: str,
        num_points: Optional[int] = None,
        num_pairs: int = 10,
        color_mode: str = "separate",
        cost_fn: ArrayLike = None,
        seed: int = 42,
    ):
        super().__init__()
        if cost_fn is None:
            raise ValueError("cost_fn must be provided")
        self.name = name
        self.mesh_folder = mesh_folder
        self.n_points = num_points
        self.num_pairs = num_pairs
        self.color_mode = color_mode
        self.cost_fn = cost_fn
        random.seed(seed)

        # collect mesh file paths
        self.mesh_files = [
            os.path.join(mesh_folder, f)
            for f in os.listdir(mesh_folder)
            if f.lower().endswith('.ply')
        ]
        if len(self.mesh_files) < 2:
            raise ValueError(f"Need at least 2 meshes in {mesh_folder}")

        # prepare iterator over generated problems
        self._gen = self._make_generator()

    def _make_generator(self) -> Iterator[TwoMarginalProblem]:
        # shuffle and create unique pairs
        files = self.mesh_files.copy()
        random.shuffle(files)

        pairs = []
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                pairs.append((files[i], files[j]))
        if len(pairs) > self.num_pairs:
            pairs = pairs[: self.num_pairs]

        for src_path, tgt_path in pairs:
            src_pts, src_cols = _load_and_sample_mesh(src_path, self.n_points)
            tgt_pts, tgt_cols = _load_and_sample_mesh(tgt_path, self.n_points)

            channel_idxs = [0, 1, 2] if self.color_mode == 'separate' else [
                {'r':0,'g':1,'b':2}[self.color_mode]
            ]
            for ch in channel_idxs:
                mu_w = src_cols[:, ch]
                mu_w /= mu_w.sum()
                nu_w = tgt_cols[:, ch]
                nu_w /= nu_w.sum()

                mu = DiscreteMeasure(points=src_pts.T, weights=mu_w)
                nu = DiscreteMeasure(points=tgt_pts.T, weights=nu_w)
                prob = TwoMarginalProblem(
                    name=f"{self.name}_{os.path.basename(src_path)}_{os.path.basename(tgt_path)}_ch{ch}",
                    mu=mu,
                    nu=nu,
                    cost_fn=self.cost_fn,
                )
                yield prob

    def __iter__(self) -> "MeshProblemIterator":
        return self

    def __next__(self) -> TwoMarginalProblem:
        try:
            return next(self._gen)
        except StopIteration:
            # reset for possible reuse
            self._gen = self._make_generator()
            raise

    def __len__(self) -> int:
        # total number of problems to be generated
        base_pairs = len(self.mesh_files) * (len(self.mesh_files) - 1) // 2
        count = min(base_pairs, self.num_pairs)
        channels = 3 if self.color_mode == 'separate' else 1
        return count * channels
