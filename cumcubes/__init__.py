# Copyright (c) Zhihao Liang. All rights reserved.
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
import numpy as np

from . import src as _C
from .utils import Timer

def marching_cubes(
    density_grid: Union[torch.Tensor, np.ndarray],
    thresh: float,
    scale: Optional[Union[float, Sequence]]=None,
    verbose: bool=False
) -> Tuple[torch.Tensor]:
    """cuda implementation of marching cubes

    Args:
        density_grid (Union[torch.Tensor, np.ndarray]):
            input density grid to realize marching cube
        thresh (float):
            thresh of marching cubes
        scale (Optional[Union[float, Sequence]], optional):
            the scale of density grid. Defaults to None.
        verbose (bool, optional):
            print verbose informations or not. Defaults to False.

    Returns:
        Tuple[torch.Tensor]: vertices and faces
    """
    assert torch.cuda.is_available(), "this package depends on the cuda avaliable PyTorch, please fix it"

    # process density_grid
    if isinstance(density_grid, np.ndarray): density_grid = torch.tensor(density_grid)
    density_grid = density_grid.cuda()
    density_grid = density_grid.to(torch.float32)

    lower: List[float]
    upper: List[float]
    # process scale as the bounding box
    if scale is None:
        lower = [0.0, 0.0, 0.0]
        upper = [density_grid.shape[0], density_grid.shape[1], density_grid.shape[2]]
    elif isinstance(scale, float):
        lower = [0.0, 0.0, 0.0]
        upper = [scale, scale, scale]
    elif isinstance(scale, (list, tuple, np.ndarray, torch.Tensor)):
        if len(scale) == 3:
            lower = [0.0, 0.0, 0.0]
            upper = [i for i in scale]
        elif len(scale) == 2:
            if isinstance(scale[0], float):
                lower = [scale[0]] * 3
                upper = [scale[1]] * 3
            else:
                assert len(scale[0]) == len(scale[1]) == len(scale[2]) == 3
                lower = [i for i in scale[0]]
                upper = [i for i in scale[1]]
        else:
            raise TypeError()
    else:
        raise TypeError()

    vertices, faces = _C.marching_cubes(density_grid, thresh, lower, upper, verbose)
    return vertices, faces


def save_mesh(
    vertices: Union[torch.Tensor, np.ndarray],
    faces: Union[torch.Tensor, np.ndarray],
    colors: Optional[Union[torch.Tensor, np.ndarray]]=None,
    filename: Union[str, Path] = "temp.ply"
) -> None:
    """save mesh into the given filename

    Args:
        vertices (Union[torch.Tensor, np.ndarray]): vertices of the mesh to save
        faces (Union[torch.Tensor, np.ndarray]): faces of the mesh to save
        colors (Optional[Union[torch.Tensor, np.ndarray]], optional):
            vertices of the mesh to save. Defaults to None.
        filename (Union[str, Path], optional):
            the save path. Defaults to "temp.ply".
    """

    if isinstance(filename, Path):
        filename = str(filename)

    if isinstance(vertices, np.ndarray): vertices = torch.tensor(vertices)
    if isinstance(faces, np.ndarray): faces = torch.tensor(faces)

    # process colors
    if colors is None:
        colors = torch.ones_like(vertices) * 127
    elif isinstance(colors, np.ndarray):
        colors = torch.tensor(colors)
    colors = colors.to(torch.uint8)

    if filename.endswith(".ply"):
        _C.save_mesh_as_ply(filename, vertices, faces, colors)
    else:
        raise NotImplementedError()


__all__ = ["Timer", "marching_cubes", "save_mesh"]
