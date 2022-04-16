# Copyright (c) Zhihao Liang. All rights reserved.
import os
import torch
import mcubes
import numpy as np

import cumcubes

DENSITY_GRID = np.load(os.path.join(os.path.dirname(__file__), "data", "bunny.npy"))
print(f"DENSITY_GRID shape: ({DENSITY_GRID.shape[0]}, {DENSITY_GRID.shape[1]}, {DENSITY_GRID.shape[2]})")

if __name__ == "__main__":
    density_grid_cu = torch.tensor(DENSITY_GRID).cuda()
    with cumcubes.Timer("cuda marching cube: {:.6f}s"):
        vertices_cu, faces_cu = cumcubes.marching_cubes(density_grid_cu, 0, verbose=True) # verbose to print the number of vertices and faces
    with cumcubes.Timer("cumcubes save mesh: {:.6f}s\n"):
        cumcubes.save_mesh(vertices_cu, faces_cu, filename="bunny.ply")

    with cumcubes.Timer("cpu-mode cumcumbes cube: {:.6f}s\n"):
        vertices_cpu, faces_cpu = cumcubes.marching_cubes(density_grid_cu, 0, cpu=True)

    with cumcubes.Timer("cpu marching cube: {:.6f}s"):
        vertices_c, faces_c = mcubes.marching_cubes(DENSITY_GRID, 0)
    with cumcubes.Timer("mcubes save mesh: {:.6f}s"):
        mcubes.export_obj(vertices_c, faces_c, filename="bunny.obj")

    assert((vertices_cu.shape[0] == vertices_c.shape[0]))
    assert((faces_cu.shape[0] == faces_c.shape[0]))
    assert((vertices_cpu.numpy() == vertices_c).all())
    assert((faces_cpu.numpy() == faces_c).all())

