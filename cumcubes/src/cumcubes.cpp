// Copyright 2021 Zhihao Liang
#include <torch/extension.h>
#include <iostream>
#include <cstdint>
#include <tuple>

#include "cumcubes.hpp"


std::vector<torch::Tensor> marching_cubes(
    const torch::Tensor& density_grid,
    const float thresh,
    const bool verbose=false
) {
    // check
    CHECK_INPUT(density_grid);
    TORCH_CHECK(density_grid.ndimension() == 3)
    
    std::vector<Tensor> results = marching_cubes_wrapper(density_grid, thresh, verbose);
    
    return results;
}

void save_mesh(
    const std::string filename,
    torch::Tensor vertices,
    torch::Tensor faces
) {
    CHECK_CONTIGUOUS(vertices);
    CHECK_CONTIGUOUS(faces);

    if (vertices.is_cuda()) { vertices = vertices.to(torch::kCPU); }
    if (faces.is_cuda()) { faces = faces.to(torch::kCPU); }
    save_mesh_wrapper(filename, vertices, faces);
}

