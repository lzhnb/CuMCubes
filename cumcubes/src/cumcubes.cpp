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

    torch::Tensor colors = torch::ones_like(vertices) * 0.5;

    std::ofstream ply_file(filename, std::ios::out | std::ios::binary);
    ply_file << "ply\n";
    ply_file << "format binary_little_endian 1.0\n";
    ply_file << "element vertex " << vertices.size(0) << std::endl;
    ply_file << "property float x\n";
    ply_file << "property float y\n";
    ply_file << "property float z\n";
    ply_file << "property float red\n";
    ply_file << "property float blue\n";
    ply_file << "property float green\n";
    ply_file << "element face " << faces.size(0) << std::endl;
    ply_file << "property list int int vertex_index\n";

    ply_file << "end_header\n";

    const int32_t num_vertices = vertices.size(0), num_faces = faces.size(0);

    torch::Tensor vertices_colors = torch::cat({vertices, colors}, 1); // [num_vertices, 4]
    const float* vertices_colors_ptr = vertices_colors.data_ptr<float>();
    ply_file.write((char *)&(vertices_colors_ptr[0]), num_vertices * 6 * sizeof(float));

    torch::Tensor faces_head = torch::ones({num_faces, 1},
        torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU)) * 3;
    
    torch::Tensor padded_faces = torch::cat({faces_head, faces}, 1); // [num_faces, 4]
    CHECK_CONTIGUOUS(padded_faces);
    
    const int32_t* faces_ptr = padded_faces.data_ptr<int32_t>();
    ply_file.write((char *)&(faces_ptr[0]), num_faces * 4 * sizeof(int32_t));

    ply_file.close();
}

