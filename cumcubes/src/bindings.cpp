// Copyright 2021 Zhihao Liang

// This file contains only Python bindings
#include <torch/extension.h>

#include "cumcubes.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("marching_cubes", &mc::marching_cubes);
    m.def("marching_cubes_func", &mc::marching_cubes_func);
    m.def("save_mesh_as_ply", &mc::save_mesh_as_ply);
}

