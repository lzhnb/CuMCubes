// Copyright 2021 Zhihao Liang

// This file contains only Python bindings
#include <torch/extension.h>

#include "cumcubes.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("marching_cubes", &marching_cubes);
    m.def("save_mesh", &save_mesh);
}

