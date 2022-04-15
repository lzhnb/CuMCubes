# CuMCubes
`CuMCubes` is an **CUDA** implementation of the marching cubes algorithm to extract iso-surfaces from volumetric data. The volumetric data can be given as a three-dimensional `torch.Tensor` or as a Python function `f(x, y, z)`.

## Installation
```sh
python setup.py install
```
Or use
```sh
pip install .
```

## Example
```sh
# toy examples from the PyMCubes
python examples/sphere.py
python examples/function.py
```


## TODO
- [x] Python wrapper
- [x] Examples
- [x] Realizing `marching_cubes_func` (NOTE: need to accelerate, put the loop in cpp)
- [ ] C++ template support
- [ ] Optimize the code
- [ ] Release as python package
- [ ] Sparse Marching Cubes


## Acknowledgement
[instant-npg](https://github.com/NVlabs/instant-ngp)
[PyMCubes](https://github.com/pmneila/PyMCubes)
[AnalyticMesh](https://github.com/Gorilla-Lab-SCUT/AnalyticMesh)

