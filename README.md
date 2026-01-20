# PyVista 3D Flow Visualization

Experiments in visualizing 3D turbulent and complex fluid flows from CFD simulations using [PyVista](https://docs.pyvista.org/).

## Features

- **Q-criterion vortex detection** - Identify and render vortical structures
- **Isosurface rendering** - Visualize scalar fields (velocity, pressure, etc.)
- **Streamlines** - Trace fluid particle paths through the flow
- **Interactive 3D exploration** - Rotate, zoom, and inspect flow features

## Installation

```bash
pip install numpy pyvista trame trame-vtk trame-vuetify ipywidgets
```

## Data Format

Notebooks expect simulation data in NumPy `.npz` format:

| Key | Description |
|-----|-------------|
| `u`, `v`, `w` | Velocity components (3D arrays) |
| `x`, `y`, `z` | Grid coordinates (1D arrays) |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `3-flow.ipynb` | Backward-facing step flow with Q-criterion and streamlines |

## Quick Start

```python
import numpy as np
import pyvista as pv

pv.set_jupyter_backend("trame")

# Load data
data = np.load("simulation.npz")
grid = pv.RectilinearGrid(data['x'], data['y'], data['z'])

# Add velocity field
u = data['u'].transpose(2, 1, 0).flatten(order='F')
v = data['v'].transpose(2, 1, 0).flatten(order='F')
w = data['w'].transpose(2, 1, 0).flatten(order='F')
grid.point_data['velocity'] = np.column_stack((u, v, w))

# Visualize streamlines
pl = pv.Plotter()
streamlines = grid.streamlines(vectors='velocity', n_points=100)
pl.add_mesh(streamlines.tube(radius=0.01), cmap='coolwarm')
pl.show()
```

## Documentation

See [docs/VISUALIZATION_PLAN.md](docs/VISUALIZATION_PLAN.md) for a comprehensive guide on 3D turbulent flow visualization techniques and the implementation roadmap.

## Citation

This project uses [PyVista](https://docs.pyvista.org/) for 3D visualization. If you use this work, please cite PyVista:

> Sullivan, C.B., & Kaszynski, A. (2019). PyVista: 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK). *Journal of Open Source Software*, 4(37), 1450. https://doi.org/10.21105/joss.01450

```bibtex
@article{sullivan2019pyvista,
  doi = {10.21105/joss.01450},
  url = {https://doi.org/10.21105/joss.01450},
  year = {2019},
  month = {May},
  publisher = {The Open Journal},
  volume = {4},
  number = {37},
  pages = {1450},
  author = {Bane Sullivan and Alexander Kaszynski},
  title = {{PyVista}: {3D} plotting and mesh analysis through a streamlined interface for the {Visualization Toolkit} ({VTK})},
  journal = {Journal of Open Source Software}
}
```

## License

MIT License - see [LICENSE](LICENSE)
