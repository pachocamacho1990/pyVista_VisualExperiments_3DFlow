# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a single-notebook Python project for 3D fluid flow visualization using PyVista. It visualizes backward-facing step CFD simulation data with Q-criterion vortex detection, isosurface rendering, and streamline visualization.

## Dependencies

```bash
pip install numpy pyvista trame trame-vtk trame-vuetify ipywidgets
```

## Running the Notebook

Open `3-flow.ipynb` in Jupyter and run cells sequentially. The notebook expects simulation data from an external `.npz` file (default path: `../numerical_fluid_dynamics/simulations/backward_facing_step_3d/outputs/solution_3d.npz`).

## Architecture

The notebook contains three functional layers:

**Data Loading** (`load_solution`): Loads `.npz` files containing velocity components (u, v, w) and spatial coordinates (x, y, z).

**Grid Processing** (`create_grid`, `compute_q_criterion`):
- Converts NumPy arrays to PyVista `RectilinearGrid`
- Arrays are transposed from (nz, ny, nx) to (nx, ny, nz) for PyVista compatibility
- Q-criterion computation uses velocity gradient tensor to identify vortex regions (Q > 0)

**Visualization** (`visualize`): Renders isosurfaces, streamlines, and geometry with configurable camera and output options.

## Key Technical Details

**Array transposition pattern** - PyVista expects (nx, ny, nz) ordering:
```python
u_t = u.transpose(2, 1, 0).flatten(order='F')
```

**PyVista backend configuration**:
```python
pv.set_jupyter_backend("trame")
pv.global_theme.trame.jupyter_server_name = "my_viewer"  # Keeps URL consistent
```

**Input data format** - `.npz` archive with keys: `u`, `v`, `w` (3D velocity arrays), `x`, `y`, `z` (1D coordinate arrays).

## Known Issues

- `max_time` parameter in `streamlines()` is deprecated in PyVista 0.46+; use `max_step_length` instead
- Reuse the plotter instance (`pl.clear()`) to avoid creating new viewer windows
