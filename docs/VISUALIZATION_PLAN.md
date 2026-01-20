# 3D Turbulent Flow Visualization Plan

A comprehensive guide for implementing advanced visualization techniques for turbulent and vortical fluid flows using PyVista.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Vortex Identification Methods](#2-vortex-identification-methods)
3. [Geometric Flow Visualization](#3-geometric-flow-visualization)
4. [Texture-Based Methods](#4-texture-based-methods)
5. [Volume Rendering](#5-volume-rendering)
6. [Lagrangian Coherent Structures](#6-lagrangian-coherent-structures)
7. [Derived Scalar Fields](#7-derived-scalar-fields)
8. [Topological Analysis](#8-topological-analysis)
9. [Time-Varying Data and Animation](#9-time-varying-data-and-animation)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [References](#11-references)

---

## 1. Introduction

### 1.1 The Challenge

Visualizing three-dimensional turbulent flows presents unique challenges:

- **High dimensionality**: Vector fields with 3 components at each point in 3D space
- **Multi-scale structures**: Turbulence contains eddies spanning orders of magnitude in size
- **Occlusion**: 3D structures obscure each other in any 2D projection
- **Information density**: Too much detail obscures the essential flow physics

### 1.2 Visualization Categories

Flow visualization techniques can be classified into four main categories:

| Category | Description | Examples |
|----------|-------------|----------|
| **Direct** | Map vector data directly to visual primitives | Arrow glyphs, color mapping |
| **Geometric** | Integrate along flow direction | Streamlines, pathlines, streaklines |
| **Texture-based** | Advect texture patterns with flow | Line Integral Convolution (LIC) |
| **Feature-based** | Extract and display flow structures | Vortex cores, separation surfaces |

### 1.3 Input Data Requirements

All visualizations expect CFD data in a standard format:

```python
# Required fields
u, v, w    # Velocity components (3D arrays)
x, y, z    # Grid coordinates (1D arrays)

# Optional fields (computed or provided)
p          # Pressure
rho        # Density
T          # Temperature
```

### 1.4 PyVista Array Conventions

PyVista expects data in `(nx, ny, nz)` ordering with Fortran-style flattening:

```python
# Transpose from CFD solver ordering (nz, ny, nx) to PyVista ordering
u_pv = u.transpose(2, 1, 0).flatten(order='F')
```

---

## 2. Vortex Identification Methods

Vortex identification remains an open problem in fluid mechanics—no universally accepted definition of a vortex exists. Multiple criteria have been proposed, each with strengths and limitations.

### 2.1 Velocity Gradient Tensor Decomposition

All major vortex identification methods begin with the velocity gradient tensor:

$$
\nabla \mathbf{u} = \begin{pmatrix}
\frac{\partial u}{\partial x} & \frac{\partial u}{\partial y} & \frac{\partial u}{\partial z} \\
\frac{\partial v}{\partial x} & \frac{\partial v}{\partial y} & \frac{\partial v}{\partial z} \\
\frac{\partial w}{\partial x} & \frac{\partial w}{\partial y} & \frac{\partial w}{\partial z}
\end{pmatrix}
$$

This tensor decomposes into symmetric and antisymmetric parts:

- **Strain rate tensor**: $\mathbf{S} = \frac{1}{2}(\nabla\mathbf{u} + \nabla\mathbf{u}^T)$
- **Rotation rate tensor**: $\mathbf{\Omega} = \frac{1}{2}(\nabla\mathbf{u} - \nabla\mathbf{u}^T)$

### 2.2 Q-Criterion (Hunt et al., 1988)

**Definition:**
$$Q = \frac{1}{2}\left(\|\mathbf{\Omega}\|_F^2 - \|\mathbf{S}\|_F^2\right)$$

**Physical interpretation:**
- $Q > 0$: Rotation dominates strain → vortex region
- $Q < 0$: Strain dominates rotation

**Strengths:**
- Simple to compute
- Well-established in literature
- Available directly in PyVista

**Weaknesses:**
- Threshold selection is arbitrary
- Cannot distinguish vortex strength from size
- Affected by strong shear layers

**PyVista implementation:**
```python
grid = grid.compute_derivative(scalars="velocity", qcriterion=True)
iso = grid.contour(isosurfaces=[q_threshold], scalars='qcriterion')
```

### 2.3 λ₂-Criterion (Jeong & Hussain, 1995)

**Definition:** Based on eigenvalues of $\mathbf{S}^2 + \mathbf{\Omega}^2$

A point is in a vortex core if $\lambda_2 < 0$ (the second eigenvalue is negative).

**Physical basis:** Identifies pressure minima due to rotation (excluding unsteady straining and viscous effects).

**Strengths:**
- More physically motivated than Q-criterion
- Better at excluding shear layers

**Weaknesses:**
- Computationally more expensive
- Threshold $\lambda_2 = 0$ often produces cluttered visualizations; small negative values work better
- Difficulty distinguishing individual vortices when many are present

**Implementation approach:**
```python
def compute_lambda2(grid):
    """Compute λ₂ criterion for vortex identification."""
    grad = grid.compute_derivative(scalars="velocity", gradient=True)["gradient"]
    grad = grad.reshape(-1, 3, 3)

    S = 0.5 * (grad + grad.transpose(0, 2, 1))
    Omega = 0.5 * (grad - grad.transpose(0, 2, 1))

    # S² + Ω²
    tensor = S @ S + Omega @ Omega

    # Compute eigenvalues and take the second largest
    eigenvalues = np.linalg.eigvalsh(tensor)  # Returns sorted ascending
    lambda2 = eigenvalues[:, 1]  # Second eigenvalue

    grid.point_data["lambda2"] = lambda2
    return grid
```

### 2.4 Δ-Criterion (Chong et al., 1990)

**Definition:** Based on the discriminant of the characteristic equation of $\nabla\mathbf{u}$:

$$\Delta = \left(\frac{Q}{3}\right)^3 + \left(\frac{R}{2}\right)^2$$

where $R = -\det(\nabla\mathbf{u})$ and $Q = -\frac{1}{2}\text{tr}(\nabla\mathbf{u}^2)$

**Physical interpretation:** $\Delta > 0$ indicates complex eigenvalues, implying local swirling motion.

**Notes:** Tends to produce more fragmented vortex structures compared to Q and λ₂.

### 2.5 Ω-Criterion (Liu et al., 2016)

**Definition:** Normalized ratio of vortical to total motion:

$$\Omega = \frac{\|\mathbf{\Omega}\|_F^2}{\|\mathbf{S}\|_F^2 + \|\mathbf{\Omega}\|_F^2 + \epsilon}$$

where $\epsilon$ is a small number to avoid division by zero.

**Range:** $\Omega \in [0, 1]$, where $\Omega > 0.5$ indicates vortex regions.

**Strengths:**
- Normalized, so threshold is consistent across flows
- Consistent results with Q and λ₂ in most regions

### 2.6 Rortex / Liutex (Liu et al., 2018)

**Definition:** Extracts only the rigid-body rotation component of motion, excluding shear.

**Advantages:**
- Highest correlation with actual swirling strength
- Not contaminated by pure shear
- Provides both magnitude and direction of rotation

**Status:** Newer method, not yet in standard libraries. Requires custom implementation.

### 2.7 Comparison Summary

| Method | Threshold | Shear Sensitivity | Computational Cost | PyVista Support |
|--------|-----------|-------------------|-------------------|-----------------|
| Q-criterion | Arbitrary | Moderate | Low | Built-in |
| λ₂-criterion | ~0 or small negative | Low | Medium | Custom |
| Δ-criterion | > 0 | Moderate | Medium | Custom |
| Ω-criterion | 0.5 | Low | Low | Custom |
| Rortex | Arbitrary | Very Low | High | Custom |

### 2.8 Practical Recommendations

1. **Start with Q-criterion** for quick exploration (built into PyVista)
2. **Use λ₂** when shear contamination is a concern
3. **Try Ω-criterion** for consistent thresholding across different flows
4. **Threshold selection**: Start at 1-10% of maximum value; adjust based on structures revealed

---

## 3. Geometric Flow Visualization

Geometric methods integrate particle trajectories through the velocity field.

### 3.1 Streamlines (Steady Flow)

**Definition:** Curves everywhere tangent to the instantaneous velocity field.

$$\frac{d\mathbf{x}}{ds} = \mathbf{u}(\mathbf{x})$$

**Use case:** Steady flows or instantaneous snapshots.

**PyVista implementation:**
```python
streamlines = grid.streamlines(
    vectors='velocity',
    source_center=(x0, y0, z0),
    source_radius=r,
    n_points=100,
    max_time=200.0,        # Deprecated in PyVista 0.46+
    # max_step_length=0.1  # Use this instead for newer versions
)
pl.add_mesh(streamlines.tube(radius=0.01), cmap='coolwarm')
```

**Seeding strategies:**
| Strategy | Description | Use Case |
|----------|-------------|----------|
| Point sphere | Random points in a sphere | General exploration |
| Inlet surface | Seed from boundary mesh | Capture entering flow |
| Rake (line) | Points along a line | Cross-section analysis |
| Critical points | Seed near saddles/foci | Capture separatrices |

### 3.2 Pathlines (Time-Varying Flow)

**Definition:** Actual trajectory of a fluid particle over time.

$$\frac{d\mathbf{x}}{dt} = \mathbf{u}(\mathbf{x}, t)$$

**Key difference from streamlines:** Accounts for temporal evolution of the velocity field.

**Implementation:** Requires time-series data and interpolation between timesteps.

### 3.3 Streaklines (Time-Varying Flow)

**Definition:** Locus of particles that have passed through a fixed point in space.

**Physical analogy:** Dye injection in a flow.

**Use case:** Best for visualizing time-dependent phenomena with temporal coherence.

### 3.4 Surface Streamlines (Skin Friction Lines)

For wall-bounded flows, visualize the limiting streamlines on solid surfaces:

```python
# Extract wall surface
wall = grid.extract_surface()
# Compute wall shear stress or use near-wall velocity
wall_streamlines = wall.streamlines_on_surface(vectors='wall_shear')
```

### 3.5 Streamline Rendering Techniques

| Technique | Description | PyVista Method |
|-----------|-------------|----------------|
| Lines | Simple polylines | Default output |
| Tubes | Cylindrical tubes around lines | `.tube(radius=r)` |
| Ribbons | Flat ribbons showing rotation | `.ribbon(width=w)` |
| Illuminated | Shaded for depth perception | Lighting options |

---

## 4. Texture-Based Methods

### 4.1 Line Integral Convolution (LIC)

**Concept:** Convolve a noise texture along streamlines to produce a coherent pattern that follows the flow.

**Advantages:**
- Shows entire vector field at pixel resolution
- Reveals all structural features without seeding decisions
- Intuitive visual appearance

**Limitations in 3D:**
- Volume LIC produces dense, hard-to-interpret images
- Requires careful depth cues and transparency

**Implementation status:** Not directly available in PyVista. Consider:
- VTK's `vtkImageDataLIC2D` for 2D slices
- Custom implementation with texture advection
- External tools (ParaView has LIC plugin)

### 4.2 Strategies for 3D LIC

1. **2D slices:** Apply LIC to planar cuts through the volume
2. **Surface LIC:** Apply to isosurfaces or boundaries
3. **Volume LIC with transparency:** Use transfer functions to show only high-interest regions
4. **Halos:** Add depth cues with visibility-based halos

### 4.3 Texture Advection

For time-varying flows, advect texture patterns with the flow:

```
T(x, t+Δt) = T(x - u·Δt, t)
```

This maintains temporal coherence and reveals transport patterns.

---

## 5. Volume Rendering

### 5.1 Overview

Volume rendering displays 3D scalar fields by casting rays through the data and accumulating color/opacity.

**Best for:** Visualizing scalar quantities like vorticity magnitude, Q-criterion, enstrophy throughout the volume.

### 5.2 Transfer Function Design

The transfer function maps scalar values to color and opacity:

```python
# PyVista volume rendering
pl.add_volume(
    grid,
    scalars='Q',
    cmap='hot',
    opacity='sigmoid',     # Predefined opacity curve
    # opacity=[0, 0, 0.2, 0.5, 1.0],  # Custom opacity points
)
```

**Opacity strategies for turbulence:**
- **Threshold:** Zero opacity below threshold, solid above
- **Ramp:** Linear increase from threshold
- **Sigmoid:** Smooth transition highlighting mid-range values
- **Highlight extremes:** Low opacity for moderate values, high for extremes

### 5.3 Multi-Dimensional Transfer Functions

Use multiple scalar fields to define opacity:

| Component | Purpose |
|-----------|---------|
| Velocity magnitude | Flow intensity |
| Q-criterion | Vortex strength |
| Helicity | Chirality/handedness |
| Divergence | Compressibility |

### 5.4 Depth Cues

Volume rendering lacks natural depth cues. Enhance with:
- Ambient occlusion
- Gradient-based shading
- Depth-based opacity attenuation

---

## 6. Lagrangian Coherent Structures

### 6.1 Concept

Lagrangian Coherent Structures (LCS) are material surfaces that organize fluid transport. They reveal:

- **Attracting LCS:** Where fluid particles converge
- **Repelling LCS:** Where fluid particles diverge
- **Elliptic LCS:** Lagrangian vortex boundaries

### 6.2 Finite-Time Lyapunov Exponent (FTLE)

**Definition:** Measures the maximum stretching rate of fluid elements over a finite time:

$$\text{FTLE} = \frac{1}{|T|} \ln \sqrt{\lambda_{\max}(\mathbf{C})}$$

where $\mathbf{C}$ is the Cauchy-Green deformation tensor.

**Computation:**
1. Seed particles on a regular grid
2. Advect particles forward (or backward) in time
3. Compute deformation gradient from final positions
4. Extract maximum eigenvalue of $\mathbf{C} = \mathbf{F}^T \mathbf{F}$

**Interpretation:**
- **Forward-time FTLE ridges:** Repelling LCS (unstable manifolds)
- **Backward-time FTLE ridges:** Attracting LCS (stable manifolds)

### 6.3 Implementation Approach

```python
def compute_ftle(grid, velocity_func, T, dt):
    """
    Compute FTLE field.

    Parameters:
    -----------
    grid : pyvista.RectilinearGrid
        Grid defining initial particle positions
    velocity_func : callable
        Function returning velocity at (x, y, z, t)
    T : float
        Integration time (positive=forward, negative=backward)
    dt : float
        Time step for integration
    """
    # 1. Get initial positions
    points = grid.points.copy()

    # 2. Advect particles
    final_points = advect_particles(points, velocity_func, T, dt)

    # 3. Compute deformation gradient at each point
    # Using finite differences of final positions
    F = compute_deformation_gradient(points, final_points, grid)

    # 4. Compute Cauchy-Green tensor C = F^T @ F
    C = np.einsum('...ji,...jk->...ik', F, F)

    # 5. Get maximum eigenvalue
    eigenvalues = np.linalg.eigvalsh(C)
    lambda_max = eigenvalues[..., -1]

    # 6. Compute FTLE
    ftle = np.log(np.sqrt(np.maximum(lambda_max, 1e-10))) / np.abs(T)

    grid.point_data['FTLE'] = ftle.flatten(order='F')
    return grid
```

### 6.4 Visualization

- Ridges in FTLE field correspond to LCS
- Use isosurfaces or volume rendering of FTLE
- Compare forward and backward FTLE for complete picture

### 6.5 Computational Considerations

- **Cost:** Particle advection is expensive for large grids
- **GPU acceleration:** Highly parallelizable
- **Time series required:** Need velocity data at multiple time steps

---

## 7. Derived Scalar Fields

Beyond vortex criteria, several scalar fields reveal turbulence characteristics.

### 7.1 Vorticity Magnitude

**Definition:** $|\boldsymbol{\omega}| = |\nabla \times \mathbf{u}|$

**PyVista:**
```python
grid = grid.compute_derivative(scalars="velocity", vorticity=True)
grid.point_data['vorticity_mag'] = np.linalg.norm(
    grid.point_data['vorticity'].reshape(-1, 3), axis=1
)
```

**Use:** General measure of rotation; includes shear contributions.

### 7.2 Enstrophy

**Definition:** $\mathcal{E} = \frac{1}{2}|\boldsymbol{\omega}|^2$

**Physical meaning:** Measure of rotational kinetic energy; related to dissipation in turbulence.

**Use:** Quantifying small-scale vorticity dynamics.

### 7.3 Helicity

**Definition:** $H = \mathbf{u} \cdot \boldsymbol{\omega}$

**Physical meaning:**
- Measures alignment between velocity and vorticity
- Indicates chirality (handedness) of flow structures
- Related to linking and knotting of vortex lines

**Properties:**
- Positive: Right-handed helical structures
- Negative: Left-handed helical structures
- Zero: Planar or non-helical flow

**Implementation:**
```python
def compute_helicity(grid):
    grid = grid.compute_derivative(scalars="velocity", vorticity=True)
    velocity = grid.point_data['velocity']
    vorticity = grid.point_data['vorticity']
    helicity = np.sum(velocity * vorticity, axis=1)
    grid.point_data['helicity'] = helicity
    return grid
```

### 7.4 Kinetic Energy

**Definition:** $E = \frac{1}{2}|\mathbf{u}|^2$

**Use:** Identify high-energy regions; useful for turbulence intensity.

### 7.5 Divergence

**Definition:** $\nabla \cdot \mathbf{u}$

**Use:**
- Should be ~0 for incompressible flows (verification)
- Non-zero indicates compressibility effects or numerical errors

**PyVista:**
```python
grid = grid.compute_derivative(scalars="velocity", divergence=True)
```

### 7.6 Strain Rate Magnitude

**Definition:** $|S| = \sqrt{2 S_{ij} S_{ij}}$

**Use:** Identifies regions of high deformation rate.

### 7.7 Summary Table

| Field | Formula | Physical Meaning | Sign |
|-------|---------|------------------|------|
| Vorticity | $\nabla \times \mathbf{u}$ | Local rotation rate | Vector |
| Enstrophy | $\frac{1}{2}\|\boldsymbol{\omega}\|^2$ | Rotational energy | Positive |
| Helicity | $\mathbf{u} \cdot \boldsymbol{\omega}$ | Flow chirality | ± |
| Q-criterion | $\frac{1}{2}(\|\Omega\|^2 - \|S\|^2)$ | Vortex indicator | ± |
| Divergence | $\nabla \cdot \mathbf{u}$ | Compressibility | ± |

---

## 8. Topological Analysis

### 8.1 Critical Points

Critical points are locations where $\mathbf{u} = \mathbf{0}$. They organize the flow topology.

**Classification (2D/surface):**

| Type | Eigenvalues | Flow Pattern |
|------|-------------|--------------|
| **Node (source)** | Both positive real | Flow radiates outward |
| **Node (sink)** | Both negative real | Flow converges inward |
| **Saddle** | Opposite signs | Flow approaches on one axis, departs on other |
| **Focus (source)** | Complex, positive real part | Spiral outward |
| **Focus (sink)** | Complex, negative real part | Spiral inward |
| **Center** | Pure imaginary | Closed orbits |

**Classification (3D):**
More complex; saddles have 2D stable/unstable manifolds.

### 8.2 Separation and Attachment

**Separation:** Flow leaves a surface along a separation line originating from a saddle point.

**Attachment:** Flow approaches a surface along an attachment line.

**Topological rule (surfaces):** $\sum N - \sum S = 0$

### 8.3 Detection Algorithm

```python
def find_critical_points(grid):
    """
    Find critical points where velocity magnitude is near zero.

    Returns approximate locations; refinement needed for exact positions.
    """
    velocity = grid.point_data['velocity']
    vel_mag = np.linalg.norm(velocity, axis=1)

    # Find local minima of velocity magnitude
    threshold = 0.01 * vel_mag.max()
    candidates = grid.points[vel_mag < threshold]

    # Classify based on velocity gradient eigenvalues
    # (requires computing Jacobian at each candidate)
    return candidates
```

### 8.4 Separatrices

Lines/surfaces connecting critical points that divide the flow into distinct regions.

**Visualization:** Seed streamlines from saddle points along eigenvector directions.

---

## 9. Time-Varying Data and Animation

### 9.1 Challenges

- **Data volume:** Hundreds of timesteps × full 3D fields
- **Temporal coherence:** Avoid flickering or discontinuous features
- **Feature tracking:** Same vortex at different times

### 9.2 Animation Approaches

**1. Sequential frame rendering:**
```python
pl.open_gif('animation.gif')
for timestep in timesteps:
    data = load_timestep(timestep)
    grid = create_grid(data)

    pl.clear()
    # Add visualization elements
    pl.write_frame()
pl.close()
```

**2. Pathlines/streaklines:** Integrate particle positions across time.

**3. Feature tracking:** Identify and follow coherent structures.

### 9.3 Temporal Filtering

Smooth time-varying visualizations to reduce noise:
- Moving average of scalar fields
- Temporal interpolation between frames
- Feature-based filtering (only track persistent structures)

### 9.4 Data Management

For large time series:
- **In-situ processing:** Compute derived quantities during simulation
- **Temporal subsampling:** Store every Nth timestep
- **Region of interest:** Only save high-activity regions
- **Compression:** Use lossy compression for visualization data

---

## 10. Implementation Roadmap

### Phase 1: Core Infrastructure

**Goal:** Establish reusable functions for arbitrary geometries.

| Task | Description | Priority |
|------|-------------|----------|
| Generic data loader | Support multiple CFD formats (npz, VTK, CGNS) | High |
| Grid utilities | Coordinate transforms, slicing, subsampling | High |
| Velocity gradient tensor | Efficient computation with PyVista | High |
| Scalar field module | Q, λ₂, vorticity, helicity, enstrophy | High |

**Deliverable:** `core/` module with reusable functions.

### Phase 2: Vortex Visualization

**Goal:** Implement and compare vortex identification methods.

| Task | Description | Priority |
|------|-------------|----------|
| Q-criterion notebook | Document threshold selection strategies | High |
| λ₂ implementation | Add to scalar field module | Medium |
| Ω-criterion | Normalized method for consistent thresholds | Medium |
| Comparison notebook | Side-by-side visualization of methods | Medium |

**Deliverable:** `notebooks/vortex_identification.ipynb`

### Phase 3: Advanced Streamlines

**Goal:** Sophisticated geometric visualization.

| Task | Description | Priority |
|------|-------------|----------|
| Seeding strategies | Inlet, rake, sphere, surface-based | High |
| Rendering options | Tubes, ribbons, color mapping | Medium |
| Surface streamlines | Wall shear stress visualization | Medium |
| Critical point seeding | Automatic saddle/focus detection | Low |

**Deliverable:** `notebooks/streamlines_advanced.ipynb`

### Phase 4: Volume Rendering

**Goal:** Effective 3D scalar field visualization.

| Task | Description | Priority |
|------|-------------|----------|
| Transfer function presets | Turbulence-optimized opacity curves | High |
| Multi-field rendering | Combine Q-criterion with velocity | Medium |
| Slice + volume hybrid | 2D slices with 3D context | Medium |

**Deliverable:** `notebooks/volume_rendering.ipynb`

### Phase 5: Lagrangian Analysis

**Goal:** Implement FTLE and LCS detection.

| Task | Description | Priority |
|------|-------------|----------|
| Particle advection | Efficient integration (RK4) | High |
| FTLE computation | Forward and backward time | High |
| LCS extraction | Ridge detection in FTLE field | Medium |
| Time series support | Load and interpolate temporal data | High |

**Deliverable:** `notebooks/lagrangian_coherent_structures.ipynb`

### Phase 6: Animation and Interaction

**Goal:** Time-varying visualization and user interaction.

| Task | Description | Priority |
|------|-------------|----------|
| GIF/video export | Orbital rotation, time evolution | High |
| Interactive widgets | Threshold sliders, camera presets | Medium |
| Pathline animation | Particle tracking visualization | Medium |

**Deliverable:** `notebooks/animation.ipynb`

### Folder Structure

```
pyVista_VisualExperiments_3DFlow/
├── README.md
├── LICENSE
├── CLAUDE.md
├── .gitignore
│
├── docs/
│   └── VISUALIZATION_PLAN.md      # This document
│
├── core/                           # Reusable modules (future)
│   ├── __init__.py
│   ├── loaders.py                 # Data loading utilities
│   ├── grids.py                   # Grid manipulation
│   ├── scalars.py                 # Derived scalar fields
│   └── vortex.py                  # Vortex identification
│
├── notebooks/
│   ├── 3-flow.ipynb               # Current: backward-facing step
│   ├── vortex_identification.ipynb
│   ├── streamlines_advanced.ipynb
│   ├── volume_rendering.ipynb
│   ├── lagrangian_coherent_structures.ipynb
│   └── animation.ipynb
│
└── examples/                       # Output images and animations
    └── my_flow.png
```

---

## 11. References

### Vortex Identification

1. Hunt, J.C.R., Wray, A.A., & Moin, P. (1988). Eddies, streams, and convergence zones in turbulent flows. *Center for Turbulence Research Report CTR-S88*.

2. Jeong, J., & Hussain, F. (1995). On the identification of a vortex. *Journal of Fluid Mechanics*, 285, 69-94.

3. Liu, C., et al. (2016). New omega vortex identification method. *Science China Physics, Mechanics & Astronomy*, 59(8), 684711.

4. Liu, C., et al. (2018). Rortex—A new vortex vector definition and vorticity tensor and vector decompositions. *Physics of Fluids*, 30(3), 035103.

### Lagrangian Coherent Structures

5. Haller, G. (2015). Lagrangian coherent structures. *Annual Review of Fluid Mechanics*, 47, 137-162.

6. Shadden, S.C., Lekien, F., & Marsden, J.E. (2005). Definition and properties of Lagrangian coherent structures from finite-time Lyapunov exponents in two-dimensional aperiodic flows. *Physica D*, 212(3-4), 271-304.

### Flow Topology

7. Perry, A.E., & Fairlie, B.D. (1974). Critical points in flow patterns. *Advances in Geophysics*, 18, 299-315.

8. Tobak, M., & Peake, D.J. (1982). Topology of three-dimensional separated flows. *Annual Review of Fluid Mechanics*, 14(1), 61-85.

### Texture-Based Visualization

9. Cabral, B., & Leedom, L.C. (1993). Imaging vector fields using line integral convolution. *Proceedings of SIGGRAPH '93*, 263-270.

### Software

10. Sullivan, C.B., & Kaszynski, A. (2019). PyVista: 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK). *Journal of Open Source Software*, 4(37), 1450. https://doi.org/10.21105/joss.01450

### Online Resources

- [PyVista Documentation](https://docs.pyvista.org/)
- [PyVista Streamlines Example](https://docs.pyvista.org/examples/01-filter/streamlines.html)
- [PyVista Compute Derivative](https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.compute_derivative)
- [VTK Documentation](https://vtk.org/documentation/)

---

## Citation

This project uses [PyVista](https://docs.pyvista.org/) for 3D visualization. If you use this work, please cite PyVista:

> Sullivan, C.B., & Kaszynski, A. (2019). PyVista: 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK). *Journal of Open Source Software*, 4(37), 1450. https://doi.org/10.21105/joss.01450

**BibTeX:**
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

---

*Document version: 1.0*
*Last updated: 2025*
