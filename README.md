# 2D CFD Solver (CutCell)

This is a 2D CFD solver for incompressible laminar flow implemented in Rust.

## Features
- **Algorithm**: Block-coupled pressure-velocity solver
- **Discretization**: Finite Volume Method (FVM)
- **Grid**: Colocated grid with CutCell method for arbitrary geometries
- **Geometries**: Backwards Step, Channel with Obstacle
- **UI**: egui-based interface for real-time visualization and control

## How to Run

1. Ensure you have Rust installed.
2. Run the application:
   ```bash
   cargo run --release
   ```

## Controls
- **Geometry**: Select between Backwards Step and Channel with Obstacle.
- **Mesh Parameters**: Adjust min/max cell size for the CutCell mesher.
- **Solver Parameters**: Adjust timestep.
- **Plot Field**: Select which field to visualize (Pressure, Velocity X/Y/Mag).
- **Initialize / Reset**: Generates the mesh and initializes the solver.
- **Run / Pause**: Start or stop the simulation.

## Implementation Details
- **Mesh**: Quadtree-based CutCell generation using Signed Distance Functions (SDF).
- **Solver**: Block-coupled momentum/continuity system with GPU-accelerated Krylov solvers and AMG preconditioning.
- **Visualization**: Real-time plotting using `egui_plot`.
