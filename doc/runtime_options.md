# Runtime Options
:::{note}
Options in the documentation are specified by command-line flag and YAML format interchangeably.
See {ref}`yaml-options` for details.
:::

## Common Options
HONEE is controlled via command-line options.
The following options are common among all problem types:

:::{list-table} Common Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value

* - `-ceed`
  - CEED resource specifier
  - `/cpu/self/opt/blocked`

* - `-problem`
  - Problem to solve (`advection`, `density_current`, `euler_vortex`, `shocktube`, `blasius`, `channel`, `gaussian_wave`, and `taylor_green`)
  - `density_current`

* - `-implicit`
  - Use implicit time integrator formulation
  -

* - `-degree`
  - Polynomial degree of tensor product basis (must be >= 1)
  - `1`

* - `-q_extra`
  - Number of extra quadrature points
  - `0`

* - `-ts_monitor_wall_force`
  - Viewer for the force on each no-slip wall, e.g., `ascii:force.csv:ascii_csv` to write a CSV file.
  -

* - `-ts_monitor_total_kinetic_energy`
  - Viewer for the total kinetic energy in the domain and other terms, e.g., `ascii:total_ke.csv:ascii_csv` to write a CSV file.
  -

* - `-ts_monitor_total_kinetic_energy_interval`
  - Number of timesteps between calculating and printing the total kinetic energy
  - `1`

* - `-ts_monitor_cfl`
  - Viewer for the min/max CFL in the domain e.g., `ascii:cfl.csv:ascii_csv` to write a CSV file.
  -

* - `-ts_monitor_cfl_interval`
  - Number of timesteps between calculating and printing the min/max CFL
  - `1`

* - `-honee_check_step_interval`
  - Number of time steps between checking the solution for Nans. Negative interval indicates it will not run.
  - `-1`

* - `-honee_max_wall_time_duration`
  - Wall clock duration of simulation before it should be stopped. Acceptable formats are `hh`, `hh:mm`, and `hh:mm:ss`. Simulation is stopped at `start_time + duration - buffer`
  - `'0'`

* - `-honee_max_wall_time_buffer`
  - Approximate time required to exit simulation cleanly (write checkpoints, etc.)
  - `'00:01'`

* - `-honee_max_wall_time_interval`
  - Number of time steps between checking whether simulation should stop based on `-honee_max_wall_time_duration`
  - `1`

* - `-mesh_transform`
  - Transform the mesh, usually for an initial box mesh.
  - `none`

* - `-help`
  - View comprehensive information about run-time options
  -
:::

### File I/O Options

:::{list-table} File I/O Options
:header-rows: 1

* - Option
  - Description
  - Default value

* - `-dm_plex_filename`
  - Filename of mesh file to load in
  -

* - `-ts_monitor_solution`
  - PETSc output format, such as `cgns:output-%d.cgns` (requires PETSc `--download-cgns`)
  -

* - `-ts_monitor_solution_interval`
  - Number of time steps between visualization output frames.
  - `1`

* - `-viewer_cgns_batch_size`
  - Number of frames written per CGNS file if the CGNS file name includes a format specifier (`%d`).
  - `20`

* - `-checkpoint_interval`
  - Number of steps between writing binary checkpoints. `0` has no output, `-1` outputs final state only
  - `0`

* - `-checkpoint_vtk`
  - Checkpoints include VTK (`*.vtu`) files for visualization. Consider `-ts_monitor_solution`instead.
  - `false`

* - `-viz_refine`
  - Use regular refinement for VTK visualization
  - `0`

* - `-output_dir`
  - Output directory for binary checkpoints and VTK files (if enabled).
  - `.`

* - `-output_add_stepnum2bin`
  - Whether to add step numbers to output binary files
  - `false`

* - `-continue_filename`
  - Path to file from which to continue from. Either binary file or CGNS
  -

* - `-ts_eval_times`
  - Sets intermediate time points to evaluate the solution at. See [PETSc documentation](https://petsc.org/main/manualpages/TS/TSSetEvaluationTimes/) for more details.
  -

* - `-ts_eval_solutions_view`
  - PETSc output format for `-ts_eval_times` solutions to be written to
  -

:::

Note that to use `-continue_filename` with CGNS files, the same file must be used with `-dm_plex_filename` and `-dm_plex_cgns_parallel`.

### Testing Options

:::{list-table} Testing Options
:header-rows: 1

* - Option
  - Description
  - Default value

* - `-test_type`
  - Run in test mode and specify whether solution (`solver`) or turbulent statistics (`turb_spanstats`) output should be verified
  - `none`

* - `-compare_final_state_atol`
  - Test absolute tolerance
  - `1E-11`

* - `-compare_final_state_filename`
  - Test filename
  -

* - `-newtonian_unit_tests`
  - Run unit tests of Newtonian state variable transformation functions
  - `false`

* - `-riemann_solver_unit_tests`
  - Run unit tests of Riemann problem solver and it's Jacobian
  - `false`
:::

### Logging Options

Some of these are PETSc options here as reference, while others are custom to HONEE.

:::{list-table} Logging Options
:header-rows: 1

* - Option
  - Description
  - Default value

* - `-ts_pre_view`
  - View PETSc `TS` solver configuration before it begins it's solve
  -

* - `-mass_ksp_view_pre_ts_solve`
  - View mass KSP once before `TSSolve()` is called
  -

* - `-ts_monitor`
  - View log for every timestep taken by the `TS` solver
  -

* - `-snes_monitor`
  - View log for every iteration taken by the `SNES` solver
  -

* - `-snes_converged_reason`
  - View convergence reason for every iteration taken by the `SNES` solver
  -

* - `-ksp_converged_reason`
  - View convergence reason for every iteration taken by the `KSP` solver
  -

* - `-log_view`
  - View PETSc performance log
  -

* - `-ksp_post_solve_residual`
  - Print KSP residual summary information after each
  -
:::

### Nondimensionalization
These options allow the units used during solving to be changed.
For problems where solution components can differ by many orders of magnitude, this can help problem conditioning

:::{caution}
This feature may be broken for certain use cases. If you discover a bug related to nondimensionalization, please submit a issue to the HONEE repo so that we can address it.
:::

:::{list-table} Nondimensionalization Options
:header-rows: 1

* - Option
  - Description
  - Default value

* - `-units_meter`
  - 1 meter in scaled length units
  - `1`

* - `-units_second`
  - 1 second in scaled time units
  - `1`

* - `-units_kilogram`
  - 1 kilogram in scaled mass units
  - `1`

* - `-units_Kelvin`
  - 1 Kelvin in scaled temperature units
  - `1`
:::

(bc-flags)=
## Boundary conditions

:::{list-table} Boundary Condition Options
:header-rows: 1

* - Option
  - Description

* - `-bc_wall`
  - Use wall boundary conditions on this list of faces

* - `-wall_comps`
  - An array of constrained component numbers for wall BCs

* - `-bc_slip`
  - Use weak slip boundary condition on this list of faces

* - `-bc_symmetry_x`
  - Use symmetry boundary conditions, for the x component, on this list of faces

* - `-bc_symmetry_y`
  - Use symmetry boundary conditions, for the y component, on this list of faces

* - `-bc_symmetry_z`
  - Use symmetry boundary conditions, for the z component, on this list of faces

* - `-bc_inflow`
  - Use inflow boundary conditions on this list of faces

* - `-bc_outflow`
  - Use outflow boundary conditions on this list of faces

* - `-bc_freestream`
  - Use freestream boundary conditions on this list of faces
:::

For the case of a square/cubic mesh, the list of face indices to be used with `-bc_wall`, `bc_inflow`, `bc_outflow`, `bc_freestream`  and/or `-bc_symmetry_x`, `-bc_symmetry_y`, and `-bc_symmetry_z` are:

:::{list-table} 2D Face ID Labels
:header-rows: 1
* - PETSc Face Name
  - Cartesian direction
  - Face ID

* - faceMarkerBottom
  - -z
  - 1

* - faceMarkerRight
  - +x
  - 2

* - faceMarkerTop
  - +z
  - 3

* - faceMarkerLeft
  - -x
  - 4
:::

:::{list-table} 3D Face ID Labels
:header-rows: 1
* - PETSc Face Name
  - Cartesian direction
  - Face ID

* - faceMarkerBottom
  - -z
  - 1

* - faceMarkerTop
  - +z
  - 2

* - faceMarkerFront
  - -y
  - 3

* - faceMarkerBack
  - +y
  - 4

* - faceMarkerRight
  - +x
  - 5

* - faceMarkerLeft
  - -x
  - 6

:::

## Newtonian viscosity, Ideal Gas

:::{list-table} Newtonian Ideal Gas problems Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-stab`
  - Stabilization method (`none`, `su`, or `supg`)
  - `none`
  -

* - `-Ctau_t`
  - Stabilization time constant, $C_t$
  - `1.0`
  -

* - `-Ctau_v`
  - Stabilization viscous constant, $C_v$
  - `36, 60, 128 for degree = 1, 2, 3`
  -

* - `-Ctau_C`
  - Stabilization continuity constant, $C_c$
  - `1.0`
  -

* - `-Ctau_M`
  - Stabilization momentum constant, $C_m$
  - `1.0`
  -

* - `-Ctau_E`
  - Stabilization energy constant, $C_E$
  - `1.0`
  -

* - `-div_diff_flux_projection_method`
  - Method used to calculate divergence of diffusive flux projection (`none`, `direct`, or `indirect`)
  - `none`
  -

* - `-div_diff_flux_projection_ksp*`
  - Control the KSP object for the projection of the divergence of diffusive flux
  - N/A
  -

* - `-cv`
  - Heat capacity at constant volume
  - `717`
  - `J/(kg K)`

* - `-cp`
  - Heat capacity at constant pressure
  - `1004`
  - `J/(kg K)`

* - `-gravity`
  - Gravitational acceleration vector
  - `0,0,0`
  - `m/s^2`

* - `-lambda`
  - Stokes hypothesis second viscosity coefficient
  - `-2/3`
  -

* - `-mu`
  - Shear dynamic viscosity coefficient
  - `1.8e-5`
  -  `Pa s`

* - `-k`
  - Thermal conductivity
  - `0.02638`
  - `W/(m K)`

* - `-state_var`
  - State variables to solve solution with. `conservative` ($\rho, \rho \bm{u}, \rho e$), `primitive` ($P, \bm{u}, T$), or `entropy` ($\frac{\gamma - s}{\gamma - 1} - \frac{\rho}{P} (e - c_v T),\ \frac{\rho}{P} \bm{u},\ -\frac{\rho}{P}$) where  $s = \ln(P\rho^{-\gamma})$
  - `conservative`
  - string
:::

### Newtonian Reference State Options
These options set the Newtonian reference state.

:::{list-table} Newtonian Reference State Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value

* - `-reference_pressure`
  - Reference/initial pressure. Also used as the default for `-ic_boundary_layer_wall_pressure`.
  - `1.01e5`

* - `-reference_velocity`
  - Reference/initial velocity (3 values). Also used as the default for `-ic_boundary_layer_wall_velocity`.
  - `0,0,0`

* - `-reference_temperature`
  - Reference/initial temperature. Also used as the default for `-ic_boundary_layer_wall_temperature`.
  - `288.15`
:::

(yaml-options)=
## YAML vs Command line options

Options for HONEE and PETSc can be given by either command line options or via a YAML file.
Command-line flags can be represented in a YAML file by removing the leading `-` and by allowing underscores (`_`) to be converted into a hierarchy of options (so that options with redundant prefixes can be more easily grouped together).
For example,
```
-dm_plex_shape zbox -dm_plex_box_faces 10,12,4 -dm_plex_box_bd none,none,periodic
```
could be represented as
```yaml
dm_plex:
  shape: zbox
  box:
    faces: 10,12,4
    bd: none,none,periodic
```

A YAML file of options can be passed to HONEE via `-options_file <yaml file>`.
Overriding of options are done in the order that HONEE/PETSc reads them.
Given the following YAML file:
```yaml
# ./options.yaml
option_a: 1
```
Running
```
navierstokes -option_a 2 -options_file options.yaml
```
will result in `option_a == 1`, but
```
navierstokes -option_a 2 -options_file options.yaml -option_a 3
```
will result in `option_a == 3`.

See [PETSc documentation](https://petsc.org/release/manual/other/#adding-options-from-a-file) for more advanced uses of the yaml file system.
