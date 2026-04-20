# Changelog/Release Notes

On this page we provide a summary of the main API changes, new features and examples for each release of HONEE.

(main)=

## Current `main` branch

### New features

- For strong STG, element size is calculated from cell-to-face basis. `-stg_dx` no longer needed in that case.
- Add projection of the divergence of diffusive flux for Navier-Stokes equations, see !27
- Add sine and square wave initial conditions for advection-diffusion example, see !38
- Add diffusion component of stabilization tau formulation for advection-diffusion, see !42
- Add projection of the divergence of diffusive flux for advection-diffusion equations, see !43
- Add boundary layer IC and wind type to advection-diffusion equations, see !46
- Add `-bl_height_factor` option for boundary layer IC in advection-diffusion equations, see !70
- Add `-ts_monitor_total_kinetic_energy`
- Add `-honee_check_step_interval` to verify solution validity during simulation
- Add `-honee_max_wall_time_{duration,buffer,interval}` for stopping a simulation based on expired wall clock time
- Add initial condition loading from CGNS files. Requires PETSc to be built with CGNS with parallel support (e.g. with MPI).
- Add `-ts_monitor_cfl` for CFL statistics monitoring
- Add `HONEE_StrongBCInsert` and `HONEE_StrongBCCeed` log events
- Add `-ksp_post_solve_residual` to print KSP residual summary information (can be done for `-mass` as well)
- Enable projection of the divergence of diffusive flux for explicit Navier-Stokes
- Add `-mass_ksp_view_pre_ts_solve` to view the mass KSP once before `TSSolve()` is run
- Spanwise statistics writes a logging message when a file is written, which includes post-processing information (e.g. time span for the statistics)
- Add `-ts_eval_solutions_view` to output solutions gathered via `-ts_eval_times`
- Add `-taylorgreen_background_velocity` to add a background velocity to the initial condition
- Add `HoneeGetGitVersion()` to access the Git commit and dirty state of the repository at build time.
- Add `HoneeGetVersion()` to access the semantic version of HONEE at runtime.
- Add `HoneeGetBuildConfiguration()` function for retrieving build configuration info.
- Extend `BCDefinition` to handle boundary integral setup (`BCDefinition{Get,Set}DM()`, `BCDefinition{Get,Set}Context()`, `BCDefinitionSetIFunction()`, `BCDefinitionSetIJacobian()`, and `BCDefinitionAddOperators()`)
- Add `HoneeSetContainer()`, `HoneeGetContainer()`, and `HoneeHasContainer()` for adding misc structs to a `Honee` object
- Add `-ts_monitor_spanstats_cflpe` to calculate the spanwise statistics of CFL and Peclet number
- Add `-ts_monitor_smartsim_solution` for writing solution vector to SmartSim database
- Add wall distance calculations, see documentation for details
- Add Initial condition boundary layer.
- Refactor of boundary condition prescribing options
- Add HONEE, libCEED, and PETSc version information to output and CSV files

### Internal
- Add `HoneeLoadInitialCondition()` to consolidate initial condition logic
- Move `Honee` typedef to `include/honee.h`
- Remove `SimpleBC` struct
- Convert `Honee` to a `PetscObject`

### Breaking changes
- Deprecate `-continue` and `-continue_time_filename`. Continuing from file controlled by `-continue_filename`
- Change `-checkpoint_interval` default value from `10` to `0` (CGNS is the standard restart file format now)
- Update definition of `PetscOptionsItems` to reflect new status as pointer-to-struct rather than just struct. Requires latest PETSc.
- Change `-ts_monitor_turbulence_spanstats_viewer` -> `-ts_monitor_turbulence_spanstats` and a few similar changes, see !108
- Make `-dm_plex_csr_alg mat` the default. This allows for better partition cuts when using isoperiodicity, otherwise it has practically no impact.
- Change `-mesh_transform` option to `-meshtransform_type`
- Change `-platemesh_*` options to `-meshtransform_platemesh_*`
