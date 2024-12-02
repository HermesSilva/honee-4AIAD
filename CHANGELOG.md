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

### Internal
- Add `HoneeLoadInitialCondition()` to consolidate initial condition logic

### Breaking changes
- Deprecate `-continue` and `-continue_time_filename`. Continuing from file controlled by `-continue_filename`
