// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

//                        HONEE - High-Order Navier-stokes Equation Evaluator
// Sample runs:
//
//     ./navierstokes -ceed /cpu/self -options_file gaussianwave.yml
//     ./navierstokes -ceed /gpu/cuda -problem advection -degree 1
//

const char help[] =
    "HONEE - High-Order Navier-stokes Equation Evaluator\n"
    "HONEE documentation may be found at https://honee.phypid.org\n";

#include <navierstokes.h>
#include <petscdevice.h>

#include <ceed.h>
#include <petscdmplex.h>
#include <petscts.h>

int main(int argc, char **argv) {
  Honee     honee;
  MPI_Comm  comm;
  PetscEnum test_type = (PetscEnum)TESTTYPE_NONE;
  PetscBool has_help;

  PetscCall(HoneePrePetscInitialize());
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscCall(PetscOptionsGetEnum(NULL, NULL, "-test_type", TestTypes, &test_type, NULL));
  PetscCall(PetscOptionsHasHelp(NULL, &has_help));
  if (test_type == (PetscEnum)TESTTYPE_NONE && !has_help) {
    char version[256];
    PetscCall(PrintRunBanner(comm, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(HoneeGetVersionString(version, 256));
    PetscCall(PetscPrintf(comm, "%s\n", version));
    PetscCall(HoneeGetCeedVersionString(version, 256));
    PetscCall(PetscPrintf(comm, "%s\n", version));
    PetscCall(PetscGetVersion(version, 256));
    PetscCall(PetscPrintf(comm, "%s\n", version));
  }

  PetscCall(HoneeInit(comm, &honee));
  AppCtx      app_ctx  = honee->app_ctx;
  ProblemData problem  = honee->problem_data;
  Physics     phys_ctx = honee->phys;

  // ---------------------------------------------------------------------------
  // Process command line options
  // ---------------------------------------------------------------------------
  // -- Process general command line options
  PetscCall(ProcessCommandLineOptions(honee));
  PetscCall(BoundaryConditionSetUp(honee, problem, app_ctx));

  // ---------------------------------------------------------------------------
  // Initialize libCEED
  // ---------------------------------------------------------------------------
  // -- Initialize backend
  Ceed ceed;
  {  // Initialize Ceed
    PetscCheck(CeedInit(app_ctx->ceed_resource, &ceed) == CEED_ERROR_SUCCESS, comm, PETSC_ERR_LIB, "Ceed initialization failed");
    honee->ceed = ceed;
    PetscCheck(CeedSetErrorHandler(ceed, CeedErrorStore) == CEED_ERROR_SUCCESS, comm, PETSC_ERR_LIB, "Setting libCEED error handler failed");

    {  // For SYCL backend, get stream handle from PETSc
      const char *resource;
      PetscCallCeed(ceed, CeedGetResource(ceed, &resource));
      if (strstr(resource, "/gpu/sycl")) {
        PetscDeviceContext dctx;
        PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
        void *stream_handle;
        PetscCall(PetscDeviceContextGetStreamHandle(dctx, &stream_handle));
        PetscCallCeed(ceed, CeedSetStream(ceed, stream_handle));
      }
    }
  }

  DM dm;
  {  // Create DM
    VecType     vec_type = NULL;
    MatType     mat_type = NULL;
    CeedMemType mem_type_backend;

    PetscCallCeed(ceed, CeedGetPreferredMemType(ceed, &mem_type_backend));
    switch (mem_type_backend) {
      case CEED_MEM_HOST:
        vec_type = VECSTANDARD;
        break;
      case CEED_MEM_DEVICE: {
        const char *resource;
        PetscCallCeed(ceed, CeedGetResource(ceed, &resource));
        if (strstr(resource, "/gpu/cuda")) vec_type = VECCUDA;
        else if (strstr(resource, "/gpu/hip")) vec_type = VECKOKKOS;
        else if (strstr(resource, "/gpu/sycl")) vec_type = VECKOKKOS;
        else vec_type = VECSTANDARD;
      }
    }
    if (strstr(vec_type, VECCUDA)) mat_type = MATAIJCUSPARSE;
    else if (strstr(vec_type, VECKOKKOS)) mat_type = MATAIJKOKKOS;
    else mat_type = MATAIJ;

    PetscCall(CreateDM(honee, problem, mat_type, vec_type, &dm));
    honee->dm = dm;
    PetscCall(DMSetApplicationContext(dm, honee));
    PetscCall(HoneeMeshTransformFromOptions(dm));
  }

  {  // Run problem setup function
    PetscErrorCode (*p)(ProblemData, DM, void *);
    PetscCall(PetscFunctionListFind(app_ctx->problems, app_ctx->problem_name, &p));
    PetscCheck(p, PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Problem '%s' not found", app_ctx->problem_name);
    PetscCall((*p)(problem, dm, &honee));
  }

  // -- Set up DM
  PetscCall(SetUpDM(dm, problem, app_ctx->degree, app_ctx->q_extra, phys_ctx));

  // -- Refine DM for high-order viz
  if (app_ctx->viz_refine) PetscCall(VizRefineDM(dm, honee, problem, phys_ctx));

  // ---------------------------------------------------------------------------
  // Create solution vectors
  // ---------------------------------------------------------------------------
  // -- Set up global state vector Q
  Vec Q;
  PetscCall(DMCreateGlobalVector(dm, &Q));
  PetscCall(VecZeroEntries(Q));
  PetscCall(DMCreateLocalVector(dm, &honee->Q_loc));
  PetscCall(DMCreateLocalVector(dm, &honee->Q_dot_loc));
  PetscCall(VecZeroEntries(honee->Q_dot_loc));

  // ---------------------------------------------------------------------------
  // Set up libCEED
  // ---------------------------------------------------------------------------
  // -- Set up libCEED objects
  PetscCall(SetupLibceed(ceed, dm, honee, app_ctx, problem));

  for (PetscInt i = 0; i < problem->num_bc_defs; i++) {
    PetscCall(BCDefinitionViewFromOptions(problem->bc_defs[i], NULL, "-bc_definitions_view"));
  }

  // ---------------------------------------------------------------------------
  // Set up ICs
  // ---------------------------------------------------------------------------
  // -- Fix multiplicity for ICs
  PetscCall(ICs_FixMultiplicity(dm, honee, honee->Q_loc, Q, 0.0));

  // ---------------------------------------------------------------------------
  // Record boundary values from initial condition
  // ---------------------------------------------------------------------------
  // -- This overrides DMPlexInsertBoundaryValues().
  //    We use this for the main simulation DM because the reference DMPlexInsertBoundaryValues() is very slow on the GPU due to extra device-to-host
  //    communication. If we disable this, we should still get the same results due to the problem->bc function, but with potentially much slower
  //    execution.
  if (problem->set_bc_from_ics) {
    PetscCall(SetBCsFromICs(dm, Q, honee->Q_loc));
  }

  // ---------------------------------------------------------------------------
  // Gather initial Q values in case of continuation of simulation
  // ---------------------------------------------------------------------------
  // -- Set up initial values from binary file
  if (app_ctx->use_continue_file) {
    PetscCall(HoneeLoadInitialCondition(app_ctx->cont_file, &app_ctx->cont_steps, &app_ctx->cont_time, Q));
  }

  // -- Zero Q_loc
  PetscCall(VecZeroEntries(honee->Q_loc));

  // ---------------------------------------------------------------------------
  // TS: Create, setup, and solve
  // ---------------------------------------------------------------------------
  TS          ts;
  PetscScalar final_time;
  PetscCall(TSSolve_NS(dm, honee, app_ctx, phys_ctx, problem, Q, &final_time, &ts));

  // ---------------------------------------------------------------------------
  // Post-processing
  // ---------------------------------------------------------------------------
  PetscCall(PostProcess(ts, dm, problem, honee, Q, final_time));

  // Cleanup
  PetscCall(VecDestroy(&Q));
  PetscCall(DMDestroy(&dm));
  PetscCall(TSDestroy(&ts));

  if (app_ctx->test_type != TESTTYPE_NONE) {
    PetscInt  num_options_left = 0;
    PetscBool options_left     = PETSC_TRUE;

    PetscCall(PetscOptionsLeftGet(NULL, &num_options_left, NULL, NULL));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-options_left", &options_left, NULL));
    if (options_left)
      PetscCheck(num_options_left == 0, PETSC_COMM_WORLD, -1,
                 "There are unused options. This is not allowed. See error message for the unused options (or use -options_left directly)");
    PetscCheck(honee->app_ctx->has_test_run, PETSC_COMM_WORLD, -1, "HONEE was run with test mode, but no test was done.");
  }

  PetscCall(HoneeDestroy(&honee));
  return PetscFinalize();
}

// ---------- Test Cases ----------

//TESTARGS(name="Channel, ICBL") -ceed {ceed_resource} -test_type solver -options_file tests/channel_icbl_test.yaml -compare_final_state_atol 2e-11 -compare_final_state_filename tests/output/ICBL.bin
//TESTARGS(name="Blasius, Wall Distance (does not verify projection)") -ceed {ceed_resource} -test_type walldistance -options_file tests/blasius_test.yaml -compare_final_state_atol 2e-11 -compare_final_state_filename tests/output/walldistance_blasius.bin -ts_max_steps 0 -walldistance_p_values 2,3 -walldistance_projection_view binary:projection.bin
//TESTARGS(name="Channel explicit div(F_diff) verify") -ceed {ceed_resource} -test_type solver -options_file tests/channel_divdiff_verify.yaml -div_diff_flux_projection_method direct -dm_plex_box_faces 4,3,1 -ts_max_steps 5 -compare_final_state_atol 2e-11 -compare_final_state_filename tests/output/channel_div_diff_verify_indirect_explicit.bin
//TESTARGS(name="Gaussian Wave, CGNS load",only="cpu") -ceed {ceed_resource} -options_file tests/gaussianwave_cgns_load.yaml -test_type solver -ts_max_steps 7 -dm_plex_filename tests/gaussianwave_test_5.cgns -continue_filename tests/gaussianwave_test_5.cgns -dm_plex_cgns_parallel -dm_plex_box_label -dm_plex_box_label_bd none,none,periodic -compare_final_state_atol 2e-11 -compare_final_state_filename tests/output/gaussianwave-cgns-load.bin
//TESTARGS(name="Taylor-Green Vortex Total KE") -ceed {ceed_resource} -test_type solver -options_file examples/taylor_green_vortex.yaml -dm_plex_box_faces 2,2,2 -ts_max_steps 3 -ts_monitor_total_kinetic_energy_interval 2 -ts_monitor_total_kinetic_energy ascii:taylorgreen_totalke.csv:ascii_csv -honee_max_wall_time_duration 00:05 -honee_check_step_interval 1 -compare_final_state_atol 1e-10 -compare_final_state_filename tests/output/taylor-green.bin
//TESTARGS(name="Flat Plate STG",only="cpu") -ceed {ceed_resource} -options_file examples/flatplate_STG.yaml -dm_plex_box_faces 3,63,3 -dm_plex_box_upper -3.082,2.4,.00726 -ts_max_steps 1 -test_type solver -compare_final_state_atol 1e-10 -compare_final_state_filename tests/output/flatplate-STG.bin
//TESTARGS(name="Advection 2D, boundary layer IC",only="cpu") -ceed {ceed_resource} -test_type solver -options_file examples/advection_bl.yaml -dm_plex_box_faces 3,3 -ts_max_steps 0 -advection_ic_bl_height_factor 0.5 -compare_final_state_atol 1e-12 -compare_final_state_filename tests/output/adv2d-boundary-layer-ic.bin
//TESTARGS(name="Advection 2D, implicit square wave, direct div(F_diff)") -ceed {ceed_resource} -test_type solver -options_file examples/advection_wave.yaml -snes_mf -snes_fd -ts_type alpha -dm_plex_box_faces 5,5 -ts_max_steps 5 -diffusion_coeff 5e-3 -div_diff_flux_projection_method direct -ts_monitor_cfl ascii:adv2d-wave-square-direct_divdiff_cfl.csv:ascii_csv -compare_final_state_atol 1e-12 -compare_final_state_filename tests/output/adv2d-wave-square-direct_divdiff.bin
//TESTARGS(name="Advection 2D, explicit square wave, indirect div(F_diff)") -ceed {ceed_resource} -test_type solver -options_file examples/advection_wave.yaml -ts_max_steps 5 -dm_plex_box_faces 5,5 -diffusion_coeff 1e-2 -Ctau_d 2 -div_diff_flux_projection_method indirect -compare_final_state_atol 1e-12 -compare_final_state_filename tests/output/adv2d-wave-square-indirect_divdiff.bin
//TESTARGS(name="Advection 2D, sine wave IC",only="cpu") -ceed {ceed_resource} -test_type solver -options_file examples/advection_wave.yaml -ts_max_steps 0 -dm_plex_box_faces 3,3 -advection_ic_wave_type sine -compare_final_state_atol 1e-12 -compare_final_state_filename tests/output/adv2d-wave-sine.bin
//TESTARGS(name="Newtonian and Riemann Solver Unit Tests",only="cpu") -ceed {ceed_resource} -test_type solver -options_file examples/gaussianwave.yaml -compare_final_state_atol 1e100 -compare_final_state_filename tests/output/gaussianwave-IDL-entropy.bin -dm_plex_box_faces 5,5,1 -ts_max_steps 0 -newtonian_unit_tests -riemann_solver_unit_tests
//TESTARGS(name="Gaussian Wave, IDL and Entropy variables") -ceed {ceed_resource} -test_type solver -options_file examples/gaussianwave.yaml -compare_final_state_atol 2e-11 -compare_final_state_filename tests/output/gaussianwave-IDL-entropy.bin -state_var entropy -dm_plex_box_faces 5,5,1 -ts_max_steps 5 -idl_decay_time 2e-3 -idl_length 0.25 -idl_start 0 -idl_pressure 70
//TESTARGS(name="Blasius, SGS DataDriven Sequential Torch",only="torch") -ceed {ceed_resource} -options_file tests/blasius_stgtest.yaml -sgs_model_type data_driven -sgs_model_dd_leakyrelu_alpha 0.3 -sgs_model_dd_parameter_dir examples/dd_sgs_data -ts_time_step 2e-9 -state_var primitive -ksp_rtol 1e-12 -snes_rtol 1e-12 -stg_mean_only -stg_fluctuating_IC -test_type solver -compare_final_state_atol 1e-10 -compare_final_state_filename tests/output/blasius-sgs-data-driven.bin -sgs_model_dd_implementation sequential_torch -sgs_model_dd_torch_model_path ./tests/createPyTorchModel/NNModel_HIT_fp64_jit.pt
//TESTARGS(name="Blasius, SGS DataDriven Sequential Ceed") -ceed {ceed_resource} -options_file tests/blasius_stgtest.yaml -sgs_model_type data_driven -sgs_model_dd_leakyrelu_alpha 0.3 -sgs_model_dd_parameter_dir examples/dd_sgs_data -ts_time_step 2e-9 -state_var primitive -ksp_rtol 1e-12 -snes_rtol 1e-12 -stg_mean_only -stg_fluctuating_IC -test_type solver -compare_final_state_atol 1e-10 -compare_final_state_filename tests/output/blasius-sgs-data-driven.bin -sgs_model_dd_implementation sequential_ceed
//TESTARGS(name="Gaussian Wave, explicit, supg, IDL") -ceed {ceed_resource} -test_type solver -options_file examples/gaussianwave.yaml -compare_final_state_atol 1e-8 -compare_final_state_filename tests/output/gaussianwave-explicit.bin -dm_plex_box_faces 2,2,1 -ts_max_steps 5 -degree 3 -implicit false -ts_type rk -stab supg -state_var conservative -mass_ksp_type gmres -mass_pc_jacobi_type diagonal -idl_decay_time 2e-3 -idl_length 0.25 -idl_start 0 -idl_pressure 70
//TESTARGS(name="Advection 2D, rotation, explicit, supg, consistent mass") -ceed {ceed_resource} -test_type solver -options_file tests/advection_2d_test.yaml -Ctaus 0.5 -mass_ksp_type gmres -mass_pc_type vpbjacobi -compare_final_state_atol 1e-10 -compare_final_state_filename tests/output/adv2d-rotation-explicit-stab-supg-consistent-mass.bin
//TESTARGS(name="Advection, skew") -ceed {ceed_resource} -test_type solver -options_file examples/advection.yaml -ts_max_steps 5 -wind_type translation -units_kilogram 1e-9 -units_meter 1e-2 -units_second 1e-2 -wind_translation -0.5547002,0.83205029,0 -advection_ic_type skew  -dm_plex_box_faces 2,1,1 -degree 2 -stab supg -stab_tau advdiff_shakib -Ctau_a 4 -Ctau_d 0 -ksp_type gmres -diffusion_coeff 5e-4 -compare_final_state_atol 7e-10 -compare_final_state_filename tests/output/adv-skew.bin
//TESTARGS(name="Blasius, bc_slip, Indirect Diffusive Flux Projection") -ceed {ceed_resource} -test_type solver -options_file tests/blasius_diffflux.yaml -ts_max_steps 5 -dm_plex_box_faces 3,20,1 -meshtransform_platemesh_nDelta 10 -meshtransform_platemesh_growth 1.2 -compare_final_state_atol 2E-11 -compare_final_state_filename tests/output/blasius-bc_slip_indirect.bin -div_diff_flux_projection_method indirect
//TESTARGS(name="Blasius, bc_slip, Direct Diffusive Flux Projection") -ceed {ceed_resource} -test_type solver -options_file tests/blasius_diffflux.yaml -ts_max_steps 5 -dm_plex_box_faces 3,20,1 -meshtransform_platemesh_nDelta 10 -meshtransform_platemesh_growth 1.2 -compare_final_state_atol 2E-11 -compare_final_state_filename tests/output/blasius-bc_slip.bin -div_diff_flux_projection_method direct
//TESTARGS(name="Advection, rotation, cosine, direct div(F_diff)") -ceed {ceed_resource} -test_type solver -options_file examples/advection.yaml -ts_max_steps 5 -advection_ic_type cosine_hill -dm_plex_box_faces 2,2,1 -diffusion_coeff 5e-3 -units_kilogram 1e-9 -units_meter 1e-2 -units_second 1e-2 -div_diff_flux_projection_method direct -compare_final_state_atol 1e-10 -compare_final_state_filename tests/output/adv-rotation-cosine.bin
//TESTARGS(name="Advection, rotation, cylinder, explicit, CFL/Pe Statistics") -ceed {ceed_resource} -test_type spanstats -options_file examples/advection.yaml -ts_max_steps 5 -advection_ic_type cylinder -dm_plex_box_faces 2,2,1 -diffusion_coeff 5e-3 -implicit 0 -stab supg -advection_ic_bubble_continuity back_sharp -ts_monitor_spanstats_cflpe -units_kilogram 1e-9 -units_meter 1e-2 -units_second 1e-2 -compare_final_state_atol 1e-10 -compare_final_state_filename tests/output/cflpe-spanstats-stats-adv-rotation.bin
//TESTARGS(name="Gaussian Wave, using MatShell") -ceed {ceed_resource} -test_type solver -options_file examples/gaussianwave.yaml -compare_final_state_atol 1e-8 -compare_final_state_filename tests/output/gaussianwave-shell.bin -dm_plex_box_faces 2,2,1 -ts_max_steps 5 -degree 3 -amat_type shell -pc_type vpbjacobi -ts_alpha_radius 0.5
//TESTARGS(name="Taylor-Green Vortex IC") -ceed {ceed_resource} -problem taylor_green -test_type solver -dm_plex_dim 3 -dm_plex_box_faces 6,6,6 -ts_max_steps 0 -taylorgreen_background_velocity 1,1,1 -compare_final_state_atol 1e-12 -compare_final_state_filename tests/output/taylor-green-IC.bin
//TESTARGS(name="Blasius, SGS DataDriven Fused") -ceed {ceed_resource} -options_file tests/blasius_stgtest.yaml -sgs_model_type data_driven -sgs_model_dd_leakyrelu_alpha 0.3 -sgs_model_dd_parameter_dir examples/dd_sgs_data -ts_time_step 2e-9 -state_var primitive -ksp_rtol 1e-12 -snes_rtol 1e-12 -stg_mean_only -stg_fluctuating_IC -test_type solver -compare_final_state_atol 1e-10 -compare_final_state_filename tests/output/blasius-sgs-data-driven.bin
//TESTARGS(name="Blasius, Anisotropic Differential Filter") -ceed {ceed_resource} -test_type diff_filter -options_file tests/blasius_test.yaml -compare_final_state_atol 5e-10 -compare_final_state_filename tests/output/blasius_diff_filter_aniso_vandriest.bin -diff_filter_monitor -ts_max_steps 0 -state_var primitive -diff_filter_friction_length 1e-5 -diff_filter_wall_damping_function van_driest -diff_filter_ksp_rtol 1e-8 -diff_filter_grid_based_width -diff_filter_width_scaling 1,0.7,1
//TESTARGS(name="Blasius, Isotropic Differential Filter") -ceed {ceed_resource} -test_type diff_filter -options_file tests/blasius_test.yaml -compare_final_state_atol 2e-12 -compare_final_state_filename tests/output/blasius_diff_filter_iso.bin -diff_filter_monitor -ts_max_steps 0 -diff_filter_width_scaling 4.2e-5,4.2e-5,4.2e-5 -diff_filter_ksp_atol 1e-14 -diff_filter_ksp_rtol 1e-16
//TESTARGS(name="Gaussian Wave, with IDL") -ceed {ceed_resource} -test_type solver -options_file examples/gaussianwave.yaml -compare_final_state_atol 2e-11 -compare_final_state_filename tests/output/gaussianwave-IDL.bin -dm_plex_box_faces 5,5,1 -ts_max_steps 5 -idl_decay_time 2e-3 -idl_length 0.25 -idl_start 0 -ts_alpha_radius 0.5 -idl_pressure 70
//TESTARGS(name="Spanwise Turbulence Statistics") -ceed {ceed_resource} -test_type turb_spanstats -options_file tests/stats_test.yaml -ts_monitor_spanstats_turbulence -compare_final_state_atol 1E-11 -compare_final_state_filename tests/output/turb-spanstats-stats.bin
//TESTARGS(name="Spanwise CFL/Pe Statistics") -ceed {ceed_resource} -test_type spanstats -options_file tests/stats_test.yaml -ts_monitor_spanstats_cflpe -compare_final_state_atol 1E-11 -compare_final_state_filename tests/output/cflpe-spanstats-stats.bin
//TESTARGS(name="Blasius") -ceed {ceed_resource} -test_type solver -options_file tests/blasius_test.yaml -compare_final_state_atol 2E-11 -compare_final_state_filename tests/output/blasius.bin
//TESTARGS(name="Blasius, STG Inflow") -ceed {ceed_resource} -test_type solver -options_file tests/blasius_stgtest.yaml -compare_final_state_atol 2E-11 -compare_final_state_filename tests/output/blasius_STG.bin
//TESTARGS(name="Blasius, STG Inflow, Weak Temperature") -ceed {ceed_resource} -test_type solver -options_file tests/blasius_stgtest.yaml -compare_final_state_atol 1E-11 -compare_final_state_filename tests/output/blasius_STG_weakT.bin -weakT
//TESTARGS(name="Blasius, Strong STG Inflow") -ceed {ceed_resource} -test_type solver -options_file tests/blasius_stgtest.yaml -compare_final_state_atol 1E-10 -compare_final_state_filename tests/output/blasius_STG_strongBC.bin -stg_strong true -bc_inflow_natural_type consistent
//TESTARGS(name="Channel") -ceed {ceed_resource} -test_type solver -options_file examples/channel.yaml -compare_final_state_atol 2e-11 -compare_final_state_filename tests/output/channel.bin -dm_plex_box_faces 5,5,1 -ts_max_steps 5
//TESTARGS(name="Channel, Primitive") -ceed {ceed_resource} -test_type solver -options_file examples/channel.yaml -compare_final_state_atol 2e-11 -compare_final_state_filename tests/output/channel-prim.bin -dm_plex_box_faces 5,5,1 -ts_max_steps 5 -state_var primitive
//TESTARGS(name="Density Current, explicit") -ceed {ceed_resource} -test_type solver -options_file tests/densitycurrent.yaml -q_extra 2 -compare_final_state_atol 1E-11 -compare_final_state_filename tests/output/dc-explicit.bin
//TESTARGS(name="Density Current, implicit, no stabilization") -ceed {ceed_resource} -test_type solver -options_file tests/densitycurrent.yaml -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -implicit -ts_type alpha -compare_final_state_atol 5E-4 -compare_final_state_filename tests/output/dc-implicit-stab-none.bin
//TESTARGS(name="Advection, rotation, implicit, SUPG stabilization") -ceed {ceed_resource} -test_type solver -options_file tests/advection_3d_test.yaml -stab supg -bc_names boundary -bc_boundary_face_ids 1,2,3,4,5,6 -bc_boundary_essential_comps 4 -compare_final_state_atol 5E-4 -compare_final_state_filename tests/output/adv-rotation-implicit-stab-supg.bin
//TESTARGS(name="Advection, translation, implicit, SU stabilization") -ceed {ceed_resource} -test_type solver -options_file tests/advection_3d_test.yaml -wind_type translation -wind_translation .53,-1.33,-2.65 -bc_names boundary -bc_boundary_face_ids 1,2,3,4,5,6 -bc_boundary_natural_type inflow-outflow -compare_final_state_atol 5E-4 -compare_final_state_filename tests/output/adv-translation-implicit-stab-su.bin
//TESTARGS(name="Advection 2D, rotation, explicit, strong form") -ceed {ceed_resource} -test_type solver -options_file tests/advection_2d_test.yaml -stab none -strong_form 1 -compare_final_state_atol 5E-11 -compare_final_state_filename tests/output/adv2d-rotation-explicit-strong.bin
//TESTARGS(name="Advection 2D, rotation, implicit, SUPG stabilization") -ceed {ceed_resource} -test_type solver -options_file tests/advection_2d_test.yaml -CtauS .3 -implicit -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -dm_mat_preallocate_skip 0 -ts_type alpha -dm_plex_box_faces 1,1 -compare_final_state_atol 5E-4 -compare_final_state_filename tests/output/adv2d-rotation-implicit-stab-supg.bin
//TESTARGS(name="Euler, implicit") -ceed {ceed_resource} -test_type solver -options_file tests/eulervortext_test.yaml -dm_plex_box_faces 1,1,2 -implicit -dm_mat_preallocate_skip 0 -ts_type alpha -ts_time_step 1e-3 -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -compare_final_state_atol 5E-4 -compare_final_state_filename tests/output/euler-implicit.bin
//TESTARGS(name="Euler, explicit") -ceed {ceed_resource} -test_type solver -options_file tests/eulervortext_test.yaml -dm_plex_box_faces 2,2,1 -q_extra 2 -ts_time_step 1e-7 -ts_rk_type 5bs -ts_rtol 1e-10 -ts_atol 1e-10 -compare_final_state_atol 1E-7 -compare_final_state_filename tests/output/euler-explicit.bin
//TESTARGS(name="Sod Shocktube, explicit, SU stabilization, y-z-beta shock capturing") -ceed {ceed_resource} -test_type solver -options_file tests/sod_shocktube.yaml -compare_final_state_atol 1E-11 -compare_final_state_filename tests/output/shocktube-explicit-su-yzb.bin
