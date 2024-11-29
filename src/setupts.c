// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Time-stepping functions for Navier-Stokes example using PETSc

#include <ceed.h>
#include <petscdmplex.h>
#include <petscts.h>

#include <navierstokes.h>
#include "../qfunctions/newtonian_state.h"

// @brief Insert Boundary values if it's a new time
PetscErrorCode UpdateBoundaryValues(Honee honee, Vec Q_loc, PetscReal t) {
  PetscFunctionBeginUser;
  if (honee->time_bc_set != t) {
    PetscCall(DMPlexInsertBoundaryValues(honee->dm, PETSC_TRUE, Q_loc, t, NULL, NULL, NULL));
    honee->time_bc_set = t;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// RHS (Explicit time-stepper) function setup
//   This is the RHS of the ODE, given as u_t = G(t,u)
//   This function takes in a state vector Q and writes into G
PetscErrorCode RHS_NS(TS ts, PetscReal t, Vec Q, Vec G, void *user_data) {
  Honee        honee = *(Honee *)user_data;
  Ceed         ceed  = honee->ceed;
  PetscScalar  dt;
  Vec          Q_loc = honee->Q_loc;
  PetscMemType q_mem_type;

  PetscFunctionBeginUser;
  // Update time dependent data
  PetscCall(UpdateBoundaryValues(honee, Q_loc, t));
  if (honee->phys->solution_time_label)
    PetscCallCeed(ceed, CeedOperatorSetContextDouble(honee->op_rhs_ctx->op, honee->phys->solution_time_label, &t));
  PetscCall(TSGetTimeStep(ts, &dt));
  if (honee->phys->timestep_size_label)
    PetscCallCeed(ceed, CeedOperatorSetContextDouble(honee->op_rhs_ctx->op, honee->phys->timestep_size_label, &dt));

  PetscCall(DMGlobalToLocal(honee->dm, Q, INSERT_VALUES, Q_loc));
  if (honee->app_ctx->divFdiffproj_method != DIV_DIFF_FLUX_PROJ_NONE) PetscCall(DivDiffFluxProjectionApply(honee->diff_flux_proj, Q_loc));
  PetscCall(ApplyCeedOperatorLocalToGlobal(Q_loc, G, honee->op_rhs_ctx));

  PetscCall(VecReadPetscToCeed(Q_loc, &q_mem_type, honee->q_ceed));

  // Inverse of the mass matrix
  PetscCall(KSPSolve(honee->mass_ksp, G, G));

  PetscCall(VecReadCeedToPetsc(honee->q_ceed, q_mem_type, Q_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Surface forces function setup
static PetscErrorCode Surface_Forces_NS(DM dm, Vec G_loc, PetscInt num_walls, const PetscInt walls[], PetscScalar *reaction_force) {
  DMLabel            face_label;
  const PetscScalar *g_array;
  PetscInt           dim  = 3;
  MPI_Comm           comm = PetscObjectComm((PetscObject)dm);
  PetscSection       section;

  PetscFunctionBeginUser;
  PetscCall(DMGetLabel(dm, "Face Sets", &face_label));
  PetscCall(VecGetArrayRead(G_loc, &g_array));
  for (PetscInt w = 0; w < num_walls; w++) {
    const PetscInt wall = walls[w], *points;
    IS             wall_is;
    PetscInt       num_points, num_comp = 0;

    PetscCall(DMLabelGetStratumIS(face_label, wall, &wall_is));
    if (!wall_is) continue;  // No wall points on this process, skip

    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(PetscSectionGetFieldComponents(section, 0, &num_comp));
    PetscCall(ISGetSize(wall_is, &num_points));
    PetscCall(ISGetIndices(wall_is, &points));
    for (PetscInt i = 0; i < num_points; i++) {
      const PetscInt           p = points[i];
      const StateConservative *r;
      PetscInt                 dof;

      PetscCall(DMPlexPointLocalRead(dm, p, g_array, &r));
      PetscCall(PetscSectionGetDof(section, p, &dof));
      for (PetscInt node = 0; node < dof / num_comp; node++) {
        for (PetscInt j = 0; j < dim; j++) {
          reaction_force[w * dim + j] -= r[node].momentum[j];
        }
      }
    }
    PetscCall(ISRestoreIndices(wall_is, &points));
    PetscCall(ISDestroy(&wall_is));
  }
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, reaction_force, dim * num_walls, MPIU_SCALAR, MPI_SUM, comm));
  PetscCall(VecRestoreArrayRead(G_loc, &g_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Implicit time-stepper function setup
PetscErrorCode IFunction_NS(TS ts, PetscReal t, Vec Q, Vec Q_dot, Vec G, void *user_data) {
  Honee        honee = *(Honee *)user_data;
  Ceed         ceed  = honee->ceed;
  PetscScalar  dt;
  Vec          Q_loc = honee->Q_loc, Q_dot_loc = honee->Q_dot_loc, G_loc;
  PetscMemType q_mem_type, q_dot_mem_type, g_mem_type;

  PetscFunctionBeginUser;
  // Get local vectors
  PetscCall(DMGetNamedLocalVector(honee->dm, "ResidualLocal", &G_loc));

  // Update time dependent data
  PetscCall(UpdateBoundaryValues(honee, Q_loc, t));
  if (honee->phys->solution_time_label) PetscCallCeed(ceed, CeedOperatorSetContextDouble(honee->op_ifunction, honee->phys->solution_time_label, &t));
  PetscCall(TSGetTimeStep(ts, &dt));
  if (honee->phys->timestep_size_label) PetscCallCeed(ceed, CeedOperatorSetContextDouble(honee->op_ifunction, honee->phys->timestep_size_label, &dt));

  // Global-to-local
  PetscCall(DMGlobalToLocalBegin(honee->dm, Q, INSERT_VALUES, Q_loc));
  PetscCall(DMGlobalToLocalBegin(honee->dm, Q_dot, INSERT_VALUES, Q_dot_loc));
  PetscCall(DMGlobalToLocalEnd(honee->dm, Q, INSERT_VALUES, Q_loc));
  if (honee->app_ctx->divFdiffproj_method != DIV_DIFF_FLUX_PROJ_NONE) PetscCall(DivDiffFluxProjectionApply(honee->diff_flux_proj, Q_loc));
  PetscCall(DMGlobalToLocalEnd(honee->dm, Q_dot, INSERT_VALUES, Q_dot_loc));

  // Place PETSc vectors in CEED vectors
  PetscCall(VecReadPetscToCeed(Q_loc, &q_mem_type, honee->q_ceed));
  PetscCall(VecReadPetscToCeed(Q_dot_loc, &q_dot_mem_type, honee->q_dot_ceed));
  PetscCall(VecPetscToCeed(G_loc, &g_mem_type, honee->g_ceed));

  // Apply CEED operator
  PetscCall(PetscLogEventBegin(FLUIDS_CeedOperatorApply, Q, G, 0, 0));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCeed(honee->ceed, CeedOperatorApply(honee->op_ifunction, honee->q_ceed, honee->g_ceed, CEED_REQUEST_IMMEDIATE));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogEventEnd(FLUIDS_CeedOperatorApply, Q, G, 0, 0));

  // Restore vectors
  PetscCall(VecReadCeedToPetsc(honee->q_ceed, q_mem_type, Q_loc));
  PetscCall(VecReadCeedToPetsc(honee->q_dot_ceed, q_dot_mem_type, Q_dot_loc));
  PetscCall(VecCeedToPetsc(honee->g_ceed, g_mem_type, G_loc));

  if (honee->app_ctx->sgs_model_type == SGS_MODEL_DATA_DRIVEN) {
    PetscCall(SgsDDApplyIFunction(honee, Q_loc, G_loc));
  }

  // Local-to-Global
  PetscCall(VecZeroEntries(G));
  PetscCall(DMLocalToGlobal(honee->dm, G_loc, ADD_VALUES, G));

  // Restore vectors
  PetscCall(DMRestoreNamedLocalVector(honee->dm, "ResidualLocal", &G_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormIJacobian_NS(TS ts, PetscReal t, Vec Q, Vec Q_dot, PetscReal shift, Mat J, Mat J_pre, void *user_data) {
  Honee     honee = *(Honee *)user_data;
  PetscBool J_is_matceed, J_is_mffd, J_pre_is_matceed, J_pre_is_mffd;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectTypeCompare((PetscObject)J, MATMFFD, &J_is_mffd));
  PetscCall(PetscObjectTypeCompare((PetscObject)J, MATCEED, &J_is_matceed));
  PetscCall(PetscObjectTypeCompare((PetscObject)J_pre, MATMFFD, &J_pre_is_mffd));
  PetscCall(PetscObjectTypeCompare((PetscObject)J_pre, MATCEED, &J_pre_is_matceed));

  PetscCall(MatCeedSetContextReal(honee->mat_ijacobian, "ijacobian time shift", shift));

  if (J_is_matceed || J_is_mffd) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  } else PetscCall(MatCeedAssembleCOO(honee->mat_ijacobian, J));

  if (J_pre_is_matceed && J != J_pre) {
    PetscCall(MatAssemblyBegin(J_pre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J_pre, MAT_FINAL_ASSEMBLY));
  } else if (!J_pre_is_matceed && !J_pre_is_mffd && J != J_pre) {
    PetscCall(MatCeedAssembleCOO(honee->mat_ijacobian, J_pre));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode WriteOutput(Honee honee, Vec Q, PetscInt step_no, PetscScalar time) {
  Vec         Q_loc;
  char        file_path[PETSC_MAX_PATH_LEN];
  PetscViewer viewer;

  PetscFunctionBeginUser;
  if (honee->app_ctx->checkpoint_vtk) {
    // Set up output
    PetscCall(DMGetLocalVector(honee->dm, &Q_loc));
    PetscCall(PetscObjectSetName((PetscObject)Q_loc, "StateVec"));
    PetscCall(VecZeroEntries(Q_loc));
    PetscCall(DMGlobalToLocal(honee->dm, Q, INSERT_VALUES, Q_loc));

    // Output
    PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/ns-%03" PetscInt_FMT ".vtu", honee->app_ctx->output_dir, step_no));

    PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)Q), file_path, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(Q_loc, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    if (honee->dm_viz) {
      Vec         Q_refined, Q_refined_loc;
      char        file_path_refined[PETSC_MAX_PATH_LEN];
      PetscViewer viewer_refined;

      PetscCall(DMGetGlobalVector(honee->dm_viz, &Q_refined));
      PetscCall(DMGetLocalVector(honee->dm_viz, &Q_refined_loc));
      PetscCall(PetscObjectSetName((PetscObject)Q_refined_loc, "Refined"));

      PetscCall(MatInterpolate(honee->interp_viz, Q, Q_refined));
      PetscCall(VecZeroEntries(Q_refined_loc));
      PetscCall(DMGlobalToLocal(honee->dm_viz, Q_refined, INSERT_VALUES, Q_refined_loc));

      PetscCall(
          PetscSNPrintf(file_path_refined, sizeof file_path_refined, "%s/nsrefined-%03" PetscInt_FMT ".vtu", honee->app_ctx->output_dir, step_no));

      PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)Q_refined), file_path_refined, FILE_MODE_WRITE, &viewer_refined));
      PetscCall(VecView(Q_refined_loc, viewer_refined));
      PetscCall(DMRestoreLocalVector(honee->dm_viz, &Q_refined_loc));
      PetscCall(DMRestoreGlobalVector(honee->dm_viz, &Q_refined));
      PetscCall(PetscViewerDestroy(&viewer_refined));
    }
    PetscCall(DMRestoreLocalVector(honee->dm, &Q_loc));
  }

  // Save data in a binary file for continuation of simulations
  if (honee->app_ctx->add_stepnum2bin) {
    PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/ns-solution-%" PetscInt_FMT ".bin", honee->app_ctx->output_dir, step_no));
  } else {
    PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/ns-solution.bin", honee->app_ctx->output_dir));
  }
  PetscCall(PetscViewerBinaryOpen(honee->comm, file_path, FILE_MODE_WRITE, &viewer));

  time /= honee->units->second;  // Dimensionalize time back
  PetscCall(HoneeWriteBinaryVec(viewer, Q, time, step_no));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// CSV Monitor
PetscErrorCode TSMonitor_WallForce(TS ts, PetscInt step_no, PetscReal time, Vec Q, void *ctx) {
  Honee             honee = ctx;
  Vec               G_loc;
  PetscInt          num_wall = honee->app_ctx->wall_forces.num_wall, dim = 3;
  const PetscInt   *walls  = honee->app_ctx->wall_forces.walls;
  PetscViewer       viewer = honee->app_ctx->wall_forces.viewer;
  PetscViewerFormat format = honee->app_ctx->wall_forces.viewer_format;
  PetscScalar      *reaction_force;
  PetscBool         is_ascii;

  PetscFunctionBeginUser;
  if (!viewer) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetNamedLocalVector(honee->dm, "ResidualLocal", &G_loc));
  PetscCall(PetscCalloc1(num_wall * dim, &reaction_force));
  PetscCall(Surface_Forces_NS(honee->dm, G_loc, num_wall, walls, reaction_force));
  PetscCall(DMRestoreNamedLocalVector(honee->dm, "ResidualLocal", &G_loc));

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &is_ascii));

  if (is_ascii) {
    if (format == PETSC_VIEWER_ASCII_CSV && !honee->app_ctx->wall_forces.header_written) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Step,Time,Wall,ForceX,ForceY,ForceZ\n"));
      honee->app_ctx->wall_forces.header_written = PETSC_TRUE;
    }
    for (PetscInt w = 0; w < num_wall; w++) {
      PetscInt wall = walls[w];
      if (format == PETSC_VIEWER_ASCII_CSV) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT ",%g,%" PetscInt_FMT ",%g,%g,%g\n", step_no, time, wall,
                                         reaction_force[w * dim + 0], reaction_force[w * dim + 1], reaction_force[w * dim + 2]));

      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "Wall %" PetscInt_FMT " Forces: Force_x = %12g, Force_y = %12g, Force_z = %12g\n", wall,
                                         reaction_force[w * dim + 0], reaction_force[w * dim + 1], reaction_force[w * dim + 2]));
      }
    }
  }
  PetscCall(PetscFree(reaction_force));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// User provided TS Monitor
PetscErrorCode TSMonitor_NS(TS ts, PetscInt step_no, PetscReal time, Vec Q, void *ctx) {
  Honee honee = ctx;

  PetscFunctionBeginUser;
  // Print every 'checkpoint_interval' steps
  if (honee->app_ctx->checkpoint_interval <= 0 || step_no % honee->app_ctx->checkpoint_interval != 0 ||
      (honee->app_ctx->cont_steps == step_no && step_no != 0)) {
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(WriteOutput(honee, Q, step_no, time));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSPostStep_CheckStep(TS ts) {
  Honee     honee;
  PetscReal norm;
  PetscInt  step;
  Vec       Q;

  PetscFunctionBeginUser;
  PetscCall(TSGetApplicationContext(ts, &honee));
  PetscCall(TSGetStepNumber(ts, &step));
  PetscCall(TSGetSolution(ts, &Q));
  if (step % honee->app_ctx->check_step_interval) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecNorm(Q, NORM_1, &norm));
  if (PetscIsNanReal(norm)) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts), "Solution diverged: Nans found in solution\n"));
    PetscCall(TSSetConvergedReason(ts, TS_DIVERGED_NONLINEAR_SOLVE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TS: Create, setup, and solve
PetscErrorCode TSSolve_NS(DM dm, Honee honee, AppCtx app_ctx, Physics phys, ProblemData problem, Vec *Q, PetscScalar *f_time, TS *ts) {
  MPI_Comm    comm = honee->comm;
  TSAdapt     adapt;
  PetscScalar final_time;

  PetscFunctionBeginUser;
  PetscCall(TSCreate(comm, ts));
  PetscCall(TSSetDM(*ts, dm));
  PetscCall(TSSetApplicationContext(*ts, honee));
  if (phys->implicit) {
    PetscCall(TSSetType(*ts, TSBDF));
    if (honee->op_ifunction) {
      PetscCall(TSSetIFunction(*ts, NULL, IFunction_NS, &honee));
    } else {  // Implicit integrators can fall back to using an RHSFunction
      PetscCall(TSSetRHSFunction(*ts, NULL, RHS_NS, &honee));
    }
    if (honee->mat_ijacobian) {
      PetscCall(DMTSSetIJacobian(dm, FormIJacobian_NS, &honee));
    }
  } else {
    PetscCheck(honee->op_rhs_ctx, comm, PETSC_ERR_ARG_NULL, "Problem does not provide RHSFunction");
    PetscCall(TSSetType(*ts, TSRK));
    PetscCall(TSRKSetType(*ts, TSRK5F));
    PetscCall(TSSetRHSFunction(*ts, NULL, RHS_NS, &honee));
  }
  PetscCall(TSSetMaxTime(*ts, 500. * honee->units->second));
  PetscCall(TSSetExactFinalTime(*ts, TS_EXACTFINALTIME_STEPOVER));
  if (app_ctx->test_type == TESTTYPE_NONE) PetscCall(TSSetErrorIfStepFails(*ts, PETSC_FALSE));
  PetscCall(TSSetTimeStep(*ts, 1.e-2 * honee->units->second));
  PetscCall(TSGetAdapt(*ts, &adapt));
  PetscCall(TSAdaptSetStepLimits(adapt, 1.e-12 * honee->units->second, 1.e2 * honee->units->second));
  PetscCall(TSSetFromOptions(*ts));
  if (honee->mat_ijacobian) {
    if (app_ctx->amat_type && !strcmp(app_ctx->amat_type, MATSHELL)) {
      SNES snes;
      KSP  ksp;
      Mat  Pmat, Amat;

      PetscCall(TSGetSNES(*ts, &snes));
      PetscCall(SNESGetKSP(snes, &ksp));
      PetscCall(CreateSolveOperatorsFromMatCeed(ksp, honee->mat_ijacobian, PETSC_FALSE, &Amat, &Pmat));
      PetscCall(TSSetIJacobian(*ts, honee->mat_ijacobian, Pmat, NULL, NULL));
      PetscCall(MatDestroy(&Amat));
      PetscCall(MatDestroy(&Pmat));
    }
  }
  honee->time_bc_set = -1.0;  // require all BCs be updated
  if (app_ctx->cont_steps) {  // continue from previous timestep data
    PetscCall(TSSetTime(*ts, app_ctx->cont_time * honee->units->second));
    PetscCall(TSSetStepNumber(*ts, app_ctx->cont_steps));
  }
  if (app_ctx->test_type == TESTTYPE_NONE) {
    PetscCall(TSMonitorSet(*ts, TSMonitor_NS, honee, NULL));
  }
  if (app_ctx->wall_forces.viewer) {
    PetscCall(TSMonitorSet(*ts, TSMonitor_WallForce, honee, NULL));
  }
  if (app_ctx->turb_spanstats_enable) {
    PetscCall(TSMonitorSet(*ts, TSMonitor_TurbulenceStatistics, honee, NULL));
    CeedScalar previous_time = app_ctx->cont_time * honee->units->second;
    PetscCallCeed(honee->ceed,
                  CeedOperatorSetContextDouble(honee->spanstats.op_stats_collect_ctx->op, honee->spanstats.previous_time_label, &previous_time));
  }
  PetscCall(TSMonitorSetFromOptions(*ts, "-ts_monitor_total_kinetic_energy", "Monitor total kinetic energy balance terms in the domain", NULL,
                                    TSMonitor_TotalKineticEnergy, SetupMontiorTotalKineticEnergy));
  if (app_ctx->check_step_interval > 0) PetscCall(TSSetPostStep(*ts, TSPostStep_CheckStep));
  if (app_ctx->diff_filter_monitor) PetscCall(TSMonitorSet(*ts, TSMonitor_DifferentialFilter, honee, NULL));

  if (app_ctx->sgs_train_enable) {
    PetscCall(TSMonitorSet(*ts, TSMonitor_SGS_DD_Training, honee, NULL));
    PetscCall(TSSetPostStep(*ts, TSPostStep_SGS_DD_Training));
  }

  if (app_ctx->test_type == TESTTYPE_NONE) PetscCall(PrintRunInfo(honee, honee->phys, problem, *ts));
  // Solve
  PetscReal start_time;
  PetscInt  start_step;
  PetscCall(TSGetTime(*ts, &start_time));
  PetscCall(TSGetStepNumber(*ts, &start_step));

  PetscCall(PetscLogDefaultBegin());  // So we can use PetscLogStageGetPerfInfo without -log_view
  PetscPreLoadBegin(PETSC_FALSE, "Fluids Solve");
  PetscCall(TSSetTime(*ts, start_time));
  PetscCall(TSSetStepNumber(*ts, start_step));
  if (PetscPreLoadingOn) {
    // LCOV_EXCL_START
    SNES      snes;
    Vec       Q_preload;
    PetscReal rtol;
    PetscCall(VecDuplicate(*Q, &Q_preload));
    PetscCall(VecCopy(*Q, Q_preload));
    PetscCall(TSGetSNES(*ts, &snes));
    PetscCall(SNESGetTolerances(snes, NULL, &rtol, NULL, NULL, NULL));
    PetscCall(SNESSetTolerances(snes, PETSC_DEFAULT, .99, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(TSSetSolution(*ts, Q_preload));
    PetscCall(TSStep(*ts));
    PetscCall(SNESSetTolerances(snes, PETSC_DEFAULT, rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(VecDestroy(&Q_preload));
    // LCOV_EXCL_STOP
  } else {
    PetscCall(PetscBarrier((PetscObject)*ts));
    PetscCall(TSSolve(*ts, *Q));
  }
  PetscPreLoadEnd();

  PetscCall(TSGetSolveTime(*ts, &final_time));
  *f_time = final_time;

  if (app_ctx->test_type == TESTTYPE_NONE) {
    PetscInt step_no;
    PetscCall(TSGetStepNumber(*ts, &step_no));
    if (honee->app_ctx->checkpoint_interval > 0 || honee->app_ctx->checkpoint_interval == -1) {
      PetscCall(WriteOutput(honee, *Q, step_no, final_time));
    }

    PetscLogStage      stage_id;
    PetscEventPerfInfo stage_perf;

    PetscCall(PetscLogStageGetId("Fluids Solve", &stage_id));
    PetscCall(PetscLogStageGetPerfInfo(stage_id, &stage_perf));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Time taken for solution (sec): %g\n", stage_perf.time));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
