// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Utility functions for setting up EULER_VORTEX

#include "../qfunctions/eulervortex.h"

#include <ceed.h>
#include <petscdm.h>

#include <navierstokes.h>

PetscErrorCode NS_EULER_VORTEX(ProblemData problem, DM dm, void *ctx, SimpleBC bc) {
  EulerTestType        euler_test;
  Honee                honee = *(Honee *)ctx;
  StabilizationType    stab;
  MPI_Comm             comm = honee->comm;
  Ceed                 ceed = honee->ceed;
  PetscBool            implicit;
  EulerContext         euler_ctx;
  CeedQFunctionContext euler_qfctx;
  PetscInt             dim;

  PetscFunctionBeginUser;
  PetscCall(PetscCalloc1(1, &euler_ctx));

  // ------------------------------------------------------
  //               SET UP DENSITY_CURRENT
  // ------------------------------------------------------
  problem->ics.qf_func_ptr                 = ICsEuler;
  problem->ics.qf_loc                      = ICsEuler_loc;
  problem->apply_vol_rhs.qf_func_ptr       = Euler;
  problem->apply_vol_rhs.qf_loc            = Euler_loc;
  problem->apply_vol_ifunction.qf_func_ptr = IFunction_Euler;
  problem->apply_vol_ifunction.qf_loc      = IFunction_Euler_loc;
  problem->apply_inflow.qf_func_ptr        = TravelingVortex_Inflow;
  problem->apply_inflow.qf_loc             = TravelingVortex_Inflow_loc;
  problem->apply_outflow.qf_func_ptr       = Euler_Outflow;
  problem->apply_outflow.qf_loc            = Euler_Outflow_loc;
  problem->jac_data_size_vol               = 0;
  problem->jac_data_size_sur               = 0;
  problem->compute_exact_solution_error    = PETSC_TRUE;
  problem->print_info                      = PRINT_EULER_VORTEX;

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  CeedScalar vortex_strength = 5.;   // -
  CeedScalar c_tau           = 0.5;  // -
  // c_tau = 0.5 is reported as "optimal" in Hughes et al 2010
  PetscReal center[3]  = {0.},         // m
      mean_velocity[3] = {1., 1., 0};  // m/s
  PetscReal domain_min[3], domain_max[3], domain_size[3] = {0.};
  PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
  PetscCall(DMGetDimension(dm, &dim));
  for (PetscInt i = 0; i < dim; i++) domain_size[i] = domain_max[i] - domain_min[i];

  // ------------------------------------------------------
  //             Create the PETSc context
  // ------------------------------------------------------
  PetscScalar meter  = 1e-2;  // 1 meter in scaled length units
  PetscScalar second = 1e-2;  // 1 second in scaled time units

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  PetscOptionsBegin(comm, NULL, "Options for EULER_VORTEX problem", NULL);
  // -- Physics
  PetscCall(PetscOptionsScalar("-vortex_strength", "Strength of Vortex", NULL, vortex_strength, &vortex_strength, NULL));
  PetscInt  n = dim;
  PetscBool user_velocity;
  PetscCall(PetscOptionsRealArray("-mean_velocity", "Background velocity vector", NULL, mean_velocity, &n, &user_velocity));
  for (PetscInt i = 0; i < dim; i++) center[i] = .5 * domain_size[i];
  n = dim;
  PetscCall(PetscOptionsRealArray("-center", "Location of vortex center", NULL, center, &n, NULL));
  PetscCall(PetscOptionsBool("-implicit", "Use implicit (IFunction) formulation", NULL, implicit = PETSC_FALSE, &implicit, NULL));
  PetscCall(PetscOptionsEnum("-euler_test", "Euler test option", NULL, EulerTestTypes, (PetscEnum)(euler_test = EULER_TEST_ISENTROPIC_VORTEX),
                             (PetscEnum *)&euler_test, NULL));
  PetscCall(PetscOptionsEnum("-stab", "Stabilization method", NULL, StabilizationTypes, (PetscEnum)(stab = STAB_NONE), (PetscEnum *)&stab, NULL));
  PetscCall(PetscOptionsScalar("-c_tau", "Stabilization constant", NULL, c_tau, &c_tau, NULL));
  // -- Units
  PetscCall(PetscOptionsScalar("-units_meter", "1 meter in scaled length units", NULL, meter, &meter, NULL));
  meter = fabs(meter);
  PetscCall(PetscOptionsScalar("-units_second", "1 second in scaled time units", NULL, second, &second, NULL));
  second = fabs(second);

  // -- Warnings
  if (stab == STAB_SUPG && !implicit) {
    PetscCall(PetscPrintf(comm, "Warning! Use -stab supg only with -implicit\n"));
  }
  if (user_velocity && (euler_test == EULER_TEST_1 || euler_test == EULER_TEST_3)) {
    PetscCall(PetscPrintf(comm, "Warning! Background velocity vector for -euler_test t1 and -euler_test t3 is (0,0,0)\n"));
  }

  PetscOptionsEnd();

  // ------------------------------------------------------
  //           Set up the PETSc context
  // ------------------------------------------------------
  honee->units->meter  = meter;
  honee->units->second = second;

  // ------------------------------------------------------
  //           Set up the libCEED context
  // ------------------------------------------------------
  // -- Scale variables to desired units
  for (PetscInt i = 0; i < 3; i++) {
    center[i] *= meter;
    domain_size[i] *= meter;
    mean_velocity[i] *= (meter / second);
  }

  // -- QFunction Context
  honee->phys->implicit       = implicit;
  euler_ctx->curr_time        = 0.;
  euler_ctx->implicit         = implicit;
  euler_ctx->euler_test       = euler_test;
  euler_ctx->center[0]        = center[0];
  euler_ctx->center[1]        = center[1];
  euler_ctx->center[2]        = center[2];
  euler_ctx->vortex_strength  = vortex_strength;
  euler_ctx->c_tau            = c_tau;
  euler_ctx->mean_velocity[0] = mean_velocity[0];
  euler_ctx->mean_velocity[1] = mean_velocity[1];
  euler_ctx->mean_velocity[2] = mean_velocity[2];
  euler_ctx->stabilization    = stab;

  PetscCallCeed(ceed, CeedQFunctionContextCreate(honee->ceed, &euler_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(euler_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*euler_ctx), euler_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(euler_qfctx, CEED_MEM_HOST, FreeContextPetsc));
  PetscCallCeed(ceed, CeedQFunctionContextRegisterDouble(euler_qfctx, "solution time", offsetof(struct EulerContext_, curr_time), 1,
                                                         "Physical time of the solution"));
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(euler_qfctx, &problem->ics.qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(euler_qfctx, &problem->apply_vol_rhs.qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(euler_qfctx, &problem->apply_vol_ifunction.qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(euler_qfctx, &problem->apply_inflow.qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(euler_qfctx, &problem->apply_outflow.qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&euler_qfctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PRINT_EULER_VORTEX(Honee honee, ProblemData problem, AppCtx app_ctx) {
  MPI_Comm     comm = honee->comm;
  Ceed         ceed = honee->ceed;
  EulerContext euler_ctx;

  PetscFunctionBeginUser;
  PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->ics.qfctx, CEED_MEM_HOST, &euler_ctx));
  PetscCall(PetscPrintf(comm,
                        "  Problem:\n"
                        "    Problem Name                       : %s\n"
                        "    Test Case                          : %s\n"
                        "    Background Velocity                : %f,%f,%f\n"
                        "    Stabilization                      : %s\n",
                        app_ctx->problem_name, EulerTestTypes[euler_ctx->euler_test], euler_ctx->mean_velocity[0], euler_ctx->mean_velocity[1],
                        euler_ctx->mean_velocity[2], StabilizationTypes[euler_ctx->stabilization]));

  PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->ics.qfctx, &euler_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}
