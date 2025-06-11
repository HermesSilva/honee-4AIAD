// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Utility functions for setting up SHOCKTUBE

#include "../qfunctions/shocktube.h"

#include <ceed.h>
#include <petscdm.h>

#include <navierstokes.h>

PetscErrorCode NS_SHOCKTUBE(ProblemData problem, DM dm, void *ctx) {
  SetupContextShock    setup_context;
  Honee                honee = *(Honee *)ctx;
  MPI_Comm             comm  = honee->comm;
  Ceed                 ceed  = honee->ceed;
  PetscBool            implicit;
  PetscBool            yzb;
  PetscInt             stab;
  ShockTubeContext     shocktube_ctx;
  CeedQFunctionContext shocktube_qfctx;

  PetscFunctionBeginUser;
  PetscCall(PetscCalloc1(1, &setup_context));
  PetscCall(PetscCalloc1(1, &shocktube_ctx));

  // ------------------------------------------------------
  //               SET UP SHOCKTUBE
  // ------------------------------------------------------
  problem->ics.qf_func_ptr                 = ICsShockTube;
  problem->ics.qf_loc                      = ICsShockTube_loc;
  problem->apply_vol_rhs.qf_func_ptr       = EulerShockTube;
  problem->apply_vol_rhs.qf_loc            = EulerShockTube_loc;
  problem->apply_vol_ifunction.qf_func_ptr = NULL;
  problem->apply_vol_ifunction.qf_loc      = NULL;
  problem->num_comps_jac_data              = 0;
  problem->compute_exact_solution_error    = PETSC_FALSE;
  problem->print_info                      = PRINT_SHOCKTUBE;

  // ------------------------------------------------------
  //             Create the QFunction context
  // ------------------------------------------------------
  // Driver section initial conditions
  CeedScalar P_high   = 1.0;  // Pa
  CeedScalar rho_high = 1.0;  // kg/m^3
  // Driven section initial conditions
  CeedScalar P_low   = 0.1;    // Pa
  CeedScalar rho_low = 0.125;  // kg/m^3
  // Stabilization parameter
  CeedScalar c_tau = 0.5;  // -, based on Hughes et al (2010)
  // Tuning parameters for the YZB shock capturing
  CeedScalar Cyzb = 0.1;  // -, used in approximation of (Na),x
  CeedScalar Byzb = 2.0;  // -, 1 for smooth shocks
  //                                          2 for sharp shocks
  PetscReal domain_min[3], domain_max[3], domain_size[3];
  PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
  for (PetscInt i = 0; i < 3; i++) domain_size[i] = domain_max[i] - domain_min[i];

  // ------------------------------------------------------
  //             Create the PETSc context
  // ------------------------------------------------------
  PetscScalar meter  = 1e-2;  // 1 meter in scaled length units
  PetscScalar second = 1e-2;  // 1 second in scaled time units

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  PetscOptionsBegin(comm, NULL, "Options for SHOCKTUBE problem", NULL);

  // -- Numerical formulation options
  PetscCall(PetscOptionsBool("-implicit", "Use implicit (IFunction) formulation", NULL, implicit = PETSC_FALSE, &implicit, NULL));
  PetscCall(PetscOptionsEnum("-stab", "Stabilization method", NULL, StabilizationTypes, (PetscEnum)(stab = STAB_NONE), (PetscEnum *)&stab, NULL));
  PetscCall(PetscOptionsScalar("-c_tau", "Stabilization constant", NULL, c_tau, &c_tau, NULL));
  PetscCall(PetscOptionsBool("-yzb", "Use YZB discontinuity capturing", NULL, yzb = PETSC_FALSE, &yzb, NULL));

  // -- Units
  PetscCall(PetscOptionsScalar("-units_meter", "1 meter in scaled length units", NULL, meter, &meter, NULL));
  meter = fabs(meter);
  PetscCall(PetscOptionsScalar("-units_second", "1 second in scaled time units", NULL, second, &second, NULL));
  second = fabs(second);

  // -- Warnings
  if (stab == STAB_SUPG) {
    PetscCall(PetscPrintf(comm, "Warning! -stab supg not implemented for the shocktube problem. \n"));
  }
  if (yzb && implicit) {
    PetscCall(PetscPrintf(comm, "Warning! -yzb only implemented for explicit timestepping. \n"));
  }

  PetscOptionsEnd();

  // ------------------------------------------------------
  //           Set up the PETSc context
  // ------------------------------------------------------
  honee->units->meter  = meter;
  honee->units->second = second;

  // ------------------------------------------------------
  //           Set up the QFunction context
  // ------------------------------------------------------
  // -- Scale variables to desired units
  for (PetscInt i = 0; i < 3; i++) {
    domain_size[i] *= meter;
    domain_min[i] *= meter;
  }
  CeedScalar mid_point = 0.5 * (domain_size[0] + domain_min[0]);

  // -- Setup Context
  setup_context->mid_point = mid_point;
  setup_context->time      = 0.0;
  setup_context->P_high    = P_high;
  setup_context->rho_high  = rho_high;
  setup_context->P_low     = P_low;
  setup_context->rho_low   = rho_low;

  // -- QFunction Context
  honee->phys->implicit        = implicit;
  shocktube_ctx->implicit      = implicit;
  shocktube_ctx->stabilization = stab;
  shocktube_ctx->yzb           = yzb;
  shocktube_ctx->Cyzb          = Cyzb;
  shocktube_ctx->Byzb          = Byzb;
  shocktube_ctx->c_tau         = c_tau;

  PetscCallCeed(ceed, CeedQFunctionContextCreate(honee->ceed, &problem->ics.qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(problem->ics.qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*setup_context), setup_context));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(problem->ics.qfctx, CEED_MEM_HOST, FreeContextPetsc));

  PetscCallCeed(ceed, CeedQFunctionContextCreate(honee->ceed, &shocktube_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(shocktube_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*shocktube_ctx), shocktube_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(shocktube_qfctx, CEED_MEM_HOST, FreeContextPetsc));
  problem->apply_vol_rhs.qfctx = shocktube_qfctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PRINT_SHOCKTUBE(Honee honee, ProblemData problem, AppCtx app_ctx) {
  MPI_Comm comm = honee->comm;

  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(comm,
                        "  Problem:\n"
                        "    Problem Name                       : %s\n",
                        app_ctx->problem_name));
  PetscFunctionReturn(PETSC_SUCCESS);
}
