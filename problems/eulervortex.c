// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Utility functions for setting up EULER_VORTEX

#include "../qfunctions/eulervortex.h"

#include <navierstokes.h>

#include <petscdm.h>

#include <ceed.h>
#include <ceed/backend.h>

static PetscErrorCode PRINT_EULER_VORTEX(Honee honee, ProblemData problem, AppCtx app_ctx) {
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

static PetscErrorCode EulerVortexOutflowBCSetup_CreateIFunctionQF(BCDefinition bc_def, CeedQFunction *qf) {
  HoneeBCStruct honee_bc;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  PetscCall(HoneeBCCreateIFunctionQF(bc_def, Euler_Outflow, Euler_Outflow_loc, honee_bc->qfctx, qf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode EulerVortexInflowBCSetup_CreateIFunctionQF(BCDefinition bc_def, CeedQFunction *qf) {
  HoneeBCStruct honee_bc;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  PetscCall(HoneeBCCreateIFunctionQF(bc_def, TravelingVortex_Inflow, TravelingVortex_Inflow_loc, honee_bc->qfctx, qf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define EULER_QFCTX_KEY "Euler QFunction Ctx"

static PetscErrorCode BCDefinitionNaturalSetup_EulerInflow(BCDefinition bc_def, void *ctx) {
  HoneeBCStruct        honee_bc = (HoneeBCStruct)ctx;
  Honee                honee    = honee_bc->honee;
  Ceed                 ceed     = honee->ceed;
  CeedQFunctionContext euler_qfctx;

  PetscFunctionBeginUser;
  PetscCall(HoneeGetContainer(honee, EULER_QFCTX_KEY, &euler_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(euler_qfctx, &honee_bc->qfctx));
  honee_bc->num_comps_jac_data = 0;
  PetscCall(BCDefinitionSetIFunction(bc_def, EulerVortexInflowBCSetup_CreateIFunctionQF, HoneeBCAddIFunctionOp));
  PetscCall(BCDefinitionSetIJacobian(bc_def, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BCDefinitionNaturalSetup_EulerOutflow(BCDefinition bc_def, void *ctx) {
  HoneeBCStruct        honee_bc = (HoneeBCStruct)ctx;
  Honee                honee    = honee_bc->honee;
  Ceed                 ceed     = honee->ceed;
  CeedQFunctionContext euler_qfctx;

  PetscFunctionBeginUser;
  PetscCall(HoneeGetContainer(honee, EULER_QFCTX_KEY, &euler_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(euler_qfctx, &honee_bc->qfctx));
  honee_bc->num_comps_jac_data = 0;
  PetscCall(BCDefinitionSetIFunction(bc_def, EulerVortexOutflowBCSetup_CreateIFunctionQF, HoneeBCAddIFunctionOp));
  PetscCall(BCDefinitionSetIJacobian(bc_def, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static const char *const component_names[] = {"Density", "MomentumX", "MomentumY", "MomentumZ", "TotalEnergy"};

PetscErrorCode NS_EULER_VORTEX(ProblemData problem, DM dm, void *ctx) {
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
  PetscCall(PetscNew(&euler_ctx));

  // ------------------------------------------------------
  //               SET UP EULER VORTEX
  // ------------------------------------------------------
  problem->ics                          = (HoneeQFSpec){.qf_func_ptr = ICsEuler, .qf_loc = ICsEuler_loc};
  problem->apply_vol_rhs                = (HoneeQFSpec){.qf_func_ptr = Euler, .qf_loc = Euler_loc};
  problem->apply_vol_ifunction          = (HoneeQFSpec){.qf_func_ptr = IFunction_Euler, .qf_loc = IFunction_Euler_loc};
  problem->num_comps_jac_data           = 0;
  problem->compute_exact_solution_error = PETSC_TRUE;
  problem->print_info                   = PRINT_EULER_VORTEX;

  problem->num_components = 5;
  PetscCall(PetscMalloc1(problem->num_components, &problem->component_names));
  for (PetscInt i = 0; i < 5; i++) PetscCall(PetscStrallocpy(component_names[i], &problem->component_names[i]));

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  Units      units           = honee->units;
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
  //              Command line Options
  // ------------------------------------------------------
  PetscOptionsBegin(comm, NULL, "Options for EULER_VORTEX problem", NULL);
  // -- Physics
  PetscCall(PetscOptionsScalar("-vortex_strength", "Strength of Vortex", NULL, vortex_strength, &vortex_strength, NULL));
  PetscInt  n = dim;
  PetscBool user_velocity;
  PetscCall(PetscOptionsRealArray("-mean_velocity", "Background velocity vector", NULL, mean_velocity, &n, &user_velocity));
  for (PetscInt i = 0; i < dim; i++) center[i] = .5 * domain_size[i] / units->meter;  // Redimensionalize domain
  n = dim;
  PetscCall(PetscOptionsRealArray("-center", "Location of vortex center", NULL, center, &n, NULL));
  PetscCall(PetscOptionsBool("-implicit", "Use implicit (IFunction) formulation", NULL, implicit = PETSC_FALSE, &implicit, NULL));
  PetscCall(PetscOptionsEnum("-euler_test", "Euler test option", NULL, EulerTestTypes, (PetscEnum)(euler_test = EULER_TEST_ISENTROPIC_VORTEX),
                             (PetscEnum *)&euler_test, NULL));
  PetscCall(PetscOptionsEnum("-stab", "Stabilization method", NULL, StabilizationTypes, (PetscEnum)(stab = STAB_NONE), (PetscEnum *)&stab, NULL));
  PetscCall(PetscOptionsScalar("-c_tau", "Stabilization constant", NULL, c_tau, &c_tau, NULL));

  // -- Warnings
  if (stab == STAB_SUPG && !implicit) {
    PetscCall(PetscPrintf(comm, "Warning! Use -stab supg only with -implicit\n"));
  }
  if (user_velocity && (euler_test == EULER_TEST_1 || euler_test == EULER_TEST_3)) {
    PetscCall(PetscPrintf(comm, "Warning! Background velocity vector for -euler_test t1 and -euler_test t3 is (0,0,0)\n"));
  }

  PetscOptionsEnd();

  // ------------------------------------------------------
  //           Set up the libCEED context
  // ------------------------------------------------------
  // -- Scale variables to desired units
  for (PetscInt i = 0; i < 3; i++) {
    center[i] *= units->meter;
    mean_velocity[i] *= (units->meter / units->second);
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

  PetscCallCeed(ceed, CeedQFunctionContextReference(euler_qfctx));
  PetscCall(HoneeSetContainer(honee, EULER_QFCTX_KEY, euler_qfctx, (PetscCtxDestroyFn *)CeedQFunctionContextDestroy));
  PetscCall(BCDefinitionRegisterNaturalSetup("inflow", BCDefinitionNaturalSetup_EulerInflow));    // For legacy BC setting
  PetscCall(BCDefinitionRegisterNaturalSetup("outflow", BCDefinitionNaturalSetup_EulerOutflow));  // For legacy BC setting
  PetscCall(BCDefinitionRegisterNaturalSetup("vortex-inflow", BCDefinitionNaturalSetup_EulerInflow));
  PetscCall(BCDefinitionRegisterNaturalSetup("vortex-outflow", BCDefinitionNaturalSetup_EulerOutflow));

  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&euler_qfctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}
