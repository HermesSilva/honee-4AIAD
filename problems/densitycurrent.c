// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Utility functions for setting up DENSITY_CURRENT

#include "../qfunctions/densitycurrent.h"

#include <ceed.h>
#include <petscdm.h>

#include <navierstokes.h>

PetscErrorCode NS_DENSITY_CURRENT(ProblemData problem, DM dm, void *ctx) {
  Honee                    honee = *(Honee *)ctx;
  MPI_Comm                 comm  = honee->comm;
  Ceed                     ceed  = honee->ceed;
  DensityCurrentContext    density_current_ctx;
  CeedQFunctionContext     density_current_qfctx;
  NewtonianIdealGasContext newtonian_ig_ctx;
  PetscInt                 dim;

  PetscFunctionBeginUser;
  PetscCall(NS_NEWTONIAN_IG(problem, dm, ctx));
  PetscCall(PetscNew(&density_current_ctx));

  // ------------------------------------------------------
  //             Create the QFunction context
  // ------------------------------------------------------
  CeedScalar theta0 = 300.;   // K
  CeedScalar thetaC = -15.;   // K
  CeedScalar P0     = 1.e5;   // Pa
  CeedScalar N      = 0.01;   // 1/s
  CeedScalar rc     = 1000.;  // m (Radius of bubble)
  PetscReal  center[3], dc_axis[3] = {0, 0, 0};
  PetscReal  domain_min[3], domain_max[3], domain_size[3];
  PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
  PetscCall(DMGetDimension(dm, &dim));
  for (PetscInt i = 0; i < dim; i++) domain_size[i] = domain_max[i] - domain_min[i];

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  PetscOptionsBegin(comm, NULL, "Options for DENSITY_CURRENT problem", NULL);
  PetscCall(PetscOptionsScalar("-theta0", "Reference potential temperature", NULL, theta0, &theta0, NULL));
  PetscCall(PetscOptionsScalar("-thetaC", "Perturbation of potential temperature", NULL, thetaC, &thetaC, NULL));
  PetscCall(PetscOptionsScalar("-P0", "Atmospheric pressure", NULL, P0, &P0, NULL));
  PetscCall(PetscOptionsScalar("-N", "Brunt-Vaisala frequency", NULL, N, &N, NULL));
  PetscCall(PetscOptionsScalar("-rc", "Characteristic radius of thermal bubble", NULL, rc, &rc, NULL));
  for (PetscInt i = 0; i < dim; i++) center[i] = .5 * domain_size[i];
  PetscInt n = dim;
  PetscCall(PetscOptionsRealArray("-center", "Location of bubble center", NULL, center, &n, NULL));
  n = dim;
  PetscCall(PetscOptionsRealArray("-dc_axis",
                                  "Axis of density current cylindrical anomaly, "
                                  "or {0,0,0} for spherically symmetric",
                                  NULL, dc_axis, &n, NULL));
  {
    PetscReal norm = PetscSqrtReal(PetscSqr(dc_axis[0]) + PetscSqr(dc_axis[1]) + PetscSqr(dc_axis[2]));
    if (norm > 0) {
      for (PetscInt i = 0; i < dim; i++) dc_axis[i] /= norm;
    }
  }

  PetscOptionsEnd();

  Units units = honee->units;

  rc = fabs(rc) * units->meter;
  theta0 *= units->Kelvin;
  thetaC *= units->Kelvin;
  P0 *= units->Pascal;
  N *= (1. / units->second);
  for (PetscInt i = 0; i < dim; i++) center[i] *= units->meter;

  density_current_ctx->theta0 = theta0;
  density_current_ctx->thetaC = thetaC;
  density_current_ctx->P0     = P0;
  density_current_ctx->N      = N;
  density_current_ctx->rc     = rc;
  PetscCall(PetscArraycpy(density_current_ctx->center, center, 3));
  PetscCall(PetscArraycpy(density_current_ctx->dc_axis, dc_axis, 3));

  PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->apply_vol_rhs.qfctx, CEED_MEM_HOST, &newtonian_ig_ctx));
  density_current_ctx->newtonian_ctx = *newtonian_ig_ctx;
  PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfctx, &newtonian_ig_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextCreate(honee->ceed, &density_current_qfctx));
  PetscCallCeed(
      ceed, CeedQFunctionContextSetData(density_current_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*density_current_ctx), density_current_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(density_current_qfctx, CEED_MEM_HOST, FreeContextPetsc));

  // ------------------------------------------------------
  //               SET UP DENSITY_CURRENT
  // ------------------------------------------------------
  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->ics.qfctx));
  problem->ics = (HoneeQFSpec){.qf_func_ptr = ICsDC, .qf_loc = ICsDC_loc, .qfctx = density_current_qfctx};
  PetscFunctionReturn(PETSC_SUCCESS);
}
