// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Utility functions for setting up Taylor-Green Vortex

#include "../qfunctions/taylorgreen.h"

#include <navierstokes.h>

PetscErrorCode NS_TAYLOR_GREEN(ProblemData problem, DM dm, void *ctx, SimpleBC bc) {
  Honee                honee = *(Honee *)ctx;
  Ceed                 ceed  = honee->ceed;
  MPI_Comm             comm  = honee->comm;
  TaylorGreenContext   taylorgreen_ctx;
  CeedQFunctionContext taylorgreen_qfctx;
  SetupContext         setup_ctx;
  PetscScalar          background_velocity[3]   = {0};
  PetscInt             background_velocity_size = PETSC_STATIC_ARRAY_LENGTH(background_velocity);

  PetscFunctionBeginUser;
  PetscCall(NS_NEWTONIAN_IG(problem, dm, ctx, bc));

  PetscOptionsBegin(comm, NULL, "Options for TAYLOR-GREEN problem", NULL);
  PetscCall(PetscOptionsScalarArray("-taylorgreen_background_velocity", "Background velocity to add to Taylor-Green initial condition", NULL,
                                    background_velocity, &background_velocity_size, NULL));
  PetscOptionsEnd();

  PetscCall(PetscCalloc1(1, &taylorgreen_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->ics.qfctx, CEED_MEM_HOST, &setup_ctx));
  taylorgreen_ctx->reference = setup_ctx->reference;
  taylorgreen_ctx->gas       = setup_ctx->gas;
  taylorgreen_ctx->lx        = setup_ctx->lx;
  taylorgreen_ctx->ly        = setup_ctx->ly;
  taylorgreen_ctx->lz        = setup_ctx->lz;
  PetscCall(PetscArraycpy(taylorgreen_ctx->u, background_velocity, background_velocity_size));
  PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->ics.qfctx, &setup_ctx));

  PetscCallCeed(ceed, CeedQFunctionContextCreate(honee->ceed, &taylorgreen_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(taylorgreen_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*taylorgreen_ctx), taylorgreen_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(taylorgreen_qfctx, CEED_MEM_HOST, FreeContextPetsc));

  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->ics.qfctx));
  problem->ics.qf_func_ptr = ICsTaylorGreen;
  problem->ics.qf_loc      = ICsTaylorGreen_loc;
  problem->ics.qfctx       = taylorgreen_qfctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}
