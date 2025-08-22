// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Utility functions for setting up Taylor-Green Vortex

#include "../qfunctions/taylorgreen.h"

#include <navierstokes.h>

PetscErrorCode NS_TAYLOR_GREEN(ProblemData problem, DM dm, void *ctx) {
  Honee                honee = *(Honee *)ctx;
  Ceed                 ceed  = honee->ceed;
  MPI_Comm             comm  = honee->comm;
  TaylorGreenContext   taylorgreen_ctx;
  CeedQFunctionContext taylorgreen_qfctx;
  SetupContext         setup_ctx;

  PetscFunctionBeginUser;
  PetscCall(NS_NEWTONIAN_IG(problem, dm, ctx));

  PetscCall(PetscNew(&taylorgreen_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->ics.qfctx, CEED_MEM_HOST, &setup_ctx));
  *taylorgreen_ctx = (struct TaylorGreenContext_){
      .reference = setup_ctx->reference,
      .gas       = setup_ctx->gas,
      .lx        = setup_ctx->lx,
      .ly        = setup_ctx->ly,
      .lz        = setup_ctx->lz,
  };
  PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->ics.qfctx, &setup_ctx));

  PetscOptionsBegin(comm, NULL, "Options for TAYLOR-GREEN problem", NULL);
  PetscInt u_size = PETSC_STATIC_ARRAY_LENGTH(taylorgreen_ctx->u);
  PetscCall(PetscOptionsScalarArray("-taylorgreen_background_velocity", "Background velocity to add to Taylor-Green initial condition", NULL,
                                    taylorgreen_ctx->u, &u_size, NULL));
  PetscOptionsEnd();

  PetscCallCeed(ceed, CeedQFunctionContextCreate(honee->ceed, &taylorgreen_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(taylorgreen_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*taylorgreen_ctx), taylorgreen_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(taylorgreen_qfctx, CEED_MEM_HOST, FreeContextPetsc));

  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->ics.qfctx));
  problem->ics = (HoneeQFSpec){
      .qf_func_ptr = ICsTaylorGreen,
      .qf_loc      = ICsTaylorGreen_loc,
      .qfctx       = taylorgreen_qfctx,
  };
  PetscFunctionReturn(PETSC_SUCCESS);
}
