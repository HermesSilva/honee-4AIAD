// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Utility functions for setting up Gaussian Wave problem

#include "../qfunctions/gaussianwave.h"

#include <navierstokes.h>

#include <petscdm.h>

#include <ceed.h>

PetscErrorCode NS_GAUSSIAN_WAVE(ProblemData problem, DM dm, void *ctx) {
  Honee                    honee = *(Honee *)ctx;
  MPI_Comm                 comm  = honee->comm;
  Ceed                     ceed  = honee->ceed;
  GaussianWaveContext      gausswave_ctx;
  NewtonianIdealGasContext newtonian_ig_ctx;
  CeedQFunctionContext     gausswave_qfctx;
  const StatePrimitive    *reference;

  PetscFunctionBeginUser;
  PetscCall(NS_NEWTONIAN_IG(problem, dm, ctx));

  // -- Option Defaults
  CeedScalar epicenter[3] = {0.};   // m
  CeedScalar width        = 0.002;  // m
  CeedScalar amplitude    = 0.1;    // -

  PetscOptionsBegin(comm, NULL, "Options for GAUSSIAN_WAVE problem", NULL);
  PetscInt narray = 3;
  PetscCall(PetscOptionsScalarArray("-epicenter", "Coordinates of center of wave", NULL, epicenter, &narray, NULL));
  PetscCheck(narray == 3, comm, PETSC_ERR_ARG_SIZ, "-epicenter should recieve array of size 3, instead recieved size %" PetscInt_FMT ".", narray);
  PetscCall(PetscOptionsScalar("-width", "Width parameter for perturbation size", NULL, width, &width, NULL));
  PetscCall(PetscOptionsScalar("-amplitude", "Amplitude of the perturbation", NULL, amplitude, &amplitude, NULL));
  PetscOptionsEnd();

  width *= honee->units->meter;
  for (int i = 0; i < 3; i++) epicenter[i] *= honee->units->meter;

  // -- Set gausswave_ctx struct values
  PetscCall(PetscNew(&gausswave_ctx));
  PetscCall(HoneeGetContainer(honee, "Reference State", &reference));
  PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->apply_vol_rhs.qfctx, CEED_MEM_HOST, &newtonian_ig_ctx));
  gausswave_ctx->S_infty   = StateFromPrimitive(newtonian_ig_ctx->gas, *reference);
  gausswave_ctx->amplitude = amplitude;
  gausswave_ctx->width     = width;
  gausswave_ctx->newt_ctx  = *newtonian_ig_ctx;
  PetscCall(PetscArraycpy(gausswave_ctx->epicenter, epicenter, 3));

  PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfctx, &newtonian_ig_ctx));

  PetscCallCeed(ceed, CeedQFunctionContextCreate(honee->ceed, &gausswave_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(gausswave_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*gausswave_ctx), gausswave_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(gausswave_qfctx, CEED_MEM_HOST, FreeContextPetsc));
  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->ics.qfctx));

  switch (honee->phys->state_var) {
    case STATEVAR_CONSERVATIVE:
      problem->ics = (HoneeQFSpec){.qf_func_ptr = IC_GaussianWave_Conserv, .qf_loc = IC_GaussianWave_Conserv_loc};
      break;
    case STATEVAR_PRIMITIVE:
      problem->ics = (HoneeQFSpec){.qf_func_ptr = IC_GaussianWave_Prim, .qf_loc = IC_GaussianWave_Prim_loc};
      break;
    case STATEVAR_ENTROPY:
      problem->ics = (HoneeQFSpec){.qf_func_ptr = IC_GaussianWave_Entropy, .qf_loc = IC_GaussianWave_Entropy_loc};
      break;
  }
  problem->ics.qfctx = gausswave_qfctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}
