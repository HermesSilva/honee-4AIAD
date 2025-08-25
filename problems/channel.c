// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Utility functions for setting up Channel flow

#include "../qfunctions/channel.h"

#include <ceed.h>
#include <petscdm.h>

#include <navierstokes.h>

static PetscErrorCode DivDiffFluxVerifyMesh(DM dm);

static PetscErrorCode ChannelOutflowBCSetup_CreateIFunctionQF(BCDefinition bc_def, CeedQFunction *qf) {
  HoneeBCStruct honee_bc;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  PetscCheck(honee_bc->honee->phys->state_var == STATEVAR_CONSERVATIVE, PETSC_COMM_WORLD, PETSC_ERR_SUP,
             "Channel outflow only valid for Conservative variables, recieved %s", StateVariables[honee_bc->honee->phys->state_var]);
  PetscCall(HoneeBCCreateIFunctionQF(bc_def, Channel_Outflow, Channel_Outflow_loc, honee_bc->qfctx, qf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ChannelInflowBCSetup_CreateIFunctionQF(BCDefinition bc_def, CeedQFunction *qf) {
  HoneeBCStruct honee_bc;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  PetscCheck(honee_bc->honee->phys->state_var == STATEVAR_CONSERVATIVE, PETSC_COMM_WORLD, PETSC_ERR_SUP,
             "Channel inflow only valid for Conservative variables, recieved %s", StateVariables[honee_bc->honee->phys->state_var]);
  PetscCall(HoneeBCCreateIFunctionQF(bc_def, Channel_Inflow, Channel_Inflow_loc, honee_bc->qfctx, qf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NS_CHANNEL(ProblemData problem, DM dm, void *ctx) {
  Honee                    honee = *(Honee *)ctx;
  MPI_Comm                 comm  = honee->comm;
  Ceed                     ceed  = honee->ceed;
  ChannelContext           channel_ctx;
  NewtonianIdealGasContext newtonian_ig_ctx;
  CeedQFunctionContext     channel_qfctx;
  PetscBool                use_divdiff_verify_mesh = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(NS_NEWTONIAN_IG(problem, dm, ctx));
  PetscCall(PetscNew(&channel_ctx));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-mesh_transform_channel_div_diff_projection_verify", &use_divdiff_verify_mesh, NULL));
  if (use_divdiff_verify_mesh) PetscCall(DivDiffFluxVerifyMesh(dm));

  // -- Command Line Options
  CeedScalar umax             = 10.;   // m/s
  CeedScalar theta0           = 300.;  // K
  CeedScalar P0               = 1.e5;  // Pa
  PetscReal  body_force_scale = 1.;
  PetscOptionsBegin(comm, NULL, "Options for CHANNEL problem", NULL);
  PetscCall(PetscOptionsScalar("-umax", "Centerline velocity of the Channel", NULL, umax, &umax, NULL));
  PetscCall(PetscOptionsScalar("-theta0", "Wall temperature", NULL, theta0, &theta0, NULL));
  PetscCall(PetscOptionsScalar("-P0", "Pressure at outflow", NULL, P0, &P0, NULL));
  PetscCall(PetscOptionsReal("-body_force_scale", "Multiplier for body force", NULL, body_force_scale = 1, &body_force_scale, NULL));
  PetscOptionsEnd();

  Units units = honee->units;

  theta0 *= units->Kelvin;
  P0 *= units->Pascal;
  umax *= units->meter / units->second;

  //-- Setup Problem information
  CeedScalar H, center;
  {
    PetscReal domain_min[3], domain_max[3], domain_size[3];
    PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
    for (PetscInt i = 0; i < 3; i++) domain_size[i] = domain_max[i] - domain_min[i];

    H      = 0.5 * domain_size[1] * units->meter;
    center = H + domain_min[1] * units->meter;
  }

  // Some properties depend on parameters from NewtonianIdealGas
  PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->apply_vol_rhs.qfctx, CEED_MEM_HOST, &newtonian_ig_ctx));

  channel_ctx->center   = center;
  channel_ctx->H        = H;
  channel_ctx->theta0   = theta0;
  channel_ctx->P0       = P0;
  channel_ctx->umax     = umax;
  channel_ctx->implicit = honee->phys->implicit;
  channel_ctx->B        = body_force_scale * 2 * umax * newtonian_ig_ctx->gas.mu / (H * H);

  {
    // Calculate Body force
    CeedScalar cv = newtonian_ig_ctx->gas.cv, cp = newtonian_ig_ctx->gas.cp;
    CeedScalar Rd  = cp - cv;
    CeedScalar rho = P0 / (Rd * theta0);
    CeedScalar g[] = {channel_ctx->B / rho, 0., 0.};
    PetscCall(PetscArraycpy(newtonian_ig_ctx->g, g, 3));
  }
  channel_ctx->newt_ctx = *newtonian_ig_ctx;
  PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfctx, &newtonian_ig_ctx));

  PetscCallCeed(ceed, CeedQFunctionContextCreate(honee->ceed, &channel_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(channel_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*channel_ctx), channel_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(channel_qfctx, CEED_MEM_HOST, FreeContextPetsc));

  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->ics.qfctx));
  problem->ics = (HoneeQFSpec){.qf_func_ptr = ICsChannel, .qf_loc = ICsChannel_loc, .qfctx = channel_qfctx};

  for (PetscCount b = 0; b < problem->num_bc_defs; b++) {
    BCDefinition bc_def = problem->bc_defs[b];
    const char  *name;

    PetscCall(BCDefinitionGetInfo(bc_def, &name, NULL, NULL));
    if (honee->phys->state_var == STATEVAR_CONSERVATIVE && !strcmp(name, "outflow")) {
      HoneeBCStruct honee_bc;

      PetscCall(PetscPrintf(comm, "WARNING! Channel flow with Inflow and Outflow is currently broken.\n"));
      PetscCall(PetscNew(&honee_bc));
      PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(channel_qfctx, &honee_bc->qfctx));
      honee_bc->honee              = honee;
      honee_bc->num_comps_jac_data = honee->phys->implicit ? 11 : 0;
      PetscCall(BCDefinitionSetContext(bc_def, HoneeBCDestroy, honee_bc));

      PetscCall(BCDefinitionSetIFunction(bc_def, ChannelOutflowBCSetup_CreateIFunctionQF, HoneeBCAddIFunctionOp));
      PetscCall(BCDefinitionSetIJacobian(bc_def, NULL, NULL));
    } else if (honee->phys->state_var == STATEVAR_CONSERVATIVE && !strcmp(name, "inflow")) {
      HoneeBCStruct honee_bc;

      PetscCall(PetscPrintf(comm, "WARNING! Channel flow with Inflow and Outflow is currently broken.\n"));
      PetscCall(PetscNew(&honee_bc));
      PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(channel_qfctx, &honee_bc->qfctx));
      honee_bc->honee              = honee;
      honee_bc->num_comps_jac_data = honee->phys->implicit ? 11 : 0;
      PetscCall(BCDefinitionSetContext(bc_def, HoneeBCDestroy, honee_bc));

      PetscCall(BCDefinitionSetIFunction(bc_def, ChannelInflowBCSetup_CreateIFunctionQF, HoneeBCAddIFunctionOp));
      PetscCall(BCDefinitionSetIJacobian(bc_def, NULL, NULL));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// This function transforms the mesh coordinates to mimic the mesh used in
// *A better consistency for low-order stabilized finite element methods* Jansen et. al. 1999
// which is used to verify the projection of divergence of diffusive flux. See !27 and !94 for more details.
static PetscErrorCode DivDiffFluxVerifyMesh(DM dm) {
  PetscInt     narr, ncoords, dim;
  PetscReal    domain_min[3], domain_max[3], domain_size[3];
  PetscScalar *arr_coords;
  Vec          vec_coords;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  // Get domain boundary information
  PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
  for (PetscInt i = 0; i < 3; i++) domain_size[i] = domain_max[i] - domain_min[i];

  // Get coords array from DM
  PetscCall(DMGetCoordinatesLocal(dm, &vec_coords));
  PetscCall(VecGetLocalSize(vec_coords, &narr));
  PetscCall(VecGetArray(vec_coords, &arr_coords));

  PetscScalar(*coords)[dim] = (PetscScalar(*)[dim])arr_coords;
  ncoords                   = narr / dim;

  // Get mesh information
  PetscInt nmax = 3, faces[3];
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &nmax, NULL));
  // Get element size of the box mesh, for indexing each node
  const PetscReal dxbox = domain_size[0] / (faces[0]);

  for (PetscInt i = 0; i < ncoords; i++) {
    PetscInt x_box_index = round(coords[i][0] / dxbox);
    if (x_box_index % 2) {
      coords[i][0] = (x_box_index - 1) * dxbox + 0.5 * dxbox;
    }
  }

  PetscCall(VecRestoreArray(vec_coords, &arr_coords));
  PetscCall(DMSetCoordinatesLocal(dm, vec_coords));
  PetscFunctionReturn(0);
}
