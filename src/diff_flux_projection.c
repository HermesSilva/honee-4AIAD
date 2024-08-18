// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
/// @file
/// Functions for setting up and projecting the divergence of the diffusive flux

#include "../qfunctions/diff_flux_projection.h"

#include <petscdmplex.h>

#include <navierstokes.h>

/**
  @brief Create `DivDiffFluxProjectionData` for solution DM in `user`

  @param[in]  user                `User` context
  @param[in]  num_diff_flux_comps Number of components that makes up the diffusive flux (e.g. 1 for scalar advection-diffusion)
  @param[out] pdiff_flux_proj     The `DivDiffFluxProjectionData` object created, or `NULL` if not created
**/
PetscErrorCode DivDiffFluxProjectionCreate(User user, PetscInt num_diff_flux_comps, DivDiffFluxProjectionData *pdiff_flux_proj) {
  PetscInt                  label_value = 0, height = 0, dm_field = 0, dim, degree = user->app_ctx->degree, q_extra = user->app_ctx->q_extra;
  DMLabel                   domain_label = NULL;
  DivDiffFluxProjectionData diff_flux_proj;
  NodalProjectionData       projection;

  PetscFunctionBeginUser;
  if (user->app_ctx->divFdiffproj_method == DIV_DIFF_FLUX_PROJ_NONE) {
    *pdiff_flux_proj = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscNew(pdiff_flux_proj));
  diff_flux_proj = *pdiff_flux_proj;
  PetscCall(PetscNew(&user->diff_flux_proj->projection));
  projection                          = user->diff_flux_proj->projection;
  diff_flux_proj->method              = user->app_ctx->divFdiffproj_method;
  diff_flux_proj->num_diff_flux_comps = num_diff_flux_comps;

  PetscCall(DMClone(user->dm, &projection->dm));
  PetscCall(DMSetMatrixPreallocateSkip(projection->dm, PETSC_TRUE));
  PetscCall(DMGetDimension(projection->dm, &dim));
  switch (diff_flux_proj->method) {
    case DIV_DIFF_FLUX_PROJ_DIRECT: {
      projection->num_comp = diff_flux_proj->num_diff_flux_comps;
      PetscCall(PetscObjectSetName((PetscObject)projection->dm, "DivDiffFluxProj"));
      PetscCall(DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, degree, 1, q_extra, 1, &projection->num_comp, projection->dm));

      PetscCall(DMPlexCeedElemRestrictionCreate(user->ceed, projection->dm, domain_label, label_value, height, dm_field,
                                                &diff_flux_proj->elem_restr_div_diff_flux));
      PetscCallCeed(user->ceed, CeedElemRestrictionCreateVector(diff_flux_proj->elem_restr_div_diff_flux, &diff_flux_proj->div_diff_flux_ceed, NULL));
      PetscCall(CreateBasisFromPlex(user->ceed, projection->dm, domain_label, label_value, height, dm_field, &diff_flux_proj->basis_div_diff_flux));
      diff_flux_proj->eval_mode_div_diff_flux = CEED_EVAL_INTERP;

      {  // Create face labels on projection->dm for boundary integrals
        DMLabel  face_sets_label;
        PetscInt num_face_set_values, *face_set_values;

        PetscCall(DMGetLabel(user->dm, "Face Sets", &face_sets_label));
        PetscCall(DMLabelCreateGlobalValueArray(user->dm, face_sets_label, &num_face_set_values, &face_set_values));
        for (PetscInt f = 0; f < num_face_set_values; f++) {
          DMLabel face_orientation_label;
          char   *face_orientation_label_name;

          PetscCall(DMPlexCreateFaceLabel(user->dm, face_set_values[f], &face_orientation_label_name));
          PetscCall(DMGetLabel(user->dm, face_orientation_label_name, &face_orientation_label));
          PetscCall(DMAddLabel(projection->dm, face_orientation_label));
          PetscCall(PetscFree(face_orientation_label_name));
        }
        PetscCall(PetscFree(face_set_values));
      }
    } break;
    case DIV_DIFF_FLUX_PROJ_INDIRECT: {
      projection->num_comp = diff_flux_proj->num_diff_flux_comps * dim;
      PetscCall(PetscObjectSetName((PetscObject)projection->dm, "DiffFluxProj"));
      PetscCall(DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, degree, 1, q_extra, 1, &projection->num_comp, projection->dm));

      PetscCall(DMPlexCeedElemRestrictionQDataCreate(user->ceed, projection->dm, domain_label, label_value, height,
                                                     diff_flux_proj->num_diff_flux_comps, &diff_flux_proj->elem_restr_div_diff_flux));
      PetscCallCeed(user->ceed, CeedElemRestrictionCreateVector(diff_flux_proj->elem_restr_div_diff_flux, &diff_flux_proj->div_diff_flux_ceed, NULL));
      diff_flux_proj->basis_div_diff_flux     = CEED_BASIS_NONE;
      diff_flux_proj->eval_mode_div_diff_flux = CEED_EVAL_NONE;
    } break;
    case DIV_DIFF_FLUX_PROJ_NONE:
      SETERRQ(PetscObjectComm((PetscObject)user->dm), PETSC_ERR_ARG_WRONG, "Should not reach here with div_diff_flux_projection_method %s",
              DivDiffFluxProjectionMethods[user->app_ctx->divFdiffproj_method]);
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
};

/**
  @brief Return the objects required for the Divergence of Diffusive flux to be read by a `CeedOperator`

  @param[in]  diff_flux_proj Projection object
  @param[out] elem_restr     Element restriction for the divergence of diffusive flux, or `NULL`
  @param[out] basis          Basis for the divergence of diffusive flux, or `NULL`
  @param[out] vector         Vector for the divergence of diffusive flux, or `NULL`
  @param[out] eval_mode      Eval mode for the divergence of diffusive flux, or `NULL`
**/
PetscErrorCode DivDiffFluxProjectionGetOperatorFieldData(DivDiffFluxProjectionData diff_flux_proj, CeedElemRestriction *elem_restr, CeedBasis *basis,
                                                         CeedVector *vector, CeedEvalMode *eval_mode) {
  Ceed ceed = CeedVectorReturnCeed(diff_flux_proj->div_diff_flux_ceed);

  PetscFunctionBeginUser;
  if (elem_restr) PetscCallCeed(ceed, CeedElemRestrictionReferenceCopy(diff_flux_proj->elem_restr_div_diff_flux, elem_restr));
  if (basis) PetscCallCeed(ceed, CeedBasisReferenceCopy(diff_flux_proj->basis_div_diff_flux, basis));
  if (vector) PetscCallCeed(ceed, CeedVectorReferenceCopy(diff_flux_proj->div_diff_flux_ceed, vector));
  if (eval_mode) *eval_mode = diff_flux_proj->eval_mode_div_diff_flux;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Setup direct projection of divergence of diffusive flux

  @param[in]     user           `User` context
  @param[in]     ceed_data      `CeedData` context
  @param[in,out] diff_flux_proj Flux projection object to setup
**/
static PetscErrorCode DivDiffFluxProjectionSetup_Direct(User user, CeedData ceed_data, DivDiffFluxProjectionData diff_flux_proj) {
  Ceed                ceed       = user->ceed;
  NodalProjectionData projection = diff_flux_proj->projection;
  MPI_Comm            comm       = PetscObjectComm((PetscObject)projection->dm);

  PetscFunctionBeginUser;
  {  // Create Projection RHS OperatorApplyContext
    CeedOperator op_rhs;

    PetscCheck(diff_flux_proj->CreateRHSOperator_Direct, comm, PETSC_ERR_ARG_WRONGSTATE,
               "Must define CreateRHSOperator_Direct to use indirect div_diff_flux projection");
    PetscCall(diff_flux_proj->CreateRHSOperator_Direct(user, ceed_data, diff_flux_proj, &op_rhs));
    PetscCall(DMCreateLocalVector(projection->dm, &diff_flux_proj->DivDiffFlux_loc));
    diff_flux_proj->ceed_vec_has_array = PETSC_FALSE;
    PetscCall(OperatorApplyContextCreate(user->dm, projection->dm, ceed, op_rhs, NULL, NULL, NULL, diff_flux_proj->DivDiffFlux_loc,
                                         &projection->l2_rhs_ctx));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_rhs));
  }

  {  // -- Build Mass operator
    CeedQFunction       qf_mass;
    CeedOperator        op_mass;
    CeedBasis           basis_div_diff_flux             = NULL;
    CeedElemRestriction elem_restr_div_diff_flux_volume = NULL, elem_restr_qd;
    CeedVector          q_data;
    CeedInt             q_data_size;
    PetscInt            label_value  = 0;
    DMLabel             domain_label = NULL;

    PetscCall(DivDiffFluxProjectionGetOperatorFieldData(diff_flux_proj, &elem_restr_div_diff_flux_volume, &basis_div_diff_flux, NULL, NULL));
    PetscCall(QDataGet(ceed, projection->dm, domain_label, label_value, ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord,
                       &elem_restr_qd, &q_data, &q_data_size));

    PetscCall(CreateMassQFunction(ceed, projection->num_comp, q_data_size, &qf_mass));
    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass));
    PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "u", elem_restr_div_diff_flux_volume, basis_div_diff_flux, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "v", elem_restr_div_diff_flux_volume, basis_div_diff_flux, CEED_VECTOR_ACTIVE));

    {  // -- Setup KSP for L^2 projection
      Mat mat_mass;

      PetscCall(MatCreateCeed(projection->dm, projection->dm, op_mass, NULL, &mat_mass));

      PetscCall(KSPCreate(comm, &projection->ksp));
      PetscCall(KSPSetOptionsPrefix(projection->ksp, "div_diff_flux_projection_"));
      {  // lumped by default
        PC pc;
        PetscCall(KSPGetPC(projection->ksp, &pc));
        PetscCall(PCSetType(pc, PCJACOBI));
        PetscCall(PCJacobiSetType(pc, PC_JACOBI_ROWSUM));
        PetscCall(KSPSetType(projection->ksp, KSPPREONLY));
      }
      PetscCall(KSPSetFromOptions_WithMatCeed(projection->ksp, mat_mass));
      PetscCall(MatDestroy(&mat_mass));
    }

    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_div_diff_flux_volume));
    PetscCallCeed(ceed, CeedBasisDestroy(&basis_div_diff_flux));
    PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_mass));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_mass));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Setup indirect projection of divergence of diffusive flux

  @param[in]     user           `User` context
  @param[in]     ceed_data      `CeedData` context
  @param[in,out] diff_flux_proj Flux projection object to setup
**/
static PetscErrorCode DivDiffFluxProjectionSetup_Indirect(User user, CeedData ceed_data, DivDiffFluxProjectionData diff_flux_proj) {
  Ceed                ceed       = user->ceed;
  NodalProjectionData projection = diff_flux_proj->projection;
  CeedBasis           basis_diff_flux;
  CeedElemRestriction elem_restr_diff_flux, elem_restr_qd;
  CeedVector          q_data;
  CeedInt             q_data_size;
  MPI_Comm            comm = PetscObjectComm((PetscObject)projection->dm);

  PetscFunctionBeginUser;
  {
    PetscInt label_value = 0, height = 0, dm_field = 0;
    DMLabel  domain_label = NULL;

    PetscCall(DMPlexCeedElemRestrictionCreate(ceed, projection->dm, domain_label, label_value, height, dm_field, &elem_restr_diff_flux));
    PetscCall(CreateBasisFromPlex(ceed, projection->dm, domain_label, label_value, height, dm_field, &basis_diff_flux));
    PetscCall(QDataGet(ceed, projection->dm, domain_label, label_value, ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord,
                       &elem_restr_qd, &q_data, &q_data_size));
  }

  {
    CeedOperator op_rhs;

    PetscCheck(diff_flux_proj->CreateRHSOperator_Indirect, comm, PETSC_ERR_ARG_WRONGSTATE,
               "Must define CreateRHSOperator_Indirect to use indirect div_diff_flux projection");
    PetscCall(diff_flux_proj->CreateRHSOperator_Indirect(user, ceed_data, diff_flux_proj, &op_rhs));
    PetscCall(OperatorApplyContextCreate(user->dm, projection->dm, ceed, op_rhs, NULL, NULL, NULL, NULL, &projection->l2_rhs_ctx));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_rhs));
  }

  {  // -- Build Mass operator
    CeedQFunction qf_mass;
    CeedOperator  op_mass;

    PetscCall(CreateMassQFunction(ceed, projection->num_comp, q_data_size, &qf_mass));
    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass));
    PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "u", elem_restr_diff_flux, basis_diff_flux, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "v", elem_restr_diff_flux, basis_diff_flux, CEED_VECTOR_ACTIVE));

    {  // -- Setup KSP for L^2 projection
      Mat mat_mass;

      PetscCall(MatCreateCeed(projection->dm, projection->dm, op_mass, NULL, &mat_mass));

      PetscCall(KSPCreate(comm, &projection->ksp));
      PetscCall(KSPSetOptionsPrefix(projection->ksp, "div_diff_flux_projection_"));
      {  // lumped by default
        PC pc;
        PetscCall(KSPGetPC(projection->ksp, &pc));
        PetscCall(PCSetType(pc, PCJACOBI));
        PetscCall(PCJacobiSetType(pc, PC_JACOBI_ROWSUM));
        PetscCall(KSPSetType(projection->ksp, KSPPREONLY));
      }
      PetscCall(KSPSetFromOptions_WithMatCeed(projection->ksp, mat_mass));
      PetscCall(MatDestroy(&mat_mass));
    }
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_mass));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_mass));
  }

  {  // Create OperatorApplyContext to calculate divergence at quadrature points
    CeedQFunction       qf_calc_divergence = NULL;
    CeedOperator        op_calc_divergence;
    CeedElemRestriction elem_restr_div_diff_flux = NULL;
    PetscInt            dim;

    PetscCall(DMGetDimension(projection->dm, &dim));
    PetscCall(DivDiffFluxProjectionGetOperatorFieldData(diff_flux_proj, &elem_restr_div_diff_flux, NULL, NULL, NULL));

    switch (dim) {
      case 2:
        switch (diff_flux_proj->num_diff_flux_comps) {
          case 1:
            PetscCallCeed(ceed,
                          CeedQFunctionCreateInterior(ceed, 1, ComputeDivDiffusiveFlux2D_1, ComputeDivDiffusiveFlux2D_1_loc, &qf_calc_divergence));
            break;
        }
        break;
      case 3:
        switch (diff_flux_proj->num_diff_flux_comps) {
          case 1:
            PetscCallCeed(ceed,
                          CeedQFunctionCreateInterior(ceed, 1, ComputeDivDiffusiveFlux3D_1, ComputeDivDiffusiveFlux3D_1_loc, &qf_calc_divergence));
            break;
          case 4:
            PetscCallCeed(ceed,
                          CeedQFunctionCreateInterior(ceed, 1, ComputeDivDiffusiveFlux3D_4, ComputeDivDiffusiveFlux3D_4_loc, &qf_calc_divergence));
            break;
        }
        break;
    }
    PetscCheck(qf_calc_divergence, comm, PETSC_ERR_SUP,
               "QFunction for calculating divergence of diffusive flux does not exist for"
               " %" PetscInt_FMT " dimensional grid and %" PetscInt_FMT
               " number of components.\nA new qfunction can be easily added; see source code for pattern.",
               dim, diff_flux_proj->num_diff_flux_comps);

    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_calc_divergence, "Grad F_diff", projection->num_comp * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_calc_divergence, "qdata", q_data_size, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_calc_divergence, "Div F_diff", diff_flux_proj->num_diff_flux_comps, CEED_EVAL_NONE));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_calc_divergence, NULL, NULL, &op_calc_divergence));
    PetscCallCeed(ceed, CeedOperatorSetField(op_calc_divergence, "Grad F_diff", elem_restr_diff_flux, basis_diff_flux, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_calc_divergence, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
    PetscCallCeed(
        ceed, CeedOperatorSetField(op_calc_divergence, "Div F_diff", elem_restr_div_diff_flux, CEED_BASIS_NONE, diff_flux_proj->div_diff_flux_ceed));

    PetscCall(
        OperatorApplyContextCreate(projection->dm, NULL, ceed, op_calc_divergence, NULL, NULL, NULL, NULL, &diff_flux_proj->calc_div_diff_flux));

    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_div_diff_flux));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_calc_divergence));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_calc_divergence));
  }
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_diff_flux));
  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_diff_flux));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Setup projection of divergence of diffusive flux

  @param[in]     user           `User` context
  @param[in]     ceed_data      `CeedData` context
  @param[in,out] diff_flux_proj Flux projection object to setup
**/
PetscErrorCode DivDiffFluxProjectionSetup(User user, CeedData ceed_data, DivDiffFluxProjectionData diff_flux_proj) {
  PetscFunctionBeginUser;
  switch (user->app_ctx->divFdiffproj_method) {
    case DIV_DIFF_FLUX_PROJ_DIRECT:
      PetscCall(DivDiffFluxProjectionSetup_Direct(user, ceed_data, diff_flux_proj));
      break;
    case DIV_DIFF_FLUX_PROJ_INDIRECT:
      PetscCall(DivDiffFluxProjectionSetup_Indirect(user, ceed_data, diff_flux_proj));
      break;
    case DIV_DIFF_FLUX_PROJ_NONE:
      SETERRQ(PetscObjectComm((PetscObject)user->dm), PETSC_ERR_ARG_WRONG, "Should not reach here with div_diff_flux_projection_method %s",
              DivDiffFluxProjectionMethods[user->app_ctx->divFdiffproj_method]);
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Project the divergence of diffusive flux

  This implicitly sets the `CeedVector` input (`div_diff_flux_ceed`) to the divergence of diffusive flux.

  @param[in]  diff_flux_proj `NodalProjectionData` for the projection
  @param[in]  Q_loc          Localized solution vector
**/
PetscErrorCode DivDiffFluxProjectionApply(DivDiffFluxProjectionData diff_flux_proj, Vec Q_loc) {
  NodalProjectionData projection = diff_flux_proj->projection;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(FLUIDS_DivDiffFluxProjection, Q_loc, 0, 0, 0));
  switch (diff_flux_proj->method) {
    case DIV_DIFF_FLUX_PROJ_DIRECT: {
      Vec DivDiffFlux;

      PetscCall(DMGetGlobalVector(projection->dm, &DivDiffFlux));
      if (diff_flux_proj->ceed_vec_has_array) {
        PetscCall(VecReadCeedToPetsc(diff_flux_proj->div_diff_flux_ceed, diff_flux_proj->DivDiffFlux_memtype, diff_flux_proj->DivDiffFlux_loc));
        diff_flux_proj->ceed_vec_has_array = PETSC_FALSE;
      }
      PetscCall(ApplyCeedOperatorLocalToGlobal(Q_loc, DivDiffFlux, projection->l2_rhs_ctx));
      PetscCall(VecViewFromOptions(DivDiffFlux, NULL, "-div_diff_flux_projection_rhs_view"));

      PetscCall(KSPSolve(projection->ksp, DivDiffFlux, DivDiffFlux));
      PetscCall(VecViewFromOptions(DivDiffFlux, NULL, "-div_diff_flux_projection_view"));

      PetscCall(DMGlobalToLocal(projection->dm, DivDiffFlux, INSERT_VALUES, diff_flux_proj->DivDiffFlux_loc));
      PetscCall(VecReadPetscToCeed(diff_flux_proj->DivDiffFlux_loc, &diff_flux_proj->DivDiffFlux_memtype, diff_flux_proj->div_diff_flux_ceed));
      diff_flux_proj->ceed_vec_has_array = PETSC_TRUE;

      PetscCall(DMRestoreGlobalVector(projection->dm, &DivDiffFlux));
      break;
    }
    case DIV_DIFF_FLUX_PROJ_INDIRECT: {
      Vec DiffFlux;

      PetscCall(DMGetGlobalVector(projection->dm, &DiffFlux));
      PetscCall(ApplyCeedOperatorLocalToGlobal(Q_loc, DiffFlux, projection->l2_rhs_ctx));
      PetscCall(VecViewFromOptions(DiffFlux, NULL, "-div_diff_flux_projection_rhs_view"));

      PetscCall(KSPSolve(projection->ksp, DiffFlux, DiffFlux));
      PetscCall(VecViewFromOptions(DiffFlux, NULL, "-div_diff_flux_projection_view"));

      PetscCall(ApplyCeedOperatorGlobalToLocal(DiffFlux, NULL, diff_flux_proj->calc_div_diff_flux));
      PetscCall(DMRestoreGlobalVector(projection->dm, &DiffFlux));
    } break;
    case DIV_DIFF_FLUX_PROJ_NONE:
      SETERRQ(PetscObjectComm((PetscObject)projection->dm), PETSC_ERR_ARG_WRONG, "Should not reach here with div_diff_flux_projection_method %s",
              DivDiffFluxProjectionMethods[diff_flux_proj->method]);
      break;
  }
  PetscCall(PetscLogEventEnd(FLUIDS_DivDiffFluxProjection, Q_loc, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Destroy `DivDiffFluxProjectionData` object

  @param[in,out] diff_flux_proj Object to destroy
**/
PetscErrorCode DivDiffFluxProjectionDataDestroy(DivDiffFluxProjectionData diff_flux_proj) {
  PetscFunctionBeginUser;
  if (diff_flux_proj == NULL) PetscFunctionReturn(PETSC_SUCCESS);
  Ceed ceed = CeedVectorReturnCeed(diff_flux_proj->div_diff_flux_ceed);

  PetscCall(NodalProjectionDataDestroy(diff_flux_proj->projection));
  PetscCall(OperatorApplyContextDestroy(diff_flux_proj->calc_div_diff_flux));
  if (diff_flux_proj->ceed_vec_has_array) {
    PetscCall(VecReadCeedToPetsc(diff_flux_proj->div_diff_flux_ceed, diff_flux_proj->DivDiffFlux_memtype, diff_flux_proj->DivDiffFlux_loc));
    diff_flux_proj->ceed_vec_has_array = PETSC_FALSE;
  }
  PetscCallCeed(ceed, CeedVectorDestroy(&diff_flux_proj->div_diff_flux_ceed));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&diff_flux_proj->elem_restr_div_diff_flux));
  PetscCallCeed(ceed, CeedBasisDestroy(&diff_flux_proj->basis_div_diff_flux));
  PetscCall(VecDestroy(&diff_flux_proj->DivDiffFlux_loc));
  PetscCall(PetscFree(diff_flux_proj));
  PetscFunctionReturn(PETSC_SUCCESS);
}
