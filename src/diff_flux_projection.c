// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
/// @file
/// Functions for setting up and projecting the divergence of the diffusive flux

#include "../qfunctions/diff_flux_projection.h"

#include <petscdmplex.h>

#include <navierstokes.h>

/**
  @brief Initialize projection of divergence of diffusive flux

  Creates underlying `DM` for the projection operation and creates the restriction and basis to use with the CeedOperator

  @param[in]  user                     `User` context
  @param[out] elem_restr_div_diff_flux `CeedElemRestriction` of the divergence of diffusive flux vector
  @param[out] basis_div_diff_flux      `CeedBasis` of the divergence of diffusive flux vector
  @param[out] eval_mode_diff_flux      `CeedEvalMode` for the divergence of the diffusive flux
**/
PetscErrorCode DiffFluxProjectionInitialize(User user, CeedElemRestriction *elem_restr_div_diff_flux, CeedBasis *basis_div_diff_flux,
                                            CeedEvalMode *eval_mode_diff_flux) {
  PetscSection              section;
  PetscInt                  label_value = 0, height = 0, dm_field = 0, dim;
  DMLabel                   domain_label = NULL;
  DivDiffFluxProjectionData diff_flux_proj;
  NodalProjectionData       projection;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&user->diff_flux_proj));
  diff_flux_proj = user->diff_flux_proj;
  PetscCall(PetscNew(&user->diff_flux_proj->projection));
  projection                          = user->diff_flux_proj->projection;
  diff_flux_proj->method              = user->app_ctx->divFdiffproj_method;
  diff_flux_proj->num_diff_flux_comps = 4;

  PetscCall(DMClone(user->dm, &projection->dm));
  PetscCall(DMSetMatrixPreallocateSkip(projection->dm, PETSC_TRUE));
  PetscCall(DMGetDimension(projection->dm, &dim));
  switch (diff_flux_proj->method) {
    case DIV_DIFF_FLUX_PROJ_DIRECT: {
      projection->num_comp = diff_flux_proj->num_diff_flux_comps;
      PetscCall(PetscObjectSetName((PetscObject)projection->dm, "DivDiffFluxProj"));
      PetscCall(
          DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, user->app_ctx->degree, 1, user->app_ctx->q_extra, 1, &projection->num_comp, projection->dm));

      PetscCall(DMGetLocalSection(projection->dm, &section));
      PetscCall(PetscSectionSetFieldName(section, 0, ""));
      PetscCall(PetscSectionSetComponentName(section, 0, 0, "DivDiffusiveFlux_MomentumX"));
      PetscCall(PetscSectionSetComponentName(section, 0, 1, "DivDiffusiveFlux_MomentumY"));
      PetscCall(PetscSectionSetComponentName(section, 0, 2, "DivDiffusiveFlux_MomentumZ"));
      PetscCall(PetscSectionSetComponentName(section, 0, 3, "DivDiffusiveFlux_Energy"));

      PetscCall(DMPlexCeedElemRestrictionCreate(user->ceed, projection->dm, domain_label, label_value, height, dm_field, elem_restr_div_diff_flux));
      PetscCallCeed(user->ceed, CeedElemRestrictionCreateVector(*elem_restr_div_diff_flux, &diff_flux_proj->div_diff_flux_ceed, NULL));
      PetscCall(CreateBasisFromPlex(user->ceed, projection->dm, domain_label, label_value, height, dm_field, basis_div_diff_flux));
      *eval_mode_diff_flux = CEED_EVAL_INTERP;

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
      }
    } break;
    case DIV_DIFF_FLUX_PROJ_INDIRECT: {
      projection->num_comp = diff_flux_proj->num_diff_flux_comps * dim;
      PetscCall(PetscObjectSetName((PetscObject)projection->dm, "DiffFluxProj"));
      PetscCall(
          DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, user->app_ctx->degree, 1, user->app_ctx->q_extra, 1, &projection->num_comp, projection->dm));

      PetscCall(DMGetLocalSection(projection->dm, &section));
      PetscCall(PetscSectionSetFieldName(section, 0, ""));
      PetscCall(PetscSectionSetComponentName(section, 0, 0, "DiffusiveFlux_MomentumXX"));
      PetscCall(PetscSectionSetComponentName(section, 0, 1, "DiffusiveFlux_MomentumXY"));
      PetscCall(PetscSectionSetComponentName(section, 0, 2, "DiffusiveFlux_MomentumXZ"));
      PetscCall(PetscSectionSetComponentName(section, 0, 3, "DiffusiveFlux_MomentumYX"));
      PetscCall(PetscSectionSetComponentName(section, 0, 4, "DiffusiveFlux_MomentumYY"));
      PetscCall(PetscSectionSetComponentName(section, 0, 5, "DiffusiveFlux_MomentumYZ"));
      PetscCall(PetscSectionSetComponentName(section, 0, 6, "DiffusiveFlux_MomentumZX"));
      PetscCall(PetscSectionSetComponentName(section, 0, 7, "DiffusiveFlux_MomentumZY"));
      PetscCall(PetscSectionSetComponentName(section, 0, 8, "DiffusiveFlux_MomentumZZ"));
      PetscCall(PetscSectionSetComponentName(section, 0, 9, "DiffusiveFlux_EnergyX"));
      PetscCall(PetscSectionSetComponentName(section, 0, 10, "DiffusiveFlux_EnergyY"));
      PetscCall(PetscSectionSetComponentName(section, 0, 11, "DiffusiveFlux_EnergyZ"));

      PetscCall(DMPlexCeedElemRestrictionQDataCreate(user->ceed, projection->dm, domain_label, label_value, height,
                                                     diff_flux_proj->num_diff_flux_comps, elem_restr_div_diff_flux));
      PetscCallCeed(user->ceed, CeedElemRestrictionCreateVector(*elem_restr_div_diff_flux, &diff_flux_proj->div_diff_flux_ceed, NULL));
      *basis_div_diff_flux = CEED_BASIS_NONE;
      *eval_mode_diff_flux = CEED_EVAL_NONE;
    } break;
    case DIV_DIFF_FLUX_PROJ_NONE:
      SETERRQ(PetscObjectComm((PetscObject)user->dm), PETSC_ERR_ARG_WRONG, "Should not reach here with div_diff_flux_projection_method %s",
              DivDiffFluxProjectionMethods[user->app_ctx->divFdiffproj_method]);
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
};

/**
  @brief Setup direct projection of divergence of diffusive flux

  @param[in] ceed      `Ceed` context
  @param[in] user      `User` context
  @param[in] ceed_data `CeedData` context
  @param[in] problem   `ProblemData` context
**/
static PetscErrorCode DivDiffFluxProjectionSetup_Direct(Ceed ceed, User user, CeedData ceed_data, ProblemData problem) {
  DivDiffFluxProjectionData diff_flux_proj = user->diff_flux_proj;
  NodalProjectionData       projection     = diff_flux_proj->projection;
  CeedOperator              op_rhs;
  CeedBasis                 basis_diff_flux;
  CeedElemRestriction       elem_restr_diff_flux_volume, elem_restr_qd;
  CeedVector                q_data;
  CeedInt                   num_comp_q, q_data_size;
  PetscInt                  dim, label_value = 0;
  DMLabel                   domain_label = NULL;

  PetscFunctionBeginUser;
  // -- Get Pre-requisite things
  PetscCall(DMGetDimension(projection->dm, &dim));
  PetscCallCeed(ceed, CeedBasisGetNumComponents(ceed_data->basis_q, &num_comp_q));

  {  // Get elem_restr_diff_flux and basis_diff_flux
    CeedOperator     *sub_ops;
    CeedOperatorField op_field;
    PetscInt          sub_op_index = 0;  // will be 0 for the volume op

    PetscCallCeed(ceed, CeedCompositeOperatorGetSubList(user->op_ifunction, &sub_ops));
    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_ops[sub_op_index], "div F_diff", &op_field));
    PetscCallCeed(ceed, CeedOperatorFieldGetElemRestriction(op_field, &elem_restr_diff_flux_volume));
    PetscCallCeed(ceed, CeedOperatorFieldGetBasis(op_field, &basis_diff_flux));
  }
  PetscCall(QDataGet(ceed, projection->dm, domain_label, label_value, ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord, &elem_restr_qd,
                     &q_data, &q_data_size));

  PetscCallCeed(ceed, CeedCompositeOperatorCreate(ceed, &op_rhs));
  {  // Add the volume integral CeedOperator
    CeedQFunction qf_rhs_volume;
    CeedOperator  op_rhs_volume;

    switch (user->phys->state_var) {
      case STATEVAR_PRIMITIVE:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DivDiffusiveFluxVolumeRHS_Prim, DivDiffusiveFluxVolumeRHS_Prim_loc, &qf_rhs_volume));
        break;
      case STATEVAR_CONSERVATIVE:
        PetscCallCeed(ceed,
                      CeedQFunctionCreateInterior(ceed, 1, DivDiffusiveFluxVolumeRHS_Conserv, DivDiffusiveFluxVolumeRHS_Conserv_loc, &qf_rhs_volume));
        break;
      case STATEVAR_ENTROPY:
        PetscCallCeed(ceed,
                      CeedQFunctionCreateInterior(ceed, 1, DivDiffusiveFluxVolumeRHS_Entropy, DivDiffusiveFluxVolumeRHS_Entropy_loc, &qf_rhs_volume));
        break;
    }

    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_rhs_volume, problem->apply_vol_ifunction.qfctx));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_volume, "q", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_volume, "Grad_q", num_comp_q * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_volume, "qdata", q_data_size, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_rhs_volume, "diffusive flux RHS", projection->num_comp * dim, CEED_EVAL_GRAD));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_rhs_volume, NULL, NULL, &op_rhs_volume));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_volume, "q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_volume, "Grad_q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_volume, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_volume, "diffusive flux RHS", elem_restr_diff_flux_volume, basis_diff_flux, CEED_VECTOR_ACTIVE));

    PetscCallCeed(ceed, CeedCompositeOperatorAddSub(op_rhs, op_rhs_volume));

    PetscCallCeed(ceed, CeedOperatorDestroy(&op_rhs_volume));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_rhs_volume));
  }

  {  // Add the boundary integral CeedOperator
    CeedQFunction qf_rhs_boundary;
    DMLabel       face_sets_label;
    PetscInt      num_face_set_values, *face_set_values;
    CeedInt       q_data_size;

    // -- Build RHS operator
    switch (user->phys->state_var) {
      case STATEVAR_PRIMITIVE:
        PetscCallCeed(ceed,
                      CeedQFunctionCreateInterior(ceed, 1, DivDiffusiveFluxBoundaryRHS_Prim, DivDiffusiveFluxBoundaryRHS_Prim_loc, &qf_rhs_boundary));
        break;
      case STATEVAR_CONSERVATIVE:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DivDiffusiveFluxBoundaryRHS_Conserv, DivDiffusiveFluxBoundaryRHS_Conserv_loc,
                                                        &qf_rhs_boundary));
        break;
      case STATEVAR_ENTROPY:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DivDiffusiveFluxBoundaryRHS_Entropy, DivDiffusiveFluxBoundaryRHS_Entropy_loc,
                                                        &qf_rhs_boundary));
        break;
    }

    PetscCall(QDataBoundaryGradientGetNumComponents(user->dm, &q_data_size));
    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_rhs_boundary, problem->apply_vol_ifunction.qfctx));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_boundary, "q", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_boundary, "Grad_q", num_comp_q * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_boundary, "qdata", q_data_size, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_rhs_boundary, "diffusive flux RHS", projection->num_comp, CEED_EVAL_INTERP));

    PetscCall(DMGetLabel(projection->dm, "Face Sets", &face_sets_label));
    PetscCall(DMLabelCreateGlobalValueArray(projection->dm, face_sets_label, &num_face_set_values, &face_set_values));
    for (PetscInt f = 0; f < num_face_set_values; f++) {
      DMLabel  face_orientation_label;
      PetscInt num_orientations_values, *orientation_values;

      {
        char *face_orientation_label_name;

        PetscCall(DMPlexCreateFaceLabel(projection->dm, face_set_values[f], &face_orientation_label_name));
        PetscCall(DMGetLabel(projection->dm, face_orientation_label_name, &face_orientation_label));
        PetscCall(PetscFree(face_orientation_label_name));
      }
      PetscCall(DMLabelCreateGlobalValueArray(projection->dm, face_orientation_label, &num_orientations_values, &orientation_values));
      for (PetscInt o = 0; o < num_orientations_values; o++) {
        CeedOperator        op_rhs_boundary;
        CeedBasis           basis_q, basis_diff_flux_boundary;
        CeedElemRestriction elem_restr_qdata, elem_restr_q, elem_restr_diff_flux_boundary;
        CeedVector          q_data;
        CeedInt             q_data_size;
        PetscInt            orientation = orientation_values[o], dm_field_q = 0, height_cell = 0, height_face = 1;

        PetscCall(DMPlexCeedElemRestrictionCreate(ceed, user->dm, face_orientation_label, orientation, height_cell, dm_field_q, &elem_restr_q));
        PetscCall(DMPlexCeedElemRestrictionCreate(ceed, projection->dm, face_orientation_label, orientation, height_face, 0,
                                                  &elem_restr_diff_flux_boundary));
        PetscCall(QDataBoundaryGradientGet(ceed, user->dm, face_orientation_label, orientation, ceed_data->x_coord, &elem_restr_qdata, &q_data,
                                           &q_data_size));
        PetscCall(DMPlexCeedBasisCellToFaceCreate(ceed, user->dm, face_orientation_label, orientation, orientation, dm_field_q, &basis_q));
        PetscCall(CreateBasisFromPlex(ceed, projection->dm, face_orientation_label, orientation, height_face, 0, &basis_diff_flux_boundary));

        PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_rhs_boundary, NULL, NULL, &op_rhs_boundary));
        PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_boundary, "q", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
        PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_boundary, "Grad_q", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
        PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_boundary, "qdata", elem_restr_qdata, CEED_BASIS_NONE, q_data));
        PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_boundary, "diffusive flux RHS", elem_restr_diff_flux_boundary, basis_diff_flux_boundary,
                                                 CEED_VECTOR_ACTIVE));

        PetscCallCeed(ceed, CeedCompositeOperatorAddSub(op_rhs, op_rhs_boundary));

        PetscCallCeed(ceed, CeedOperatorDestroy(&op_rhs_boundary));
        PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qdata));
        PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q));
        PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_diff_flux_boundary));
        PetscCallCeed(ceed, CeedBasisDestroy(&basis_q));
        PetscCallCeed(ceed, CeedBasisDestroy(&basis_diff_flux_boundary));
        PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
      }
      PetscCall(PetscFree(orientation_values));
    }
    PetscCall(PetscFree(face_set_values));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_rhs_boundary));
  }

  PetscCall(DMCreateLocalVector(projection->dm, &diff_flux_proj->DivDiffFlux_loc));
  diff_flux_proj->ceed_vec_has_array = PETSC_FALSE;
  PetscCall(
      OperatorApplyContextCreate(user->dm, projection->dm, ceed, op_rhs, NULL, NULL, NULL, diff_flux_proj->DivDiffFlux_loc, &projection->l2_rhs_ctx));

  {  // -- Build Mass operator
    CeedQFunction qf_mass;
    CeedOperator  op_mass;

    PetscCall(CreateMassQFunction(ceed, projection->num_comp, q_data_size, &qf_mass));
    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass));
    PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "u", elem_restr_diff_flux_volume, basis_diff_flux, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "v", elem_restr_diff_flux_volume, basis_diff_flux, CEED_VECTOR_ACTIVE));

    {  // -- Setup KSP for L^2 projection
      Mat      mat_mass;
      MPI_Comm comm = PetscObjectComm((PetscObject)projection->dm);

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
  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_rhs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Setup indirect projection of divergence of diffusive flux

  @param[in]     ceed      `Ceed` context
  @param[in,out] user      `User` context
  @param[in]     ceed_data `CeedData` context
  @param[in]     problem   `ProblemData` context
**/
static PetscErrorCode DivDiffFluxProjectionSetup_Indirect(Ceed ceed, User user, CeedData ceed_data, ProblemData problem) {
  DivDiffFluxProjectionData diff_flux_proj = user->diff_flux_proj;
  NodalProjectionData       projection     = diff_flux_proj->projection;
  CeedBasis                 basis_diff_flux;
  CeedElemRestriction       elem_restr_diff_flux, elem_restr_qd;
  CeedVector                q_data;
  CeedInt                   num_comp_q, q_data_size;
  PetscInt                  dim;
  PetscInt                  label_value = 0, height = 0, dm_field = 0;
  DMLabel                   domain_label = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(projection->dm, &dim));
  PetscCallCeed(ceed, CeedBasisGetNumComponents(ceed_data->basis_q, &num_comp_q));

  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, projection->dm, domain_label, label_value, height, dm_field, &elem_restr_diff_flux));
  PetscCall(CreateBasisFromPlex(ceed, projection->dm, domain_label, label_value, height, dm_field, &basis_diff_flux));
  PetscCall(QDataGet(ceed, projection->dm, domain_label, label_value, ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord, &elem_restr_qd,
                     &q_data, &q_data_size));

  {  // Create RHS CeedOperator for L^2 projection
    CeedQFunction qf_rhs;
    CeedOperator  op_rhs;

    switch (user->phys->state_var) {
      case STATEVAR_PRIMITIVE:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DiffusiveFluxRHS_Prim, DiffusiveFluxRHS_Prim_loc, &qf_rhs));
        break;
      case STATEVAR_CONSERVATIVE:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DiffusiveFluxRHS_Conserv, DiffusiveFluxRHS_Conserv_loc, &qf_rhs));
        break;
      case STATEVAR_ENTROPY:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DiffusiveFluxRHS_Entropy, DiffusiveFluxRHS_Entropy_loc, &qf_rhs));
        break;
    }

    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_rhs, problem->apply_vol_ifunction.qfctx));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs, "q", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs, "Grad_q", num_comp_q * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs, "qdata", q_data_size, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_rhs, "F_diff RHS", projection->num_comp, CEED_EVAL_INTERP));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_rhs, NULL, NULL, &op_rhs));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs, "q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs, "Grad_q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs, "F_diff RHS", elem_restr_diff_flux, basis_diff_flux, CEED_VECTOR_ACTIVE));

    PetscCall(OperatorApplyContextCreate(user->dm, projection->dm, ceed, op_rhs, NULL, NULL, NULL, NULL, &projection->l2_rhs_ctx));

    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_rhs));
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
      Mat      mat_mass;
      MPI_Comm comm = PetscObjectComm((PetscObject)projection->dm);

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
    CeedQFunction       qf_calc_divergence;
    CeedOperator        op_calc_divergence;
    CeedElemRestriction elem_restr_div_diff_flux;

    {  // Get elem_restr_div_diff_flux
      CeedOperator     *sub_ops;
      CeedOperatorField op_field;
      PetscInt          sub_op_index = 0;  // will be 0 for the volume op

      PetscCallCeed(ceed, CeedCompositeOperatorGetSubList(user->op_ifunction, &sub_ops));
      PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_ops[sub_op_index], "div F_diff", &op_field));
      PetscCallCeed(ceed, CeedOperatorFieldGetElemRestriction(op_field, &elem_restr_div_diff_flux));
    }

    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ComputeDivDiffusiveFlux3D_4, ComputeDivDiffusiveFlux3D_4_loc, &qf_calc_divergence));

    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_calc_divergence, "Grad F_diff", projection->num_comp * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_calc_divergence, "qdata", q_data_size, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_calc_divergence, "Div F_diff", 4, CEED_EVAL_NONE));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_calc_divergence, NULL, NULL, &op_calc_divergence));
    PetscCallCeed(ceed, CeedOperatorSetField(op_calc_divergence, "Grad F_diff", elem_restr_diff_flux, basis_diff_flux, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_calc_divergence, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
    PetscCallCeed(
        ceed, CeedOperatorSetField(op_calc_divergence, "Div F_diff", elem_restr_div_diff_flux, CEED_BASIS_NONE, diff_flux_proj->div_diff_flux_ceed));

    PetscCall(OperatorApplyContextCreate(projection->dm, NULL, ceed, op_calc_divergence, NULL, NULL, NULL, NULL,
                                         &user->diff_flux_proj->calc_div_diff_flux));

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

  @param[in]     ceed      `Ceed` context
  @param[in,out] user      `User` context
  @param[in]     ceed_data `CeedData` context
  @param[in]     problem   `ProblemData` context
**/
PetscErrorCode DivDiffFluxProjectionSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData problem) {
  PetscFunctionBeginUser;
  switch (user->app_ctx->divFdiffproj_method) {
    case DIV_DIFF_FLUX_PROJ_DIRECT:
      PetscCall(DivDiffFluxProjectionSetup_Direct(ceed, user, ceed_data, problem));
      break;
    case DIV_DIFF_FLUX_PROJ_INDIRECT:
      PetscCall(DivDiffFluxProjectionSetup_Indirect(ceed, user, ceed_data, problem));
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
PetscErrorCode DiffFluxProjectionApply(DivDiffFluxProjectionData diff_flux_proj, Vec Q_loc) {
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
  PetscCall(NodalProjectionDataDestroy(diff_flux_proj->projection));
  PetscCall(OperatorApplyContextDestroy(diff_flux_proj->calc_div_diff_flux));
  if (diff_flux_proj->ceed_vec_has_array) {
    PetscCall(VecReadCeedToPetsc(diff_flux_proj->div_diff_flux_ceed, diff_flux_proj->DivDiffFlux_memtype, diff_flux_proj->DivDiffFlux_loc));
    diff_flux_proj->ceed_vec_has_array = PETSC_FALSE;
  }
  PetscCall(CeedVectorDestroy(&diff_flux_proj->div_diff_flux_ceed));
  PetscCall(VecDestroy(&diff_flux_proj->DivDiffFlux_loc));
  PetscCall(PetscFree(diff_flux_proj));
  PetscFunctionReturn(PETSC_SUCCESS);
}
