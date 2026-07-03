// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Utility functions for setting up ADVECTION

#include "../qfunctions/advection.h"

#include <ceed.h>
#include <ceed/backend.h>
#include <petscdm.h>

#include <navierstokes.h>

const char *const AdvDifWindTypes[] = {"ROTATION", "TRANSLATION", "BOUNDARY_LAYER", "WindType", "ADVDIF_WIND_", NULL};
const char *const AdvDifICTypes[]   = {"SPHERE", "CYLINDER", "COSINE_HILL", "SKEW", "WAVE", "BOUNDARY_LAYER", "AdvectionICType", "ADVDIF_IC_", NULL};
const char *const AdvDifWaveTypes[] = {"SINE", "SQUARE", "AdvDifWaveType", "ADVDIF_WAVE_", NULL};
const char *const AdvDifBubbleContinuityTypes[] = {"SMOOTH", "BACK_SHARP", "THICK", "COSINE", "BubbleContinuityType", "ADVDIF_BUBBLE_CONTINUITY_",
                                                   NULL};
const char *const StabilizationTauTypes[]       = {"CTAU", "ADVDIFF_SHAKIB", "StabilizationTauType", "STAB_TAU_", NULL};

static PetscErrorCode PRINT_ADVECTION(Honee honee, ProblemData problem, AppCtx app_ctx) {
  MPI_Comm         comm = honee->comm;
  Ceed             ceed = honee->ceed;
  SetupContextAdv  setup_ctx;
  AdvectionContext advection_ctx;
  PetscInt         dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(honee->dm, &dim));
  PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->ics.qfctx, CEED_MEM_HOST, &setup_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->apply_vol_rhs.qfctx, CEED_MEM_HOST, &advection_ctx));
  PetscCall(PetscPrintf(comm,
                        "  Problem:\n"
                        "    Problem Name                       : %s\n"
                        "    Stabilization                      : %s\n"
                        "    Stabilization Tau                  : %s\n"
                        "    Wind Type                          : %s\n",
                        app_ctx->problem_name, StabilizationTypes[advection_ctx->stabilization],
                        StabilizationTauTypes[advection_ctx->stabilization_tau], AdvDifWindTypes[setup_ctx->wind_type]));

  if (setup_ctx->wind_type == ADVDIF_WIND_TRANSLATION) {
    CeedScalar *wind = setup_ctx->wind;
    switch (dim) {
      case 2:
        PetscCall(PetscPrintf(comm, "    Background Wind                    : %f,%f\n", wind[0], wind[1]));
        break;
      case 3:
        PetscCall(PetscPrintf(comm, "    Background Wind                    : %f,%f,%f\n", wind[0], wind[1], wind[2]));
        break;
    }
  }

  PetscCall(PetscPrintf(comm, "    Initial Condition Type             : %s\n", AdvDifICTypes[setup_ctx->initial_condition_type]));
  switch (setup_ctx->initial_condition_type) {
    case ADVDIF_IC_SKEW:
    case ADVDIF_IC_COSINE_HILL:
    case ADVDIF_IC_BOUNDARY_LAYER:
      break;
    case ADVDIF_IC_BUBBLE_SPHERE:
    case ADVDIF_IC_BUBBLE_CYLINDER:
      PetscCall(PetscPrintf(comm, "    Bubble Continuity                  : %s\n", AdvDifBubbleContinuityTypes[setup_ctx->bubble_continuity_type]));
      break;
    case ADVDIF_IC_WAVE:
      PetscCall(PetscPrintf(comm, "    Wave Type                          : %s\n", AdvDifWaveTypes[setup_ctx->wave_type]));
      break;
  }

  PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->ics.qfctx, &setup_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfctx, &advection_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Create CeedOperator for stabilized mass KSP for explicit timestepping
//
// Only used for SUPG stabilization
PetscErrorCode CreateKSPMassOperator_AdvectionStabilized(Honee honee, CeedOperator *op_mass) {
  Ceed                 ceed = honee->ceed;
  CeedInt              num_comp_q, q_data_size;
  CeedQFunction        qf_mass = NULL;
  CeedElemRestriction  elem_restr_q, elem_restr_qd;
  CeedBasis            basis_q;
  CeedVector           q_data;
  CeedQFunctionContext qfctx = NULL;
  PetscInt             dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(honee->dm, &dim));
  {  // Get restriction and basis from the RHS function
    CeedOperator     *sub_ops;
    CeedOperatorField op_field;
    PetscInt          sub_op_index = 0;  // will be 0 for the volume op

    PetscCallCeed(ceed, CeedOperatorCompositeGetSubList(honee->op_rhs_ctx->op, &sub_ops));
    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_ops[sub_op_index], "q", &op_field));
    PetscCallCeed(ceed, CeedOperatorFieldGetData(op_field, NULL, &elem_restr_q, &basis_q, NULL));
    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_ops[sub_op_index], "qdata", &op_field));
    PetscCallCeed(ceed, CeedOperatorFieldGetData(op_field, NULL, &elem_restr_qd, NULL, &q_data));

    PetscCallCeed(ceed, CeedOperatorGetContext(sub_ops[sub_op_index], &qfctx));
  }

  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_q, &num_comp_q));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_qd, &q_data_size));

  switch (dim) {
    case 2:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, MassFunction_Advection2D, MassFunction_Advection2D_loc, &qf_mass));
      break;
    case 3:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, MassFunction_Advection, MassFunction_Advection_loc, &qf_mass));
      break;
  }

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_mass, qfctx));
  PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(qf_mass, 0));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_mass, "q_dot", 5, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_mass, "q", 5, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_mass, "qdata", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_mass, "v", 5, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_mass, "Grad_v", 5 * dim, CEED_EVAL_GRAD));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, op_mass));
  PetscCallCeed(ceed, CeedOperatorSetName(*op_mass, "RHS Mass Operator, Advection-Diffusion Stabilized"));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "q_dot", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "q", elem_restr_q, basis_q, honee->q_ceed));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "v", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "Grad_v", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_q));
  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&qfctx));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_mass));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create RHS CeedOperator for direct projection of divergence of diffusive flux

  @param[in]  honee          `Honee` context
  @param[in]  diff_flux_proj `DivDiffFluxProjectionData` object
  @param[out] op_rhs         Operator to calculate the RHS of the L^2 projection
**/
static PetscErrorCode DivDiffFluxProjectionCreateRHS_Direct_AdvDif(Honee honee, DivDiffFluxProjectionData diff_flux_proj, CeedOperator *op_rhs) {
  Ceed                 ceed       = honee->ceed;
  NodalProjectionData  projection = diff_flux_proj->projection;
  PetscInt             dim, num_comp_q;
  CeedQFunctionContext advection_qfctx = NULL;

  PetscFunctionBeginUser;
  // -- Get Pre-requisite things
  PetscCall(DMGetDimension(projection->dm, &dim));
  PetscCall(DMGetFieldNumComps(honee->dm, 0, &num_comp_q));

  {  // Get advection-diffusion QF context
    CeedOperator *sub_ops;
    PetscInt      sub_op_index = 0;  // will be 0 for the volume op

    if (honee->op_ifunction) PetscCallCeed(ceed, CeedOperatorCompositeGetSubList(honee->op_ifunction, &sub_ops));
    else PetscCallCeed(ceed, CeedOperatorCompositeGetSubList(honee->op_rhs_ctx->op, &sub_ops));
    PetscCallCeed(ceed, CeedOperatorGetContext(sub_ops[sub_op_index], &advection_qfctx));
  }
  PetscCallCeed(ceed, CeedOperatorCreateComposite(ceed, op_rhs));
  {  // Add the volume integral CeedOperator
    CeedQFunction       qf_rhs_volume = NULL;
    CeedOperator        op_rhs_volume;
    CeedVector          q_data;
    CeedElemRestriction elem_restr_qd, elem_restr_diff_flux_volume = NULL, elem_restr_q;
    CeedBasis           basis_diff_flux = NULL, basis_q;
    CeedInt             q_data_size;

    PetscCall(DMPlexCeedElemRestrictionCreate(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, PLEXHEIGHT_CELL, 0, &elem_restr_q));
    PetscCall(DMPlexCeedBasisCreate(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, PLEXHEIGHT_CELL, 0, &basis_q));
    PetscCall(DivDiffFluxProjectionGetOperatorFieldData(diff_flux_proj, &elem_restr_diff_flux_volume, &basis_diff_flux, NULL, NULL));
    PetscCall(QDataGet(ceed, projection->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, &elem_restr_qd, &q_data, &q_data_size));
    switch (dim) {
      case 2:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DivDiffusiveFluxVolumeRHS_AdvDif_2D, DivDiffusiveFluxVolumeRHS_AdvDif_2D_loc,
                                                        &qf_rhs_volume));
        break;
      case 3:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DivDiffusiveFluxVolumeRHS_AdvDif_3D, DivDiffusiveFluxVolumeRHS_AdvDif_3D_loc,
                                                        &qf_rhs_volume));
        break;
    }
    PetscCheck(qf_rhs_volume, honee->comm, PETSC_ERR_SUP, "%s not valid for DM of dimension %" PetscInt_FMT, __func__, dim);

    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_rhs_volume, advection_qfctx));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_volume, "Grad_q", num_comp_q * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_volume, "qdata", q_data_size, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_rhs_volume, "diffusive flux RHS", projection->num_comp * dim, CEED_EVAL_GRAD));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_rhs_volume, NULL, NULL, &op_rhs_volume));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_volume, "Grad_q", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_volume, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_volume, "diffusive flux RHS", elem_restr_diff_flux_volume, basis_diff_flux, CEED_VECTOR_ACTIVE));

    PetscCallCeed(ceed, CeedOperatorCompositeAddSub(*op_rhs, op_rhs_volume));

    PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q));
    PetscCallCeed(ceed, CeedBasisDestroy(&basis_q));
    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_diff_flux_volume));
    PetscCallCeed(ceed, CeedBasisDestroy(&basis_diff_flux));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_rhs_volume));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_rhs_volume));
  }

  {  // Add the boundary integral CeedOperator
    CeedQFunction   qf_rhs_boundary;
    DMLabel         face_sets_label;
    IS              face_sets_is;
    PetscInt        num_face_set_values;
    const PetscInt *face_set_values;
    CeedInt         q_data_size;

    // -- Build RHS operator
    switch (dim) {
      case 2:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DivDiffusiveFluxBoundaryRHS_AdvDif_2D, DivDiffusiveFluxBoundaryRHS_AdvDif_2D_loc,
                                                        &qf_rhs_boundary));
        break;
      case 3:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DivDiffusiveFluxBoundaryRHS_AdvDif_3D, DivDiffusiveFluxBoundaryRHS_AdvDif_3D_loc,
                                                        &qf_rhs_boundary));
        break;
    }

    PetscCall(QDataBoundaryGradientGetNumComponents(honee->dm, &q_data_size));
    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_rhs_boundary, advection_qfctx));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_boundary, "Grad_q", num_comp_q * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_boundary, "qdata", q_data_size, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_rhs_boundary, "diffusive flux RHS", projection->num_comp, CEED_EVAL_INTERP));

    PetscCall(DMGetLabel(projection->dm, "Face Sets", &face_sets_label));
    PetscCall(DMLabelGetValueISGlobal(PetscObjectComm((PetscObject)projection->dm), face_sets_label, PETSC_FALSE, &face_sets_is));
    PetscCall(ISGetSize(face_sets_is, &num_face_set_values));
    PetscCall(ISGetIndices(face_sets_is, &face_set_values));
    for (PetscInt f = 0; f < num_face_set_values; f++) {
      DMLabel         face_orientation_label;
      IS              orientation_is;
      PetscInt        num_orientations_values;
      const PetscInt *orientation_values;

      {
        char *face_orientation_label_name;

        PetscCall(DMPlexCreateFaceLabel(projection->dm, face_set_values[f], &face_orientation_label_name));
        PetscCall(DMGetLabel(projection->dm, face_orientation_label_name, &face_orientation_label));
        PetscCall(PetscFree(face_orientation_label_name));
      }
      PetscCall(DMLabelGetValueISGlobal(PetscObjectComm((PetscObject)projection->dm), face_orientation_label, PETSC_FALSE, &orientation_is));
      PetscCall(ISGetSize(orientation_is, &num_orientations_values));
      PetscCall(ISGetIndices(orientation_is, &orientation_values));
      for (PetscInt o = 0; o < num_orientations_values; o++) {
        CeedOperator        op_rhs_boundary;
        CeedBasis           basis_q, basis_diff_flux_boundary;
        CeedElemRestriction elem_restr_qdata, elem_restr_q, elem_restr_diff_flux_boundary;
        CeedVector          q_data;
        CeedInt             q_data_size;
        PetscInt            orientation = orientation_values[o], dm_field_q = 0;

        PetscCall(DMPlexCeedElemRestrictionCreate(ceed, honee->dm, face_orientation_label, orientation, PLEXHEIGHT_CELL, dm_field_q, &elem_restr_q));
        PetscCall(DMPlexCeedBasisCellToFaceCreate(ceed, honee->dm, face_orientation_label, orientation, orientation, dm_field_q, &basis_q));
        PetscCall(DMPlexCeedElemRestrictionCreate(ceed, projection->dm, face_orientation_label, orientation, PLEXHEIGHT_FACE, 0,
                                                  &elem_restr_diff_flux_boundary));
        PetscCall(DMPlexCeedBasisCreate(ceed, projection->dm, face_orientation_label, orientation, PLEXHEIGHT_FACE, 0, &basis_diff_flux_boundary));
        PetscCall(QDataBoundaryGradientGet(ceed, honee->dm, face_orientation_label, orientation, &elem_restr_qdata, &q_data, &q_data_size));

        PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_rhs_boundary, NULL, NULL, &op_rhs_boundary));
        PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_boundary, "Grad_q", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
        PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_boundary, "qdata", elem_restr_qdata, CEED_BASIS_NONE, q_data));
        PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_boundary, "diffusive flux RHS", elem_restr_diff_flux_boundary, basis_diff_flux_boundary,
                                                 CEED_VECTOR_ACTIVE));

        PetscCallCeed(ceed, CeedOperatorCompositeAddSub(*op_rhs, op_rhs_boundary));

        PetscCallCeed(ceed, CeedOperatorDestroy(&op_rhs_boundary));
        PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qdata));
        PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q));
        PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_diff_flux_boundary));
        PetscCallCeed(ceed, CeedBasisDestroy(&basis_q));
        PetscCallCeed(ceed, CeedBasisDestroy(&basis_diff_flux_boundary));
        PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
      }
      PetscCall(ISRestoreIndices(orientation_is, &orientation_values));
      PetscCall(ISDestroy(&orientation_is));
    }
    PetscCall(ISRestoreIndices(face_sets_is, &face_set_values));
    PetscCall(ISDestroy(&face_sets_is));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_rhs_boundary));
  }

  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&advection_qfctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create RHS CeedOperator for indirect projection of divergence of diffusive flux

  @param[in]  honee          `Honee` context
  @param[in]  diff_flux_proj `DivDiffFluxProjectionData` object
  @param[out] op_rhs         Operator to calculate the RHS of the L^2 projection
**/
static PetscErrorCode DivDiffFluxProjectionCreateRHS_Indirect_AdvDif(Honee honee, DivDiffFluxProjectionData diff_flux_proj, CeedOperator *op_rhs) {
  Ceed                 ceed       = honee->ceed;
  NodalProjectionData  projection = diff_flux_proj->projection;
  CeedBasis            basis_diff_flux, basis_q;
  CeedElemRestriction  elem_restr_diff_flux, elem_restr_qd, elem_restr_q;
  CeedVector           q_data;
  CeedInt              q_data_size;
  PetscInt             dim, num_comp_q;
  PetscInt             dm_field        = 0;
  CeedQFunction        qf_rhs          = NULL;
  CeedQFunctionContext advection_qfctx = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(projection->dm, &dim));
  PetscCall(DMGetFieldNumComps(honee->dm, 0, &num_comp_q));

  {  // Get advection-diffusion QF context
    CeedOperator *sub_ops;
    PetscInt      sub_op_index = 0;  // will be 0 for the volume op

    if (honee->op_ifunction) PetscCallCeed(ceed, CeedOperatorCompositeGetSubList(honee->op_ifunction, &sub_ops));
    else PetscCallCeed(ceed, CeedOperatorCompositeGetSubList(honee->op_rhs_ctx->op, &sub_ops));
    PetscCallCeed(ceed, CeedOperatorGetContext(sub_ops[sub_op_index], &advection_qfctx));
  }
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, PLEXHEIGHT_CELL, 0, &elem_restr_q));
  PetscCall(DMPlexCeedBasisCreate(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, PLEXHEIGHT_CELL, 0, &basis_q));
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, projection->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, PLEXHEIGHT_CELL, dm_field,
                                            &elem_restr_diff_flux));
  PetscCall(DMPlexCeedBasisCreate(ceed, projection->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, PLEXHEIGHT_CELL, dm_field, &basis_diff_flux));
  PetscCall(QDataGet(ceed, projection->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, &elem_restr_qd, &q_data, &q_data_size));

  switch (dim) {
    case 2:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DiffusiveFluxRHS_AdvDif_2D, DiffusiveFluxRHS_AdvDif_2D_loc, &qf_rhs));
      break;
    case 3:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DiffusiveFluxRHS_AdvDif_3D, DiffusiveFluxRHS_AdvDif_3D_loc, &qf_rhs));
      break;
  }
  PetscCheck(qf_rhs, honee->comm, PETSC_ERR_SUP, "%s not valid for DM of dimension %" PetscInt_FMT, __func__, dim);

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_rhs, advection_qfctx));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs, "Grad_q", num_comp_q * dim, CEED_EVAL_GRAD));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs, "qdata", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_rhs, "F_diff RHS", projection->num_comp, CEED_EVAL_INTERP));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_rhs, NULL, NULL, op_rhs));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_rhs, "Grad_q", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_rhs, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_rhs, "F_diff RHS", elem_restr_diff_flux, basis_diff_flux, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_rhs));
  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&advection_qfctx));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_q));
  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_diff_flux));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_diff_flux));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AdvectionInflowBCSetup_CreateIFunctionQF(BCDefinition bc_def, CeedQFunction *qf) {
  HoneeBCStruct honee_bc;
  DM            dm;
  PetscInt      dim;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  PetscCall(BCDefinitionGetDM(bc_def, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  switch (dim) {
    case 2:
      PetscCall(HoneeBCCreateIFunctionQF(bc_def, Advection2d_InOutFlow, Advection2d_InOutFlow_loc, honee_bc->qfctx, qf));
      break;
    case 3:
      PetscCall(HoneeBCCreateIFunctionQF(bc_def, Advection_InOutFlow, Advection_InOutFlow_loc, honee_bc->qfctx, qf));
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define ADVECTION_QFCTX_KEY "Advection QFunction Ctx"

static PetscErrorCode BCDefinitionNaturalSetup_Advection(BCDefinition bc_def, void *ctx) {
  HoneeBCStruct        honee_bc = (HoneeBCStruct)ctx;
  Honee                honee    = honee_bc->honee;
  Ceed                 ceed     = honee->ceed;
  CeedQFunctionContext advection_qfctx;

  PetscFunctionBeginUser;
  PetscCall(HoneeGetContainer(honee, ADVECTION_QFCTX_KEY, &advection_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(advection_qfctx, &honee_bc->qfctx));
  honee_bc->num_comps_jac_data = 0;
  PetscCall(BCDefinitionSetIFunction(bc_def, AdvectionInflowBCSetup_CreateIFunctionQF, HoneeBCAddIFunctionOp));
  PetscCall(BCDefinitionSetIJacobian(bc_def, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static const char *const component_names[] = {"Density", "MomentumX", "MomentumY", "MomentumZ", "TotalEnergy"};

PetscErrorCode NS_ADVECTION(ProblemData problem, DM dm, void *ctx) {
  AdvDifWindType       wind_type;
  AdvDifICType         advectionic_type;
  StabilizationType    stab;
  StabilizationTauType stab_tau;
  SetupContextAdv      setup_context;
  Honee                honee = *(Honee *)ctx;
  MPI_Comm             comm  = honee->comm;
  Ceed                 ceed  = honee->ceed;
  PetscBool            implicit;
  AdvectionContext     advection_ctx;
  CeedQFunctionContext advection_qfctx, ics_qfctx;
  PetscInt             dim;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&setup_context));
  PetscCall(PetscNew(&advection_ctx));
  PetscCall(DMGetDimension(dm, &dim));
  AdvDifBubbleContinuityType bubble_continuity_type = dim == 3 ? ADVDIF_BUBBLE_CONTINUITY_SMOOTH : ADVDIF_BUBBLE_CONTINUITY_COSINE;

  // ------------------------------------------------------
  //             Create the QFunction context
  // ------------------------------------------------------
  CeedScalar     rc               = 1000.;  // m (Radius of bubble)
  CeedScalar     CtauS            = 0.;     // dimensionless
  PetscBool      strong_form      = PETSC_FALSE;
  CeedScalar     E_wind           = 1.e6;  // J
  CeedScalar     Ctau_a           = PetscPowScalarInt(honee->app_ctx->degree, 2);
  CeedScalar     Ctau_d           = PetscPowScalarInt(honee->app_ctx->degree, 4);
  CeedScalar     Ctau_t           = 0.;
  PetscReal      wind[3]          = {1., 0, 0};  // m/s
  CeedScalar     diffusion_coeff  = 0.;
  CeedScalar     wave_frequency   = 2 * M_PI;
  CeedScalar     wave_phase       = 0;
  AdvDifWaveType wave_type        = ADVDIF_WAVE_SINE;
  PetscScalar    bl_height_factor = 1.;
  PetscReal      domain_min[3], domain_max[3], domain_size[3] = {0.};
  PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
  for (PetscInt i = 0; i < dim; i++) domain_size[i] = domain_max[i] - domain_min[i];

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  PetscOptionsBegin(comm, NULL, "Options for ADVECTION problem", NULL);
  // -- Physics
  PetscBool translation;
  PetscCall(PetscOptionsEnum("-wind_type", "Wind type in Advection", NULL, AdvDifWindTypes, (PetscEnum)(wind_type = ADVDIF_WIND_ROTATION),
                             (PetscEnum *)&wind_type, &translation));
  PetscInt  n = dim;
  PetscBool user_wind;
  PetscCall(PetscOptionsRealArray("-wind_translation", "Constant wind vector", NULL, wind, &n, &user_wind));
  PetscCall(PetscOptionsScalar("-diffusion_coeff", "Diffusion coefficient", NULL, diffusion_coeff, &diffusion_coeff, NULL));
  PetscCall(PetscOptionsScalar("-CtauS", "Scale coefficient for tau (nondimensional)", NULL, CtauS, &CtauS, NULL));
  PetscCall(PetscOptionsBool("-strong_form", "Strong (true) or weak/integrated by parts (false) advection residual", NULL, strong_form, &strong_form,
                             NULL));
  PetscCall(PetscOptionsScalar("-E_wind", "Total energy of inflow wind", NULL, E_wind, &E_wind, NULL));
  PetscCall(PetscOptionsEnum("-stab", "Stabilization method", NULL, StabilizationTypes, (PetscEnum)(stab = STAB_NONE), (PetscEnum *)&stab, NULL));
  PetscCall(PetscOptionsEnum("-stab_tau", "Stabilization constant, tau", NULL, StabilizationTauTypes, (PetscEnum)(stab_tau = STAB_TAU_CTAU),
                             (PetscEnum *)&stab_tau, NULL));
  PetscCall(PetscOptionsScalar("-Ctau_t", "Stabilization time constant", NULL, Ctau_t, &Ctau_t, NULL));
  PetscCall(PetscOptionsScalar("-Ctau_a", "Coefficient for the stabilization, advection component", NULL, Ctau_a, &Ctau_a, NULL));
  PetscCall(PetscOptionsScalar("-Ctau_d", "Coefficient for the stabilization, diffusion component", NULL, Ctau_d, &Ctau_d, NULL));
  PetscCall(PetscOptionsBool("-implicit", "Use implicit (IFunction) formulation", NULL, implicit = PETSC_FALSE, &implicit, NULL));
  PetscCall(PetscOptionsEnum("-advection_ic_type", "Initial condition for Advection problem", NULL, AdvDifICTypes,
                             (PetscEnum)(advectionic_type = ADVDIF_IC_BUBBLE_SPHERE), (PetscEnum *)&advectionic_type, NULL));
  // IC-specific options
  switch (advectionic_type) {
    case ADVDIF_IC_WAVE:
      PetscCall(PetscOptionsDeprecated("-wave_type", "-advection_ic_wave_type", "HONEE 0.0", NULL));
      PetscCall(PetscOptionsDeprecated("-wave_frequency", "-advection_ic_wave_frequency", "HONEE 0.0", NULL));
      PetscCall(PetscOptionsDeprecated("-wave_phase", "-advection_ic_wave_phase", "HONEE 0.0", NULL));
      PetscCall(PetscOptionsEnum("-advection_ic_wave_type", "Type of wave", NULL, AdvDifWaveTypes, (PetscEnum)wave_type, (PetscEnum *)&wave_type,
                                 NULL));
      PetscCall(PetscOptionsScalar("-advection_ic_wave_frequency", "Frequency of sine wave", NULL, wave_frequency, &wave_frequency, NULL));
      PetscCall(PetscOptionsScalar("-advection_ic_wave_phase", "Length correction", NULL, wave_phase, &wave_phase, NULL));
      break;
    case ADVDIF_IC_BOUNDARY_LAYER:
      PetscCall(PetscOptionsScalar("-advection_ic_bl_height_factor", "Height of boundary layer in IC", NULL, bl_height_factor, &bl_height_factor,
                                   NULL));
      break;
    case ADVDIF_IC_BUBBLE_CYLINDER:
    case ADVDIF_IC_BUBBLE_SPHERE:
      PetscCall(PetscOptionsDeprecated("-rc", "-advection_ic_bubble_rc", "HONEE 0.0", NULL));
      PetscCall(PetscOptionsDeprecated("-bubble_continuity", "-advection_ic_bubble_continuity", "HONEE 0.0", NULL));
      PetscCall(PetscOptionsScalar("-advection_ic_bubble_rc", "Characteristic radius of thermal bubble", NULL, rc, &rc, NULL));
      PetscCall(PetscOptionsEnum("-advection_ic_bubble_continuity", "Smooth, back_sharp, or thick", NULL, AdvDifBubbleContinuityTypes,
                                 (PetscEnum)bubble_continuity_type, (PetscEnum *)&bubble_continuity_type, NULL));  // NOLINT
      break;
    case ADVDIF_IC_SKEW:
    case ADVDIF_IC_COSINE_HILL:
      break;
  }

  // -- Warnings
  if (wind_type == ADVDIF_WIND_ROTATION && user_wind) {
    PetscCall(PetscPrintf(comm, "Warning! Use -wind_translation only with -wind_type translation\n"));
  }
  if (wind_type == ADVDIF_WIND_TRANSLATION && advectionic_type == ADVDIF_IC_BUBBLE_CYLINDER && wind[2] != 0.) {
    wind[2] = 0;
    PetscCall(PetscPrintf(comm,
                          "Warning! Background wind in the z direction should be zero (-wind_translation x,x,0) with -advection_ic_type cylinder\n"));
  }
  if (stab == STAB_NONE && CtauS != 0) {
    PetscCall(PetscPrintf(comm, "Warning! Use -CtauS only with -stab su or -stab supg\n"));
  }
  PetscOptionsEnd();

  if (stab == STAB_SUPG) problem->create_mass_operator = CreateKSPMassOperator_AdvectionStabilized;

  // ------------------------------------------------------
  //           Set up the QFunction contexts
  // ------------------------------------------------------
  // -- Scale variables to desired units
  Units units = honee->units;
  E_wind *= units->Joule;
  rc = fabs(rc) * units->meter;
  for (PetscInt i = 0; i < dim; i++) {
    wind[i] *= (units->meter / units->second);
  }

  // -- Setup Context
  setup_context->rc                     = rc;
  setup_context->lx                     = domain_size[0];
  setup_context->ly                     = domain_size[1];
  setup_context->lz                     = dim == 3 ? domain_size[2] : 0.;
  setup_context->wind[0]                = wind[0];
  setup_context->wind[1]                = wind[1];
  setup_context->wind[2]                = dim == 3 ? wind[2] : 0.;
  setup_context->wind_type              = wind_type;
  setup_context->initial_condition_type = advectionic_type;
  setup_context->bubble_continuity_type = bubble_continuity_type;
  setup_context->time                   = 0;
  setup_context->wave_frequency         = wave_frequency;
  setup_context->wave_phase             = wave_phase;
  setup_context->wave_type              = wave_type;
  setup_context->bl_height_factor       = bl_height_factor;

  // -- QFunction Context
  honee->phys->implicit            = implicit;
  advection_ctx->CtauS             = CtauS;
  advection_ctx->E_wind            = E_wind;
  advection_ctx->implicit          = implicit;
  advection_ctx->strong_form       = strong_form;
  advection_ctx->stabilization     = stab;
  advection_ctx->stabilization_tau = stab_tau;
  advection_ctx->Ctau_a            = Ctau_a;
  advection_ctx->Ctau_d            = Ctau_d;
  advection_ctx->Ctau_t            = Ctau_t;
  advection_ctx->diffusion_coeff   = diffusion_coeff;
  advection_ctx->divFdiff_method   = honee->app_ctx->divFdiffproj_method;

  PetscCallCeed(ceed, CeedQFunctionContextCreate(honee->ceed, &ics_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(ics_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*setup_context), setup_context));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(ics_qfctx, CEED_MEM_HOST, FreeContextPetsc));

  PetscCallCeed(ceed, CeedQFunctionContextCreate(honee->ceed, &advection_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(advection_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*advection_ctx), advection_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(advection_qfctx, CEED_MEM_HOST, FreeContextPetsc));
  PetscCallCeed(ceed, CeedQFunctionContextRegisterDouble(advection_qfctx, "timestep size", offsetof(struct AdvectionContext_, dt), 1,
                                                         "Size of timestep, delta t"));

  // ------------------------------------------------------
  //               SET UP ADVECTION
  // ------------------------------------------------------
  problem->print_info         = PRINT_ADVECTION;
  problem->num_comps_jac_data = 0;
  switch (dim) {
    case 2:
      problem->ics                          = (HoneeQFSpec){.qf_func_ptr = ICsAdvection2d, .qf_loc = ICsAdvection2d_loc};
      problem->apply_vol_rhs                = (HoneeQFSpec){.qf_func_ptr = RHS_Advection2d, .qf_loc = RHS_Advection2d_loc};
      problem->apply_vol_ifunction          = (HoneeQFSpec){.qf_func_ptr = IFunction_Advection2d, .qf_loc = IFunction_Advection2d_loc};
      problem->compute_exact_solution_error = PETSC_TRUE;
      break;
    case 3:
      problem->ics                          = (HoneeQFSpec){.qf_func_ptr = ICsAdvection, .qf_loc = ICsAdvection_loc};
      problem->apply_vol_rhs                = (HoneeQFSpec){.qf_func_ptr = RHS_Advection, .qf_loc = RHS_Advection_loc};
      problem->apply_vol_ifunction          = (HoneeQFSpec){.qf_func_ptr = IFunction_Advection, .qf_loc = IFunction_Advection_loc};
      problem->compute_exact_solution_error = PETSC_FALSE;
      break;
  }
  problem->ics.qfctx           = ics_qfctx;
  problem->apply_vol_rhs.qfctx = advection_qfctx;
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(advection_qfctx, &problem->apply_vol_ifunction.qfctx));

  problem->num_components = 5;
  PetscCall(PetscMalloc1(problem->num_components, &problem->component_names));
  for (PetscInt i = 0; i < 5; i++) PetscCall(PetscStrallocpy(component_names[i], &problem->component_names[i]));

  PetscCall(DivDiffFluxProjectionCreate(honee, honee->app_ctx->divFdiffproj_method, 1, &honee->diff_flux_proj));
  if (honee->diff_flux_proj) {
    DivDiffFluxProjectionData diff_flux_proj = honee->diff_flux_proj;
    NodalProjectionData       projection     = diff_flux_proj->projection;
    PetscSection              section;

    diff_flux_proj->CreateRHSOperator_Direct   = DivDiffFluxProjectionCreateRHS_Direct_AdvDif;
    diff_flux_proj->CreateRHSOperator_Indirect = DivDiffFluxProjectionCreateRHS_Indirect_AdvDif;
    PetscCall(DMGetLocalSection(projection->dm, &section));
    switch (honee->diff_flux_proj->method) {
      case DIV_DIFF_FLUX_PROJ_DIRECT: {
        PetscCall(PetscSectionSetFieldName(section, 0, ""));
        PetscCall(PetscSectionSetComponentName(section, 0, 0, "DivDiffusiveFlux_Scalar"));
      } break;
      case DIV_DIFF_FLUX_PROJ_INDIRECT: {
        PetscCall(PetscSectionSetFieldName(section, 0, ""));
        PetscCall(PetscSectionSetComponentName(section, 0, 0, "DiffusiveFlux_ScalarX"));
        PetscCall(PetscSectionSetComponentName(section, 0, 1, "DiffusiveFlux_ScalarY"));
        if (dim >= 3) PetscCall(PetscSectionSetComponentName(section, 0, 2, "DiffusiveFlux_ScalarZ"));
      } break;
      case DIV_DIFF_FLUX_PROJ_NONE:
        SETERRQ(PetscObjectComm((PetscObject)honee->dm), PETSC_ERR_ARG_WRONG, "Should not reach here with div_diff_flux_projection_method %s",
                DivDiffFluxProjectionMethods[honee->app_ctx->divFdiffproj_method]);
        break;
    }
  }

  PetscCallCeed(ceed, CeedQFunctionContextReference(advection_qfctx));
  PetscCall(HoneeSetContainer(honee, ADVECTION_QFCTX_KEY, advection_qfctx, (PetscCtxDestroyFn *)CeedQFunctionContextDestroy));
  PetscCall(BCDefinitionRegisterNaturalSetup("inflow", BCDefinitionNaturalSetup_Advection));  // For legacy BC setting
  PetscCall(BCDefinitionRegisterNaturalSetup("inflow-outflow", BCDefinitionNaturalSetup_Advection));
  PetscFunctionReturn(PETSC_SUCCESS);
}
