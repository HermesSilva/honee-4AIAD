// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Utility functions for setting up problems using the Newtonian Qfunction

#include "../qfunctions/newtonian.h"

#include <ceed.h>
#include <petscdm.h>

#include <navierstokes.h>

const char *const StateVariables[]     = {"CONSERVATIVE", "PRIMITIVE", "ENTROPY", "StateVariable", "STATEVAR_", NULL};
const char *const StabilizationTypes[] = {"NONE", "SU", "SUPG", "StabilizationType", "STAB_", NULL};

static PetscErrorCode UnitTests_Newtonian(Honee honee, NewtonianIdealGasContext gas);

static PetscErrorCode PRINT_NEWTONIAN(Honee honee, ProblemData problem, AppCtx app_ctx) {
  MPI_Comm                 comm = honee->comm;
  Ceed                     ceed = honee->ceed;
  NewtonianIdealGasContext newtonian_ctx;

  PetscFunctionBeginUser;
  PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->apply_vol_rhs.qfctx, CEED_MEM_HOST, &newtonian_ctx));
  PetscCall(PetscPrintf(comm,
                        "  Problem:\n"
                        "    Problem Name                       : %s\n"
                        "    Stabilization                      : %s\n",
                        app_ctx->problem_name, StabilizationTypes[newtonian_ctx->stabilization]));
  PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfctx, &newtonian_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Create CeedOperator for stabilized mass KSP for explicit timestepping
//
// Only used for SUPG stabilization
PetscErrorCode CreateKSPMassOperator_NewtonianStabilized(Honee honee, CeedOperator *op_mass) {
  Ceed                 ceed = honee->ceed;
  CeedInt              num_comp_q, q_data_size;
  CeedQFunction        qf_mass;
  CeedElemRestriction  elem_restr_q, elem_restr_qd;
  CeedBasis            basis_q;
  CeedVector           q_data;
  CeedQFunctionContext qfctx = NULL;
  PetscInt             dim   = 3;

  PetscFunctionBeginUser;
  {  // Get restriction and basis from the RHS function
    CeedOperator     *sub_ops;
    CeedOperatorField op_field;
    PetscInt          sub_op_index = 0;  // will be 0 for the volume op

    PetscCallCeed(ceed, CeedCompositeOperatorGetSubList(honee->op_rhs_ctx->op, &sub_ops));
    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_ops[sub_op_index], "q", &op_field));
    PetscCallCeed(ceed, CeedOperatorFieldGetData(op_field, NULL, &elem_restr_q, &basis_q, NULL));
    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_ops[sub_op_index], "qdata", &op_field));
    PetscCallCeed(ceed, CeedOperatorFieldGetData(op_field, NULL, &elem_restr_qd, NULL, &q_data));

    PetscCallCeed(ceed, CeedOperatorGetContext(sub_ops[sub_op_index], &qfctx));
  }

  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_q, &num_comp_q));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_qd, &q_data_size));

  PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, MassFunction_Newtonian_Conserv, MassFunction_Newtonian_Conserv_loc, &qf_mass));

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_mass, qfctx));
  PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(qf_mass, 0));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_mass, "q_dot", 5, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_mass, "q", 5, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_mass, "qdata", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_mass, "v", 5, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_mass, "Grad_v", 5 * dim, CEED_EVAL_GRAD));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, op_mass));
  PetscCallCeed(ceed, CeedOperatorSetName(*op_mass, "RHS Mass Operator, Newtonian Stabilized"));
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
static PetscErrorCode DivDiffFluxProjectionCreateRHS_Direct_NS(Honee honee, DivDiffFluxProjectionData diff_flux_proj, CeedOperator *op_rhs) {
  Ceed                 ceed       = honee->ceed;
  NodalProjectionData  projection = diff_flux_proj->projection;
  CeedInt              num_comp_q;
  PetscInt             dim, label_value = 0;
  DMLabel              domain_label    = NULL;
  CeedQFunctionContext newtonian_qfctx = NULL;

  PetscFunctionBeginUser;
  // -- Get Pre-requisite things
  PetscCall(DMGetDimension(projection->dm, &dim));
  PetscCallCeed(ceed, CeedBasisGetNumComponents(honee->basis_q, &num_comp_q));

  {  // Get newtonian QF context
    CeedOperator *sub_ops, main_op = honee->op_ifunction ? honee->op_ifunction : honee->op_rhs_ctx->op;
    PetscInt      sub_op_index = 0;  // will be 0 for the volume op

    PetscCallCeed(ceed, CeedCompositeOperatorGetSubList(main_op, &sub_ops));
    PetscCallCeed(ceed, CeedOperatorGetContext(sub_ops[sub_op_index], &newtonian_qfctx));
  }
  PetscCallCeed(ceed, CeedCompositeOperatorCreate(ceed, op_rhs));
  {  // Add the volume integral CeedOperator
    CeedQFunction       qf_rhs_volume;
    CeedOperator        op_rhs_volume;
    CeedVector          q_data;
    CeedElemRestriction elem_restr_qd, elem_restr_diff_flux_volume = NULL;
    CeedBasis           basis_diff_flux = NULL;
    CeedInt             q_data_size;

    PetscCall(DivDiffFluxProjectionGetOperatorFieldData(diff_flux_proj, &elem_restr_diff_flux_volume, &basis_diff_flux, NULL, NULL));
    PetscCall(QDataGet(ceed, projection->dm, domain_label, label_value, honee->elem_restr_x, honee->basis_x, honee->x_coord, &elem_restr_qd, &q_data,
                       &q_data_size));
    switch (honee->phys->state_var) {
      case STATEVAR_PRIMITIVE:
        PetscCallCeed(ceed,
                      CeedQFunctionCreateInterior(ceed, 1, DivDiffusiveFluxVolumeRHS_NS_Prim, DivDiffusiveFluxVolumeRHS_NS_Prim_loc, &qf_rhs_volume));
        break;
      case STATEVAR_CONSERVATIVE:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DivDiffusiveFluxVolumeRHS_NS_Conserv, DivDiffusiveFluxVolumeRHS_NS_Conserv_loc,
                                                        &qf_rhs_volume));
        break;
      case STATEVAR_ENTROPY:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DivDiffusiveFluxVolumeRHS_NS_Entropy, DivDiffusiveFluxVolumeRHS_NS_Entropy_loc,
                                                        &qf_rhs_volume));
        break;
    }

    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_rhs_volume, newtonian_qfctx));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_volume, "q", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_volume, "Grad_q", num_comp_q * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_volume, "qdata", q_data_size, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_rhs_volume, "diffusive flux RHS", projection->num_comp * dim, CEED_EVAL_GRAD));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_rhs_volume, NULL, NULL, &op_rhs_volume));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_volume, "q", honee->elem_restr_q, honee->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_volume, "Grad_q", honee->elem_restr_q, honee->basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_volume, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_volume, "diffusive flux RHS", elem_restr_diff_flux_volume, basis_diff_flux, CEED_VECTOR_ACTIVE));

    PetscCallCeed(ceed, CeedCompositeOperatorAddSub(*op_rhs, op_rhs_volume));

    PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_diff_flux_volume));
    PetscCallCeed(ceed, CeedBasisDestroy(&basis_diff_flux));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_rhs_volume));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_rhs_volume));
  }

  {  // Add the boundary integral CeedOperator
    CeedQFunction qf_rhs_boundary;
    DMLabel       face_sets_label;
    PetscInt      num_face_set_values, *face_set_values;
    CeedInt       q_data_size;

    // -- Build RHS operator
    switch (honee->phys->state_var) {
      case STATEVAR_PRIMITIVE:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DivDiffusiveFluxBoundaryRHS_NS_Prim, DivDiffusiveFluxBoundaryRHS_NS_Prim_loc,
                                                        &qf_rhs_boundary));
        break;
      case STATEVAR_CONSERVATIVE:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DivDiffusiveFluxBoundaryRHS_NS_Conserv, DivDiffusiveFluxBoundaryRHS_NS_Conserv_loc,
                                                        &qf_rhs_boundary));
        break;
      case STATEVAR_ENTROPY:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DivDiffusiveFluxBoundaryRHS_NS_Entropy, DivDiffusiveFluxBoundaryRHS_NS_Entropy_loc,
                                                        &qf_rhs_boundary));
        break;
    }

    PetscCall(QDataBoundaryGradientGetNumComponents(honee->dm, &q_data_size));
    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_rhs_boundary, newtonian_qfctx));
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

        PetscCall(DMPlexCeedElemRestrictionCreate(ceed, honee->dm, face_orientation_label, orientation, height_cell, dm_field_q, &elem_restr_q));
        PetscCall(DMPlexCeedBasisCellToFaceCreate(ceed, honee->dm, face_orientation_label, orientation, orientation, dm_field_q, &basis_q));
        PetscCall(DMPlexCeedElemRestrictionCreate(ceed, projection->dm, face_orientation_label, orientation, height_face, 0,
                                                  &elem_restr_diff_flux_boundary));
        PetscCall(CreateBasisFromPlex(ceed, projection->dm, face_orientation_label, orientation, height_face, 0, &basis_diff_flux_boundary));
        PetscCall(
            QDataBoundaryGradientGet(ceed, honee->dm, face_orientation_label, orientation, honee->x_coord, &elem_restr_qdata, &q_data, &q_data_size));

        PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_rhs_boundary, NULL, NULL, &op_rhs_boundary));
        PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_boundary, "q", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
        PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_boundary, "Grad_q", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
        PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_boundary, "qdata", elem_restr_qdata, CEED_BASIS_NONE, q_data));
        PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_boundary, "diffusive flux RHS", elem_restr_diff_flux_boundary, basis_diff_flux_boundary,
                                                 CEED_VECTOR_ACTIVE));

        PetscCallCeed(ceed, CeedCompositeOperatorAddSub(*op_rhs, op_rhs_boundary));

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

  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&newtonian_qfctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create RHS CeedOperator for indirect projection of divergence of diffusive flux

  @param[in]  honee          `Honee` context
  @param[in]  diff_flux_proj `DivDiffFluxProjectionData` object
  @param[out] op_rhs         Operator to calculate the RHS of the L^2 projection
**/
static PetscErrorCode DivDiffFluxProjectionCreateRHS_Indirect_NS(Honee honee, DivDiffFluxProjectionData diff_flux_proj, CeedOperator *op_rhs) {
  Ceed                 ceed       = honee->ceed;
  NodalProjectionData  projection = diff_flux_proj->projection;
  CeedBasis            basis_diff_flux;
  CeedElemRestriction  elem_restr_diff_flux, elem_restr_qd;
  CeedVector           q_data;
  CeedInt              num_comp_q, q_data_size;
  PetscInt             dim;
  PetscInt             label_value = 0, height = 0, dm_field = 0;
  DMLabel              domain_label = NULL;
  CeedQFunction        qf_rhs;
  CeedQFunctionContext newtonian_qfctx = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(projection->dm, &dim));
  PetscCallCeed(ceed, CeedBasisGetNumComponents(honee->basis_q, &num_comp_q));

  {  // Get newtonian QF context
    CeedOperator *sub_ops, main_op = honee->op_ifunction ? honee->op_ifunction : honee->op_rhs_ctx->op;
    PetscInt      sub_op_index = 0;  // will be 0 for the volume op

    PetscCallCeed(ceed, CeedCompositeOperatorGetSubList(main_op, &sub_ops));
    PetscCallCeed(ceed, CeedOperatorGetContext(sub_ops[sub_op_index], &newtonian_qfctx));
  }
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, projection->dm, domain_label, label_value, height, dm_field, &elem_restr_diff_flux));
  PetscCall(CreateBasisFromPlex(ceed, projection->dm, domain_label, label_value, height, dm_field, &basis_diff_flux));
  PetscCall(QDataGet(ceed, projection->dm, domain_label, label_value, honee->elem_restr_x, honee->basis_x, honee->x_coord, &elem_restr_qd, &q_data,
                     &q_data_size));

  switch (honee->phys->state_var) {
    case STATEVAR_PRIMITIVE:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DiffusiveFluxRHS_NS_Prim, DiffusiveFluxRHS_NS_Prim_loc, &qf_rhs));
      break;
    case STATEVAR_CONSERVATIVE:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DiffusiveFluxRHS_NS_Conserv, DiffusiveFluxRHS_NS_Conserv_loc, &qf_rhs));
      break;
    case STATEVAR_ENTROPY:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, DiffusiveFluxRHS_NS_Entropy, DiffusiveFluxRHS_NS_Entropy_loc, &qf_rhs));
      break;
  }

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_rhs, newtonian_qfctx));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs, "q", num_comp_q, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs, "Grad_q", num_comp_q * dim, CEED_EVAL_GRAD));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs, "qdata", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_rhs, "F_diff RHS", projection->num_comp, CEED_EVAL_INTERP));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_rhs, NULL, NULL, op_rhs));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_rhs, "q", honee->elem_restr_q, honee->basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_rhs, "Grad_q", honee->elem_restr_q, honee->basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_rhs, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_rhs, "F_diff RHS", elem_restr_diff_flux, basis_diff_flux, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_rhs));
  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&newtonian_qfctx));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_diff_flux));
  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_diff_flux));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BoundaryIntegralBCSetup_CreateIFunctionQF(BCDefinition bc_def, CeedQFunction *qf) {
  HoneeBCStruct honee_bc;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  Honee honee = honee_bc->honee;

  switch (honee->phys->state_var) {
    case STATEVAR_CONSERVATIVE:
      PetscCall(HoneeBCCreateIFunctionQF(bc_def, BoundaryIntegral_Conserv, BoundaryIntegral_Conserv_loc, honee_bc->qfctx, qf));
      break;
    case STATEVAR_PRIMITIVE:
      PetscCall(HoneeBCCreateIFunctionQF(bc_def, BoundaryIntegral_Prim, BoundaryIntegral_Prim_loc, honee_bc->qfctx, qf));
      break;
    case STATEVAR_ENTROPY:
      PetscCall(HoneeBCCreateIFunctionQF(bc_def, BoundaryIntegral_Entropy, BoundaryIntegral_Entropy_loc, honee_bc->qfctx, qf));
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BoundaryIntegralBCSetup_CreateIJacobianQF(BCDefinition bc_def, CeedQFunction *qf) {
  HoneeBCStruct honee_bc;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  Honee honee = honee_bc->honee;

  switch (honee->phys->state_var) {
    case STATEVAR_CONSERVATIVE:
      PetscCall(HoneeBCCreateIJacobianQF(bc_def, BoundaryIntegral_Jacobian_Conserv, BoundaryIntegral_Jacobian_Conserv_loc, honee_bc->qfctx, qf));
      break;
    case STATEVAR_PRIMITIVE:
      PetscCall(HoneeBCCreateIJacobianQF(bc_def, BoundaryIntegral_Jacobian_Prim, BoundaryIntegral_Jacobian_Prim_loc, honee_bc->qfctx, qf));
      break;
    case STATEVAR_ENTROPY:
      PetscCall(HoneeBCCreateIJacobianQF(bc_def, BoundaryIntegral_Jacobian_Entropy, BoundaryIntegral_Jacobian_Entropy_loc, honee_bc->qfctx, qf));
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NS_NEWTONIAN_IG(ProblemData problem, DM dm, void *ctx) {
  SetupContext             setup_context;
  Honee                    honee  = *(Honee *)ctx;
  CeedInt                  degree = honee->app_ctx->degree;
  StabilizationType        stab;
  StateVariable            state_var;
  MPI_Comm                 comm = honee->comm;
  Ceed                     ceed = honee->ceed;
  PetscBool                implicit;
  PetscBool                unit_tests;
  NewtonianIdealGasContext newtonian_ig_ctx;
  CeedQFunctionContext     newtonian_ig_qfctx;

  PetscFunctionBeginUser;
  // Option Defaults
  StatePrimitive reference      = {.pressure = 1.01e5, .velocity = {0}, .temperature = 288.15};
  CeedScalar     idl_decay_time = -1;
  {
    PetscCall(PetscNew(&newtonian_ig_ctx));
    const CeedScalar                 Cv_func[3]                = {36, 60, 128};
    struct NewtonianIdealGasContext_ newtonian_ig_ctx_defaults = {
        .cv           = 717., // J/(kg K)
        .cp           = 1004., // J/(kg K)
        .g            = {0, 0, 0}, // m/s^2
        .lambda       = -2. / 3., // -
        .mu           = 1.8e-5, // Pa s, dynamic viscosity
        .k            = 0.02638, // W/(m K)
        .c_tau        = 0.5 / degree, // -
        .Ctau_t       = 1.0, // -
        .Ctau_v       = Cv_func[(CeedInt)Min(3, degree) - 1],
        .Ctau_C       = 0.25 / degree,
        .Ctau_M       = 0.25 / degree,
        .Ctau_E       = 0.125,
        .idl_start    = 0,
        .idl_length   = 0,
        .idl_pressure = reference.pressure,
        .idl_enable   = PETSC_FALSE,
    };
    *newtonian_ig_ctx = newtonian_ig_ctx_defaults;
  }

  PetscOptionsBegin(comm, NULL, "Options for Newtonian Ideal Gas based problem", NULL);
  PetscCall(PetscOptionsEnum("-state_var", "State variables used", NULL, StateVariables, (PetscEnum)(state_var = STATEVAR_CONSERVATIVE),
                             (PetscEnum *)&state_var, NULL));

  // Newtonian fluid properties
  PetscCall(PetscOptionsScalar("-cv", "Heat capacity at constant volume", NULL, newtonian_ig_ctx->cv, &newtonian_ig_ctx->cv, NULL));
  PetscCall(PetscOptionsScalar("-cp", "Heat capacity at constant pressure", NULL, newtonian_ig_ctx->cp, &newtonian_ig_ctx->cp, NULL));
  PetscCall(PetscOptionsScalar("-lambda", "Stokes hypothesis second viscosity coefficient", NULL, newtonian_ig_ctx->lambda, &newtonian_ig_ctx->lambda,
                               NULL));
  PetscCall(PetscOptionsScalar("-mu", "Shear dynamic viscosity coefficient", NULL, newtonian_ig_ctx->mu, &newtonian_ig_ctx->mu, NULL));
  PetscCall(PetscOptionsScalar("-k", "Thermal conductivity", NULL, newtonian_ig_ctx->k, &newtonian_ig_ctx->k, NULL));

  PetscInt  dim          = 3;
  PetscBool given_option = PETSC_FALSE;
  PetscCall(PetscOptionsDeprecated("-g", "-gravity", "libCEED 0.11.1", NULL));
  PetscCall(PetscOptionsRealArray("-gravity", "Gravitational acceleration vector", NULL, newtonian_ig_ctx->g, &dim, &given_option));
  if (given_option) PetscCheck(dim == 3, comm, PETSC_ERR_ARG_SIZ, "Gravity vector must be size 3, %" PetscInt_FMT " values given", dim);

  // Stabilization parameters
  PetscCall(PetscOptionsEnum("-stab", "Stabilization method", NULL, StabilizationTypes, (PetscEnum)(stab = STAB_NONE), (PetscEnum *)&stab, NULL));
  PetscCall(PetscOptionsScalar("-c_tau", "Stabilization constant", NULL, newtonian_ig_ctx->c_tau, &newtonian_ig_ctx->c_tau, NULL));
  PetscCall(PetscOptionsScalar("-Ctau_t", "Stabilization time constant", NULL, newtonian_ig_ctx->Ctau_t, &newtonian_ig_ctx->Ctau_t, NULL));
  PetscCall(PetscOptionsScalar("-Ctau_v", "Stabilization viscous constant", NULL, newtonian_ig_ctx->Ctau_v, &newtonian_ig_ctx->Ctau_v, NULL));
  PetscCall(PetscOptionsScalar("-Ctau_C", "Stabilization continuity constant", NULL, newtonian_ig_ctx->Ctau_C, &newtonian_ig_ctx->Ctau_C, NULL));
  PetscCall(PetscOptionsScalar("-Ctau_M", "Stabilization momentum constant", NULL, newtonian_ig_ctx->Ctau_M, &newtonian_ig_ctx->Ctau_M, NULL));
  PetscCall(PetscOptionsScalar("-Ctau_E", "Stabilization energy constant", NULL, newtonian_ig_ctx->Ctau_E, &newtonian_ig_ctx->Ctau_E, NULL));

  dim = 3;
  PetscCall(PetscOptionsScalar("-reference_pressure", "Reference/initial pressure", NULL, reference.pressure, &reference.pressure, NULL));
  PetscCall(PetscOptionsScalarArray("-reference_velocity", "Reference/initial velocity", NULL, reference.velocity, &dim, NULL));
  PetscCall(PetscOptionsScalar("-reference_temperature", "Reference/initial temperature", NULL, reference.temperature, &reference.temperature, NULL));

  PetscCall(PetscOptionsBool("-implicit", "Use implicit (IFunction) formulation", NULL, implicit = PETSC_FALSE, &implicit, NULL));
  PetscCall(PetscOptionsBool("-newtonian_unit_tests", "Run Newtonian unit tests", NULL, unit_tests = PETSC_FALSE, &unit_tests, NULL));
  PetscCheck(!(state_var == STATEVAR_PRIMITIVE && !implicit), comm, PETSC_ERR_SUP,
             "RHSFunction is not provided for primitive variables (use -state_var primitive only with -implicit)\n");

  // IDL Settings
  {
    PetscBool idl_enable = (PetscBool)newtonian_ig_ctx->idl_enable;  // Need PetscBool variable to read in from PetscOptionsScalar()
    PetscCall(PetscOptionsScalar("-idl_decay_time", "Characteristic timescale of the pressure deviance decay. The timestep is good starting point",
                                 NULL, idl_decay_time, &idl_decay_time, &idl_enable));
    PetscCheck(!(idl_enable && idl_decay_time == 0), comm, PETSC_ERR_SUP, "idl_decay_time may not be equal to zero.");
    if (idl_decay_time < 0) idl_enable = PETSC_FALSE;
    newtonian_ig_ctx->idl_enable = idl_enable;

    PetscCall(
        PetscOptionsScalar("-idl_start", "Start of IDL in the x direction", NULL, newtonian_ig_ctx->idl_start, &newtonian_ig_ctx->idl_start, NULL));
    PetscCall(PetscOptionsScalar("-idl_length", "Length of IDL in the positive x direction", NULL, newtonian_ig_ctx->idl_length,
                                 &newtonian_ig_ctx->idl_length, NULL));
    newtonian_ig_ctx->idl_pressure = reference.pressure;
    PetscCall(PetscOptionsScalar("-idl_pressure", "Pressure IDL uses as reference (default is `-reference_pressure`)", NULL,
                                 newtonian_ig_ctx->idl_pressure, &newtonian_ig_ctx->idl_pressure, NULL));
  }
  PetscOptionsEnd();

  // Set problem information
  problem->num_comps_jac_data = 14;
  if (newtonian_ig_ctx->idl_enable) problem->num_comps_jac_data += 1;
  problem->compute_exact_solution_error = PETSC_FALSE;
  problem->print_info                   = PRINT_NEWTONIAN;
  problem->num_components               = 5;
  PetscCall(PetscMalloc1(problem->num_components, &problem->component_names));
  static const char *const conservative_component_names[] = {"Density", "MomentumX", "MomentumY", "MomentumZ", "TotalEnergy"};
  static const char *const primitive_component_names[]    = {"Pressure", "VelocityX", "VelocityY", "VelocityZ", "Temperature"};
  static const char *const entropy_component_names[]      = {"EntropyDensity", "EntropyMomentumX", "EntropyMomentumY", "EntropyMomentumZ",
                                                             "EntropyTotalEnergy"};

  switch (state_var) {
    case STATEVAR_CONSERVATIVE:
      problem->ics.qf_func_ptr                 = ICsNewtonianIG_Conserv;
      problem->ics.qf_loc                      = ICsNewtonianIG_Conserv_loc;
      problem->apply_vol_rhs.qf_func_ptr       = RHSFunction_Newtonian;
      problem->apply_vol_rhs.qf_loc            = RHSFunction_Newtonian_loc;
      problem->apply_vol_ifunction.qf_func_ptr = IFunction_Newtonian_Conserv;
      problem->apply_vol_ifunction.qf_loc      = IFunction_Newtonian_Conserv_loc;
      problem->apply_vol_ijacobian.qf_func_ptr = IJacobian_Newtonian_Conserv;
      problem->apply_vol_ijacobian.qf_loc      = IJacobian_Newtonian_Conserv_loc;
      for (PetscInt i = 0; i < 5; i++) PetscCall(PetscStrallocpy(conservative_component_names[i], &problem->component_names[i]));
      break;
    case STATEVAR_PRIMITIVE:
      problem->ics.qf_func_ptr                 = ICsNewtonianIG_Prim;
      problem->ics.qf_loc                      = ICsNewtonianIG_Prim_loc;
      problem->apply_vol_ifunction.qf_func_ptr = IFunction_Newtonian_Prim;
      problem->apply_vol_ifunction.qf_loc      = IFunction_Newtonian_Prim_loc;
      problem->apply_vol_ijacobian.qf_func_ptr = IJacobian_Newtonian_Prim;
      problem->apply_vol_ijacobian.qf_loc      = IJacobian_Newtonian_Prim_loc;
      for (PetscInt i = 0; i < 5; i++) PetscCall(PetscStrallocpy(primitive_component_names[i], &problem->component_names[i]));
      break;
    case STATEVAR_ENTROPY:
      problem->ics.qf_func_ptr                 = ICsNewtonianIG_Entropy;
      problem->ics.qf_loc                      = ICsNewtonianIG_Entropy_loc;
      problem->apply_vol_ifunction.qf_func_ptr = IFunction_Newtonian_Entropy;
      problem->apply_vol_ifunction.qf_loc      = IFunction_Newtonian_Entropy_loc;
      problem->apply_vol_ijacobian.qf_func_ptr = IJacobian_Newtonian_Entropy;
      problem->apply_vol_ijacobian.qf_loc      = IJacobian_Newtonian_Entropy_loc;
      for (PetscInt i = 0; i < 5; i++) PetscCall(PetscStrallocpy(entropy_component_names[i], &problem->component_names[i]));
      break;
  }

  if (stab == STAB_SUPG && !implicit) problem->create_mass_operator = CreateKSPMassOperator_NewtonianStabilized;

  PetscCall(DivDiffFluxProjectionCreate(honee, honee->app_ctx->divFdiffproj_method, 4, &honee->diff_flux_proj));
  if (honee->diff_flux_proj) {
    DivDiffFluxProjectionData diff_flux_proj = honee->diff_flux_proj;
    NodalProjectionData       projection     = diff_flux_proj->projection;

    diff_flux_proj->CreateRHSOperator_Direct   = DivDiffFluxProjectionCreateRHS_Direct_NS;
    diff_flux_proj->CreateRHSOperator_Indirect = DivDiffFluxProjectionCreateRHS_Indirect_NS;

    switch (honee->diff_flux_proj->method) {
      case DIV_DIFF_FLUX_PROJ_DIRECT: {
        PetscSection section;

        PetscCall(DMGetLocalSection(projection->dm, &section));
        PetscCall(PetscSectionSetFieldName(section, 0, ""));
        PetscCall(PetscSectionSetComponentName(section, 0, 0, "DivDiffusiveFlux_MomentumX"));
        PetscCall(PetscSectionSetComponentName(section, 0, 1, "DivDiffusiveFlux_MomentumY"));
        PetscCall(PetscSectionSetComponentName(section, 0, 2, "DivDiffusiveFlux_MomentumZ"));
        PetscCall(PetscSectionSetComponentName(section, 0, 3, "DivDiffusiveFlux_Energy"));
      } break;
      case DIV_DIFF_FLUX_PROJ_INDIRECT: {
        PetscSection section;

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
      } break;
      case DIV_DIFF_FLUX_PROJ_NONE:
        SETERRQ(PetscObjectComm((PetscObject)honee->dm), PETSC_ERR_ARG_WRONG, "Should not reach here with div_diff_flux_projection_method %s",
                DivDiffFluxProjectionMethods[honee->app_ctx->divFdiffproj_method]);
        break;
    }
  }

  // ------------------------------------------------------
  //           Set up the QFunction context
  // ------------------------------------------------------
  // -- Scale variables to desired units
  Units units = honee->units;
  newtonian_ig_ctx->cv *= units->J_per_kg_K;
  newtonian_ig_ctx->cp *= units->J_per_kg_K;
  newtonian_ig_ctx->mu *= units->Pascal * units->second;
  newtonian_ig_ctx->k *= units->W_per_m_K;
  for (PetscInt i = 0; i < 3; i++) newtonian_ig_ctx->g[i] *= units->m_per_squared_s;
  reference.pressure *= units->Pascal;
  for (PetscInt i = 0; i < 3; i++) reference.velocity[i] *= units->meter / units->second;
  reference.temperature *= units->Kelvin;

  PetscReal domain_min[3], domain_max[3], domain_size[3];
  PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
  for (PetscInt i = 0; i < 3; i++) domain_size[i] = (domain_max[i] - domain_min[i]) * units->meter;

  // -- Solver Settings
  honee->phys->implicit  = implicit;
  honee->phys->state_var = state_var;

  // -- QFunction Context
  newtonian_ig_ctx->stabilization   = stab;
  newtonian_ig_ctx->is_implicit     = implicit;
  newtonian_ig_ctx->state_var       = state_var;
  newtonian_ig_ctx->idl_amplitude   = 1 / (idl_decay_time * units->second);
  newtonian_ig_ctx->divFdiff_method = honee->app_ctx->divFdiffproj_method;

  // -- Setup Context
  PetscCall(PetscNew(&setup_context));
  setup_context->reference = reference;
  setup_context->gas       = *newtonian_ig_ctx;
  setup_context->lx        = domain_size[0];
  setup_context->ly        = domain_size[1];
  setup_context->lz        = domain_size[2];
  setup_context->time      = 0;

  PetscCallCeed(ceed, CeedQFunctionContextCreate(honee->ceed, &problem->ics.qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(problem->ics.qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*setup_context), setup_context));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(problem->ics.qfctx, CEED_MEM_HOST, FreeContextPetsc));
  PetscCallCeed(
      ceed, CeedQFunctionContextRegisterDouble(problem->ics.qfctx, "evaluation time", offsetof(struct SetupContext_, time), 1, "Time of evaluation"));

  PetscCallCeed(ceed, CeedQFunctionContextCreate(honee->ceed, &newtonian_ig_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(newtonian_ig_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*newtonian_ig_ctx), newtonian_ig_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(newtonian_ig_qfctx, CEED_MEM_HOST, FreeContextPetsc));
  PetscCallCeed(ceed, CeedQFunctionContextRegisterDouble(newtonian_ig_qfctx, "timestep size", offsetof(struct NewtonianIdealGasContext_, dt), 1,
                                                         "Size of timestep, delta t"));
  PetscCallCeed(ceed, CeedQFunctionContextRegisterDouble(newtonian_ig_qfctx, "ijacobian time shift",
                                                         offsetof(struct NewtonianIdealGasContext_, ijacobian_time_shift), 1,
                                                         "Shift for mass matrix in IJacobian"));
  PetscCallCeed(ceed, CeedQFunctionContextRegisterDouble(newtonian_ig_qfctx, "solution time", offsetof(struct NewtonianIdealGasContext_, time), 1,
                                                         "Current solution time"));

  problem->apply_vol_rhs.qfctx = newtonian_ig_qfctx;
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(newtonian_ig_qfctx, &problem->apply_vol_ifunction.qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(newtonian_ig_qfctx, &problem->apply_vol_ijacobian.qfctx));

  for (PetscCount b = 0; b < problem->num_bc_defs; b++) {
    BCDefinition bc_def = problem->bc_defs[b];
    const char  *name;

    PetscCall(BCDefinitionGetInfo(bc_def, &name, NULL, NULL));
    if (!strcmp(name, "slip")) {
      PetscCall(SlipBCSetup(bc_def, problem, dm, ctx, newtonian_ig_qfctx));
    } else if (!strcmp(name, "freestream")) {
      PetscCall(FreestreamBCSetup(bc_def, problem, dm, ctx, newtonian_ig_ctx, &reference));
    } else if (!strcmp(name, "outflow")) {
      PetscCall(OutflowBCSetup(bc_def, problem, dm, ctx, newtonian_ig_ctx, &reference));
    } else if (!strcmp(name, "inflow")) {
      HoneeBCStruct honee_bc;

      PetscCall(PetscNew(&honee_bc));
      PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(newtonian_ig_qfctx, &honee_bc->qfctx));
      honee_bc->honee              = honee;
      honee_bc->num_comps_jac_data = honee->phys->implicit ? 11 : 0;
      PetscCall(BCDefinitionSetContext(bc_def, HoneeBCDestroy, honee_bc));

      PetscCall(BCDefinitionSetIFunction(bc_def, BoundaryIntegralBCSetup_CreateIFunctionQF, HoneeBCAddIFunctionOp));
      PetscCall(BCDefinitionSetIJacobian(bc_def, BoundaryIntegralBCSetup_CreateIJacobianQF, HoneeBCAddIJacobianOp));
    }
  }

  if (unit_tests) PetscCall(UnitTests_Newtonian(honee, newtonian_ig_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------
// Unit test functions
//------------------------------------

static PetscErrorCode CheckQWithTolerance(const CeedScalar Q_s[5], const CeedScalar Q_a[5], const CeedScalar Q_b[5], const char *name,
                                          PetscReal rtol_0, PetscReal rtol_u, PetscReal rtol_4) {
  CeedScalar relative_error[5];  // relative error
  CeedScalar divisor_threshold = 10 * CEED_EPSILON;

  PetscFunctionBeginUser;
  relative_error[0] = (Q_a[0] - Q_b[0]) / (fabs(Q_s[0]) > divisor_threshold ? Q_s[0] : 1);
  relative_error[4] = (Q_a[4] - Q_b[4]) / (fabs(Q_s[4]) > divisor_threshold ? Q_s[4] : 1);

  CeedScalar u_magnitude = sqrt(Square(Q_s[1]) + Square(Q_s[2]) + Square(Q_s[3]));
  CeedScalar u_divisor   = u_magnitude > divisor_threshold ? u_magnitude : 1;
  for (int i = 1; i < 4; i++) {
    relative_error[i] = (Q_a[i] - Q_b[i]) / u_divisor;
  }

  if (fabs(relative_error[0]) >= rtol_0) {
    printf("%s[0] error %g (expected %.10e, got %.10e)\n", name, relative_error[0], Q_s[0], Q_a[0]);
  }
  for (int i = 1; i < 4; i++) {
    if (fabs(relative_error[i]) >= rtol_u) {
      printf("%s[%d] error %g (expected %.10e, got %.10e)\n", name, i, relative_error[i], Q_s[i], Q_a[i]);
    }
  }
  if (fabs(relative_error[4]) >= rtol_4) {
    printf("%s[4] error %g (expected %.10e, got %.10e)\n", name, relative_error[4], Q_s[4], Q_a[4]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Verify `StateFromQ` by converting A0 -> B0 -> A0_test, where A0 should equal A0_test
static PetscErrorCode TestState(StateVariable state_var_A, StateVariable state_var_B, NewtonianIdealGasContext gas, const CeedScalar A0[5],
                                CeedScalar rtol_0, CeedScalar rtol_u, CeedScalar rtol_4) {
  CeedScalar        B0[5], A0_test[5];
  char              buf[128];
  const char *const StateVariables_Initial[] = {"U", "Y", "V"};

  PetscFunctionBeginUser;
  const char *A_initial = StateVariables_Initial[state_var_A];
  const char *B_initial = StateVariables_Initial[state_var_B];

  State state_A0 = StateFromQ(gas, A0, state_var_A);
  StateToQ(gas, state_A0, B0, state_var_B);
  State state_B0 = StateFromQ(gas, B0, state_var_B);
  StateToQ(gas, state_B0, A0_test, state_var_A);

  snprintf(buf, sizeof buf, "%s->%s->%s: %s", A_initial, B_initial, A_initial, A_initial);
  PetscCall(CheckQWithTolerance(A0, A0_test, A0, buf, rtol_0, rtol_u, rtol_4));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Verify `StateFromQ_fwd` via a finite difference approximation
static PetscErrorCode TestState_fwd(StateVariable state_var_A, StateVariable state_var_B, NewtonianIdealGasContext gas, const CeedScalar A0[5],
                                    CeedScalar rtol_0, CeedScalar rtol_u, CeedScalar rtol_4) {
  CeedScalar        eps = 4e-7;  // Finite difference step
  char              buf[128];
  const char *const StateVariables_Initial[] = {"U", "Y", "V"};

  PetscFunctionBeginUser;
  const char *A_initial = StateVariables_Initial[state_var_A];
  const char *B_initial = StateVariables_Initial[state_var_B];
  State       state_0   = StateFromQ(gas, A0, state_var_A);

  for (int i = 0; i < 5; i++) {
    CeedScalar dB[5] = {0.}, dB_fd[5] = {0.};
    {  // Calculate dB using State functions
      CeedScalar dA[5] = {0};

      dA[i]          = A0[i];
      State dstate_0 = StateFromQ_fwd(gas, state_0, dA, state_var_A);
      StateToQ_fwd(gas, state_0, dstate_0, dB, state_var_B);
    }

    {  // Calculate dB_fd via finite difference approximation
      CeedScalar A1[5], B0[5], B1[5];

      for (int j = 0; j < 5; j++) A1[j] = (1 + eps * (i == j)) * A0[j];
      State state_1 = StateFromQ(gas, A1, state_var_A);
      StateToQ(gas, state_0, B0, state_var_B);
      StateToQ(gas, state_1, B1, state_var_B);
      for (int j = 0; j < 5; j++) dB_fd[j] = (B1[j] - B0[j]) / eps;
    }

    snprintf(buf, sizeof buf, "d%s->d%s: StateFrom%s_fwd i=%d: d%s", A_initial, B_initial, A_initial, i, B_initial);
    PetscCall(CheckQWithTolerance(dB_fd, dB, dB_fd, buf, rtol_0, rtol_u, rtol_4));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Test the Newtonian State transformation functions, `StateFrom*`
static PetscErrorCode UnitTests_Newtonian(Honee honee, NewtonianIdealGasContext gas) {
  Units            units = honee->units;
  const CeedScalar kg = units->kilogram, m = units->meter, sec = units->second, K = units->Kelvin;
  CeedScalar       rtol;

  PetscFunctionBeginUser;
  const CeedScalar T          = 200 * K;
  const CeedScalar rho        = 1.2 * kg / Cube(m);
  const CeedScalar P          = (HeatCapacityRatio(gas) - 1) * rho * gas->cv * T;
  const CeedScalar u_base     = 40 * m / sec;
  const CeedScalar u[3]       = {u_base, u_base * 1.1, u_base * 1.2};
  const CeedScalar e_kinetic  = 0.5 * Dot3(u, u);
  const CeedScalar e_internal = gas->cv * T;
  const CeedScalar e_total    = e_kinetic + e_internal;
  const CeedScalar gamma      = HeatCapacityRatio(gas);
  const CeedScalar entropy    = log(P) - gamma * log(rho);
  const CeedScalar rho_div_p  = rho / P;
  const CeedScalar Y0[5]      = {P, u[0], u[1], u[2], T};
  const CeedScalar U0[5]      = {rho, rho * u[0], rho * u[1], rho * u[2], rho * e_total};
  const CeedScalar V0[5]      = {(gamma - entropy) / (gamma - 1) - rho_div_p * (e_kinetic), rho_div_p * u[0], rho_div_p * u[1], rho_div_p * u[2],
                                 -rho_div_p};

  rtol = 20 * CEED_EPSILON;
  PetscCall(TestState(STATEVAR_PRIMITIVE, STATEVAR_CONSERVATIVE, gas, Y0, rtol, rtol, rtol));
  PetscCall(TestState(STATEVAR_PRIMITIVE, STATEVAR_ENTROPY, gas, Y0, rtol, rtol, rtol));
  PetscCall(TestState(STATEVAR_CONSERVATIVE, STATEVAR_PRIMITIVE, gas, U0, rtol, rtol, rtol));
  PetscCall(TestState(STATEVAR_CONSERVATIVE, STATEVAR_ENTROPY, gas, U0, rtol, rtol, rtol));
  PetscCall(TestState(STATEVAR_ENTROPY, STATEVAR_CONSERVATIVE, gas, V0, rtol, rtol, rtol));
  PetscCall(TestState(STATEVAR_ENTROPY, STATEVAR_PRIMITIVE, gas, V0, rtol, rtol, rtol));

  rtol = 5e-6;
  PetscCall(TestState_fwd(STATEVAR_PRIMITIVE, STATEVAR_CONSERVATIVE, gas, Y0, rtol, rtol, rtol));
  PetscCall(TestState_fwd(STATEVAR_PRIMITIVE, STATEVAR_ENTROPY, gas, Y0, rtol, rtol, rtol));
  PetscCall(TestState_fwd(STATEVAR_CONSERVATIVE, STATEVAR_PRIMITIVE, gas, U0, rtol, rtol, rtol));
  PetscCall(TestState_fwd(STATEVAR_CONSERVATIVE, STATEVAR_ENTROPY, gas, U0, 10 * rtol, rtol, rtol));
  PetscCall(TestState_fwd(STATEVAR_ENTROPY, STATEVAR_CONSERVATIVE, gas, V0, 5 * rtol, rtol, rtol));
  PetscCall(TestState_fwd(STATEVAR_ENTROPY, STATEVAR_PRIMITIVE, gas, V0, 5 * rtol, 5 * rtol, 5 * rtol));
  PetscFunctionReturn(PETSC_SUCCESS);
}
