// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Setup libCEED Operators for HONEE

#include <ceed.h>
#include <petscdmplex.h>
#include <smartsim.h>

#include <navierstokes.h>

// @brief Create CeedOperator for unstabilized mass KSP for explicit timestepping
static PetscErrorCode CreateKSPMassOperator_Unstabilized(Honee honee, CeedOperator *op_mass) {
  Ceed                ceed = honee->ceed;
  CeedInt             num_comp_q, q_data_size;
  CeedQFunction       qf_mass;
  CeedElemRestriction elem_restr_q, elem_restr_qd;
  CeedBasis           basis_q;
  CeedVector          q_data;

  PetscFunctionBeginUser;
  {  // Get restriction and basis from the RHS function
    CeedOperator     *sub_ops;
    CeedOperatorField op_field;
    PetscInt          sub_op_index = 0;  // will be 0 for the volume op

    PetscCallCeed(ceed, CeedOperatorCompositeGetSubList(honee->op_rhs_ctx->op, &sub_ops));
    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_ops[sub_op_index], "q", &op_field));
    PetscCallCeed(ceed, CeedOperatorFieldGetData(op_field, NULL, &elem_restr_q, &basis_q, NULL));
    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_ops[sub_op_index], "qdata", &op_field));
    PetscCallCeed(ceed, CeedOperatorFieldGetData(op_field, NULL, &elem_restr_qd, NULL, &q_data));
  }

  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_q, &num_comp_q));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_qd, &q_data_size));

  PetscCall(HoneeMassQFunctionCreate(ceed, num_comp_q, q_data_size, &qf_mass));
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, op_mass));
  PetscCallCeed(ceed, CeedOperatorSetName(*op_mass, "RHS Mass Operator"));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "u", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "v", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_q));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_mass));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Create KSP to solve the inverse mass operator for explicit time stepping schemes
static PetscErrorCode CreateKSPMass(Honee honee, ProblemData problem) {
  Ceed         ceed = honee->ceed;
  DM           dm   = honee->dm;
  CeedOperator op_mass;

  PetscFunctionBeginUser;
  if (problem->create_mass_operator) PetscCall(problem->create_mass_operator(honee, &op_mass));
  else PetscCall(CreateKSPMassOperator_Unstabilized(honee, &op_mass));

  {  // -- Setup KSP for mass operator
    Mat      mat_mass;
    Vec      Zeros_loc;
    MPI_Comm comm = PetscObjectComm((PetscObject)dm);

    PetscCall(DMCreateLocalVector(dm, &Zeros_loc));
    PetscCall(VecZeroEntries(Zeros_loc));
    PetscCall(MatCreateCeed(dm, dm, op_mass, NULL, &mat_mass));
    PetscCall(MatCeedSetLocalVectors(mat_mass, Zeros_loc, NULL));

    PetscCall(KSPCreate(comm, &honee->mass_ksp));
    PetscCall(KSPSetOptionsPrefix(honee->mass_ksp, "mass_"));
    PetscCall(PetscObjectSetName((PetscObject)honee->mass_ksp, "Explicit Mass"));
    {  // lumped by default
      PC pc;
      PetscCall(KSPGetPC(honee->mass_ksp, &pc));
      PetscCall(PCSetType(pc, PCJACOBI));
      PetscCall(PCJacobiSetType(pc, PC_JACOBI_ROWSUM));
      PetscCall(KSPSetType(honee->mass_ksp, KSPPREONLY));
    }
    PetscCall(KSPSetFromOptions_WithMatCeed(honee->mass_ksp, mat_mass));
    PetscCall(VecDestroy(&Zeros_loc));
    PetscCall(MatDestroy(&mat_mass));
  }

  PetscCallCeed(ceed, CeedOperatorDestroy(&op_mass));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupLibceed(Ceed ceed, DM dm, Honee honee, AppCtx app_ctx, ProblemData problem) {
  const PetscInt      num_comp_q = problem->num_components;
  PetscInt            dim, num_comp_x;
  CeedInt             q_data_size_vol, num_comps_jac_data = problem->num_comps_jac_data;
  CeedElemRestriction elem_restr_qd, elem_restr_q, elem_restr_x, elem_restr_jd_i = NULL;
  CeedBasis           basis_q, basis_x;
  CeedVector          q_data, x_coord, jac_data = NULL;
  CeedOperator        op_ifunction_vol = NULL, op_rhs_vol = NULL, op_ijacobian_vol = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateNumComps(dm, &num_comp_x));

  CeedElemRestriction elem_restr_diff_flux = NULL;
  CeedVector          div_diff_flux_ceed   = NULL;
  CeedBasis           basis_diff_flux      = NULL;
  CeedEvalMode        eval_mode_diff_flux  = -1;
  {  // Create bases and element restrictions
    PetscInt height = 0, dm_field = 0;
    DM       dm_coord;

    PetscCall(DMGetCoordinateDM(dm, &dm_coord));
    PetscCall(DMPlexCeedBasisCreate(ceed, dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, height, dm_field, &basis_q));

    PetscCall(DMPlexCeedElemRestrictionCreate(ceed, dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, height, 0, &elem_restr_q));
    if (num_comps_jac_data) {
      PetscCall(DMPlexCeedElemRestrictionQDataCreate(ceed, dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, height, num_comps_jac_data, &elem_restr_jd_i));
      PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_jd_i, &jac_data, NULL));
    }

    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_q, &honee->q_ceed, NULL));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_q, &honee->q_dot_ceed, NULL));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_q, &honee->g_ceed, NULL));
    PetscCall(DMPlexCeedCoordinateCreateField(ceed, dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, height, &elem_restr_x, &basis_x, &x_coord));

    PetscCall(QDataGet(ceed, dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, &elem_restr_qd, &q_data, &q_data_size_vol));
  }

  if (app_ctx->divFdiffproj_method != DIV_DIFF_FLUX_PROJ_NONE) {
    PetscCheck(honee->diff_flux_proj, honee->comm, PETSC_ERR_ARG_WRONGSTATE,
               "Divergence of diffusive flux projection requested but object not created");
    PetscCall(DivDiffFluxProjectionGetOperatorFieldData(honee->diff_flux_proj, &elem_restr_diff_flux, &basis_diff_flux, &div_diff_flux_ceed,
                                                        &eval_mode_diff_flux));
  }

  {  // -- Create QFunction for ICs
    CeedBasis     basis_xc;
    CeedQFunction qf_ics;
    CeedOperator  op_ics;

    PetscCallCeed(ceed, CeedBasisCreateProjection(basis_x, basis_q, &basis_xc));
    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, problem->ics.qf_func_ptr, problem->ics.qf_loc, &qf_ics));
    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_ics, problem->ics.qfctx));
    PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(qf_ics, 0));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ics, "x", num_comp_x, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ics, "dx", num_comp_x * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_ics, "q0", num_comp_q, CEED_EVAL_NONE));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_ics, NULL, NULL, &op_ics));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ics, "x", elem_restr_x, basis_xc, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ics, "dx", elem_restr_x, basis_xc, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ics, "q0", elem_restr_q, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(op_ics, "evaluation time", &honee->phys->ics_time_label));
    PetscCall(OperatorApplyContextCreate(NULL, dm, honee->ceed, op_ics, x_coord, NULL, NULL, honee->Q_loc, &honee->op_ics_ctx));

    PetscCallCeed(ceed, CeedBasisDestroy(&basis_xc));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_ics));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_ics));
  }

  if (problem->apply_vol_rhs.qf_func_ptr) {
    CeedQFunction qf_rhs_vol;

    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, problem->apply_vol_rhs.qf_func_ptr, problem->apply_vol_rhs.qf_loc, &qf_rhs_vol));
    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_rhs_vol, problem->apply_vol_rhs.qfctx));
    PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(qf_rhs_vol, 0));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_vol, "q", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_vol, "Grad_q", num_comp_q * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_vol, "qdata", q_data_size_vol, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_vol, "x", num_comp_x, CEED_EVAL_INTERP));
    if (app_ctx->divFdiffproj_method != DIV_DIFF_FLUX_PROJ_NONE)
      PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_vol, "div F_diff", honee->diff_flux_proj->num_diff_flux_comps, eval_mode_diff_flux));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_rhs_vol, "v", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_rhs_vol, "Grad_v", num_comp_q * dim, CEED_EVAL_GRAD));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_rhs_vol, NULL, NULL, &op_rhs_vol));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_vol, "q", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_vol, "Grad_q", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_vol, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_vol, "x", elem_restr_x, basis_x, x_coord));
    if (app_ctx->divFdiffproj_method != DIV_DIFF_FLUX_PROJ_NONE)
      PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_vol, "div F_diff", elem_restr_diff_flux, basis_diff_flux, div_diff_flux_ceed));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_vol, "v", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_vol, "Grad_v", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));

    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_rhs_vol));
  }

  if (problem->apply_vol_ifunction.qf_func_ptr) {
    CeedQFunction qf_ifunction_vol;

    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, problem->apply_vol_ifunction.qf_func_ptr, problem->apply_vol_ifunction.qf_loc,
                                                    &qf_ifunction_vol));
    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_ifunction_vol, problem->apply_vol_ifunction.qfctx));
    PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(qf_ifunction_vol, 0));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ifunction_vol, "q", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ifunction_vol, "Grad_q", num_comp_q * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ifunction_vol, "q dot", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ifunction_vol, "qdata", q_data_size_vol, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ifunction_vol, "x", num_comp_x, CEED_EVAL_INTERP));
    if (app_ctx->divFdiffproj_method != DIV_DIFF_FLUX_PROJ_NONE)
      PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ifunction_vol, "div F_diff", honee->diff_flux_proj->num_diff_flux_comps, eval_mode_diff_flux));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_ifunction_vol, "v", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_ifunction_vol, "Grad_v", num_comp_q * dim, CEED_EVAL_GRAD));
    if (num_comps_jac_data) PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_ifunction_vol, "jac_data", num_comps_jac_data, CEED_EVAL_NONE));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_ifunction_vol, NULL, NULL, &op_ifunction_vol));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ifunction_vol, "q", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ifunction_vol, "Grad_q", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ifunction_vol, "q dot", elem_restr_q, basis_q, honee->q_dot_ceed));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ifunction_vol, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ifunction_vol, "x", elem_restr_x, basis_x, x_coord));
    if (app_ctx->divFdiffproj_method != DIV_DIFF_FLUX_PROJ_NONE)
      PetscCallCeed(ceed, CeedOperatorSetField(op_ifunction_vol, "div F_diff", elem_restr_diff_flux, basis_diff_flux, div_diff_flux_ceed));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ifunction_vol, "v", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ifunction_vol, "Grad_v", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
    if (num_comps_jac_data) PetscCallCeed(ceed, CeedOperatorSetField(op_ifunction_vol, "jac_data", elem_restr_jd_i, CEED_BASIS_NONE, jac_data));

    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_ifunction_vol));
  }

  if (problem->apply_vol_ijacobian.qf_func_ptr) {
    CeedQFunction qf_ijacobian_vol;

    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, problem->apply_vol_ijacobian.qf_func_ptr, problem->apply_vol_ijacobian.qf_loc,
                                                    &qf_ijacobian_vol));
    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_ijacobian_vol, problem->apply_vol_ijacobian.qfctx));
    PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(qf_ijacobian_vol, 0));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ijacobian_vol, "dq", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ijacobian_vol, "Grad_dq", num_comp_q * dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ijacobian_vol, "qdata", q_data_size_vol, CEED_EVAL_NONE));
    if (num_comps_jac_data) PetscCallCeed(ceed, CeedQFunctionAddInput(qf_ijacobian_vol, "jac_data", num_comps_jac_data, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_ijacobian_vol, "v", num_comp_q, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_ijacobian_vol, "Grad_v", num_comp_q * dim, CEED_EVAL_GRAD));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_ijacobian_vol, NULL, NULL, &op_ijacobian_vol));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ijacobian_vol, "dq", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ijacobian_vol, "Grad_dq", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ijacobian_vol, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
    if (num_comps_jac_data) PetscCallCeed(ceed, CeedOperatorSetField(op_ijacobian_vol, "jac_data", elem_restr_jd_i, CEED_BASIS_NONE, jac_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ijacobian_vol, "v", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_ijacobian_vol, "Grad_v", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));

    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_ijacobian_vol));
  }

  // -- Create and apply CEED Composite Operator for the entire domain
  if (!honee->phys->implicit) {  // RHS
    CeedOperator op_rhs;

    PetscCallCeed(ceed, CeedOperatorCreateComposite(ceed, &op_rhs));
    PetscCallCeed(ceed, CeedOperatorCompositeAddSub(op_rhs, op_rhs_vol));
    for (PetscCount b = 0; b < problem->num_bc_defs; b++) PetscCall(BCDefinitionAddOperators(problem->bc_defs[b], op_rhs, NULL));

    PetscCall(OperatorApplyContextCreate(dm, dm, ceed, op_rhs, honee->q_ceed, honee->g_ceed, honee->Q_loc, NULL, &honee->op_rhs_ctx));

    // ----- Get Context Labels for Operator
    PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(op_rhs, "solution time", &honee->phys->solution_time_label));
    PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(op_rhs, "timestep size", &honee->phys->timestep_size_label));

    PetscCallCeed(ceed, CeedOperatorDestroy(&op_rhs));
    PetscCall(CreateKSPMass(honee, problem));
    PetscCheck(app_ctx->sgs_model_type == SGS_MODEL_NONE, honee->comm, PETSC_ERR_SUP, "SGS modeling not implemented for explicit timestepping");
  } else {  // IFunction
    CeedOperator op_ijacobian = NULL;

    // Create Composite Operaters
    PetscCallCeed(ceed, CeedOperatorCreateComposite(ceed, &honee->op_ifunction));
    PetscCallCeed(ceed, CeedOperatorCompositeAddSub(honee->op_ifunction, op_ifunction_vol));
    if (op_ijacobian_vol) {
      PetscCallCeed(ceed, CeedOperatorCreateComposite(ceed, &op_ijacobian));
      PetscCallCeed(ceed, CeedOperatorCompositeAddSub(op_ijacobian, op_ijacobian_vol));
    }
    for (PetscCount b = 0; b < problem->num_bc_defs; b++) PetscCall(BCDefinitionAddOperators(problem->bc_defs[b], honee->op_ifunction, op_ijacobian));

    // ----- Get Context Labels for Operator
    PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(honee->op_ifunction, "solution time", &honee->phys->solution_time_label));
    PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(honee->op_ifunction, "timestep size", &honee->phys->timestep_size_label));

    if (op_ijacobian) {
      PetscCall(MatCreateCeed(honee->dm, honee->dm, op_ijacobian, NULL, &honee->mat_ijacobian));
      PetscCall(MatCeedSetLocalVectors(honee->mat_ijacobian, honee->Q_dot_loc, NULL));
      PetscCallCeed(ceed, CeedOperatorDestroy(&op_ijacobian));
    }
    if (app_ctx->sgs_model_type == SGS_MODEL_DATA_DRIVEN) PetscCall(SgsDDSetup(ceed, honee, problem));
  }

  if (problem->use_strong_bc_ceed) PetscCall(SetupStrongBC_Ceed(ceed, dm, honee, problem));
  if (app_ctx->sgs_train_enable) PetscCall(SGS_DD_TrainingSetup(ceed, honee));
  if (app_ctx->divFdiffproj_method != DIV_DIFF_FLUX_PROJ_NONE) PetscCall(DivDiffFluxProjectionSetup(honee, honee->diff_flux_proj));

  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedVectorDestroy(&jac_data));
  PetscCallCeed(ceed, CeedVectorDestroy(&x_coord));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_x));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_jd_i));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_diff_flux));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_diff_flux));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_q));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_x));
  PetscCallCeed(ceed, CeedVectorDestroy(&div_diff_flux_ceed));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_ijacobian_vol));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_ifunction_vol));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_rhs_vol));
  PetscFunctionReturn(PETSC_SUCCESS);
}
