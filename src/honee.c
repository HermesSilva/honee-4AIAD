// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include <navierstokes.h>

PetscClassId HONEE_CLASSID;

// @brief Initalize `HONEE` class.
static PetscErrorCode HoneeInitClass_Private() {
  static PetscBool registered = PETSC_FALSE;

  PetscFunctionBeginUser;
  if (registered) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscClassIdRegister("Honee", &HONEE_CLASSID));
  registered = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Initialize `Honee` object

  @param[in]  comm  `MPI_Comm` for the object
  @param[out] honee The initialized `Honee` object
**/
PetscErrorCode HoneeInit(MPI_Comm comm, Honee *honee) {
  Honee       honee_;
  AppCtx      app_ctx;
  ProblemData problem;
  Physics     phys_ctx;
  Units       units;

  PetscFunctionBeginUser;
  PetscCall(HoneeInitClass_Private());
  PetscCall(PetscHeaderCreate(honee_, HONEE_CLASSID, "Honee", "HONEE", "Honee", comm, HoneeDestroy, NULL));
  PetscCall(PetscNew(&app_ctx));
  PetscCall(PetscNew(&problem));
  PetscCall(PetscNew(&phys_ctx));
  PetscCall(PetscNew(&units));
  honee_->app_ctx          = app_ctx;
  honee_->units            = units;
  honee_->phys             = phys_ctx;
  honee_->problem_data     = problem;
  problem->set_bc_from_ics = PETSC_TRUE;
  honee_->comm             = PETSC_COMM_WORLD;
  honee_->start_time       = time(NULL);

  PetscCall(RegisterLogEvents());

  *honee = honee_;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Destroy `Honee` object

  @param[in] honee Object to be destroyed
**/
PetscErrorCode HoneeDestroy(Honee *honee) {
  Honee honee_ = *honee;
  Ceed  ceed;

  PetscFunctionBeginUser;
  if (!honee_) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(honee_, HONEE_CLASSID, 1);

  ceed                = honee_->ceed;
  MPI_Comm    comm    = honee_->comm;
  AppCtx      app_ctx = honee_->app_ctx;
  ProblemData problem = honee_->problem_data;

  PetscCall(NodalProjectionDataDestroy(honee_->grad_velo_proj));
  PetscCall(SgsDDDataDestroy(honee_->sgs_dd_data));
  PetscCall(DifferentialFilterDataDestroy(honee_->diff_filter));
  PetscCall(SGS_DD_TrainingDataDestroy(honee_->sgs_dd_train));
  PetscCall(SmartSimDataDestroy(honee_->smartsim));
  PetscCall(QDataClearStoredData());
  PetscCall(DivDiffFluxProjectionDataDestroy(honee_->diff_flux_proj));

  // -- Vectors
  PetscCallCeed(ceed, CeedVectorDestroy(&honee_->x_coord));
  PetscCallCeed(ceed, CeedVectorDestroy(&honee_->q_ceed));
  PetscCallCeed(ceed, CeedVectorDestroy(&honee_->q_dot_ceed));
  PetscCallCeed(ceed, CeedVectorDestroy(&honee_->g_ceed));

  // -- Bases
  PetscCallCeed(ceed, CeedBasisDestroy(&honee_->basis_q));
  PetscCallCeed(ceed, CeedBasisDestroy(&honee_->basis_x));

  // -- Restrictions
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&honee_->elem_restr_q));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&honee_->elem_restr_x));

  // Destroy QFunction contexts after using
  {
    PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->ics.qfctx));
    PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->apply_vol_rhs.qfctx));
    PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->apply_vol_ifunction.qfctx));
    PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->apply_vol_ijacobian.qfctx));
  }

  // -- Operators
  PetscCall(OperatorApplyContextDestroy(honee_->op_ics_ctx));
  PetscCall(OperatorApplyContextDestroy(honee_->op_rhs_ctx));
  PetscCall(OperatorApplyContextDestroy(honee_->op_strong_bc_ctx));
  PetscCallCeed(ceed, CeedOperatorDestroy(&honee_->op_ifunction));

  // -- Ceed
  PetscCheck(CeedDestroy(&ceed) == CEED_ERROR_SUCCESS, comm, PETSC_ERR_LIB, "Destroying Ceed object failed");

  // ---------------------------------------------------------------------------
  // Clean up PETSc
  // ---------------------------------------------------------------------------
  // -- Vectors
  PetscCall(VecDestroy(&honee_->Q_loc));
  PetscCall(VecDestroy(&honee_->Q_dot_loc));

  PetscCall(KSPDestroy(&honee_->mass_ksp));

  // -- Matrices
  PetscCall(MatDestroy(&honee_->interp_viz));
  PetscCall(MatDestroy(&honee_->mat_ijacobian));

  // -- DM
  PetscCall(DMDestroy(&honee_->dm_viz));

  // -- Function list
  PetscCall(PetscFunctionListDestroy(&app_ctx->problems));

  PetscCall(PetscFree(app_ctx->amat_type));
  PetscCall(PetscFree(app_ctx->wall_forces.walls));
  PetscCall(PetscViewerDestroy(&app_ctx->wall_forces.viewer));

  // -- Structs
  for (PetscInt i = 0; i < problem->num_bc_defs; i++) {
    PetscCall(BCDefinitionDestroy(&problem->bc_defs[i]));
  }
  PetscCall(PetscFree(problem->bc_defs));
  PetscCall(PetscFree(honee_->units));
  PetscCall(PetscFree(problem));
  PetscCall(PetscFree(honee_->phys));
  PetscCall(PetscFree(app_ctx));
  PetscCall(PetscHeaderDestroy(&honee_));
  *honee = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
