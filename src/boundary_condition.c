// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include <navierstokes.h>
#include "petscerror.h"

/**
   @brief Add `BCDefinition` to a `PetscSegBuffer`

   @param[in]     bc_def      `BCDefinition` to add
   @param[in,out] bc_defs_seg `PetscSegBuffer` to add to
**/
static PetscErrorCode AddBCDefinitionToSegBuffer(BCDefinition bc_def, PetscSegBuffer bc_defs_seg) {
  BCDefinition *bc_def_ptr;

  PetscFunctionBeginUser;
  if (bc_def == NULL) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSegBufferGet(bc_defs_seg, 1, &bc_def_ptr));
  *bc_def_ptr = bc_def;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
   @brief Create and setup `BCDefinition`s from commandline options

   @param[in]     honee   `Honee`
   @param[in,out] problem `ProblemData`
   @param[in]     app_ctx `AppCtx`
**/
PetscErrorCode BoundaryConditionSetUp(Honee honee, ProblemData problem, AppCtx app_ctx) {
  PetscSegBuffer bc_defs_seg;
  PetscBool      flg;
  BCDefinition   bc_def;

  PetscFunctionBeginUser;
  PetscCall(PetscSegBufferCreate(sizeof(BCDefinition), 4, &bc_defs_seg));

  PetscOptionsBegin(honee->comm, NULL, "Boundary Condition Options", NULL);

  PetscCall(PetscOptionsBCDefinition("-bc_wall", "Face IDs to apply wall BC", NULL, "wall", &bc_def, NULL));
  PetscCall(AddBCDefinitionToSegBuffer(bc_def, bc_defs_seg));
  if (bc_def) {
    PetscInt num_essential_comps = 16, essential_comps[16];

    PetscCall(PetscOptionsIntArray("-wall_comps", "An array of constrained component numbers", NULL, essential_comps, &num_essential_comps, &flg));
    PetscCall(BCDefinitionSetEssential(bc_def, num_essential_comps, essential_comps));

    app_ctx->wall_forces.num_wall = bc_def->num_label_values;
    PetscCall(PetscMalloc1(bc_def->num_label_values, &app_ctx->wall_forces.walls));
    PetscCall(PetscArraycpy(app_ctx->wall_forces.walls, bc_def->label_values, bc_def->num_label_values));
  }

  {  // Symmetry Boundary Conditions
    const char *deprecated[3] = {"-bc_slip_x", "-bc_slip_y", "-bc_slip_z"};
    const char *flags[3]      = {"-bc_symmetry_x", "-bc_symmetry_y", "-bc_symmetry_z"};

    for (PetscInt j = 0; j < 3; j++) {
      PetscCall(PetscOptionsDeprecated(deprecated[j], flags[j], "libCEED 0.12.0",
                                       "Use -bc_symmetry_[x,y,z] for direct equivalency, or -bc_slip for weak, Riemann-based, direction-invariant "
                                       "slip/no-penatration boundary conditions"));
      PetscCall(PetscOptionsBCDefinition(flags[j], "Face IDs to apply symmetry BC", NULL, "symmetry", &bc_def, NULL));
      if (!bc_def) {
        PetscCall(PetscOptionsBCDefinition(deprecated[j], "Face IDs to apply symmetry BC", NULL, "symmetry", &bc_def, NULL));
      }
      PetscCall(AddBCDefinitionToSegBuffer(bc_def, bc_defs_seg));
      if (bc_def) {
        PetscInt essential_comps[1] = {j + 1};

        PetscCall(BCDefinitionSetEssential(bc_def, 1, essential_comps));
      }
    }
  }

  PetscCall(PetscOptionsBCDefinition("-bc_inflow", "Face IDs to apply inflow BC", NULL, "inflow", &bc_def, NULL));
  PetscCall(AddBCDefinitionToSegBuffer(bc_def, bc_defs_seg));

  PetscCall(PetscOptionsBCDefinition("-bc_outflow", "Face IDs to apply outflow BC", NULL, "outflow", &bc_def, NULL));
  PetscCall(AddBCDefinitionToSegBuffer(bc_def, bc_defs_seg));

  PetscCall(PetscOptionsBCDefinition("-bc_freestream", "Face IDs to apply freestream BC", NULL, "freestream", &bc_def, NULL));
  PetscCall(AddBCDefinitionToSegBuffer(bc_def, bc_defs_seg));

  PetscCall(PetscOptionsBCDefinition("-bc_slip", "Face IDs to apply slip BC", NULL, "slip", &bc_def, NULL));
  PetscCall(AddBCDefinitionToSegBuffer(bc_def, bc_defs_seg));

  PetscOptionsEnd();

  PetscCall(PetscSegBufferGetSize(bc_defs_seg, &problem->num_bc_defs));
  PetscCall(PetscSegBufferExtractAlloc(bc_defs_seg, &problem->bc_defs));
  PetscCall(PetscSegBufferDestroy(&bc_defs_seg));

  //TODO: Verify that the BCDefinition don't have overlapping claims to boundary faces
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Destroy `HoneeBCStruct` object

  @param[in] ctx Pointer to `HoneeBCStruct`
**/
PetscErrorCode HoneeBCDestroy(void **ctx) {
  HoneeBCStruct honee_bc = *(HoneeBCStruct *)ctx;
  Ceed          ceed     = honee_bc->honee->ceed;

  PetscFunctionBeginUser;
  if (honee_bc->qfctx) PetscCallCeed(ceed, CeedQFunctionContextDestroy(&honee_bc->qfctx));
  if (honee_bc->DestroyCtx) PetscCall((*honee_bc->DestroyCtx)(&honee_bc->ctx));
  PetscCall(PetscFree(honee_bc));
  *ctx = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create a QFunction matching the "standard" HONEE inputs for IFunctions

  This assumes the `bc_def` context is a `HoneeBCStruct` and inputs/outputs of the IFunction QFunction are in the correct order.

  @param[in]  bc_def      `BCDefinition` that hosts the IFunction
  @param[in]  qf_func_ptr `CeedQFunctionUser` for the IFunction
  @param[in]  qf_loc      Absolute path to source of `CeedQFunctionUser`
  @param[in]  qfctx       `CeedQFunctionContext` for the IFunction (also shared with the IJacobian, if applicable)
  @param[out] qf_ifunc    QFunction for the IFunction
**/
PetscErrorCode HoneeBCCreateIFunctionQF(BCDefinition bc_def, CeedQFunctionUser qf_func_ptr, const char *qf_loc, CeedQFunctionContext qfctx,
                                        CeedQFunction *qf_ifunc) {
  Ceed          ceed;
  DM            dm;
  PetscInt      dim, dim_sur, height = 1, num_comp_x, num_comp_q;
  CeedInt       q_data_size_sur, num_comps_jac_data;
  HoneeBCStruct honee_bc;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetDM(bc_def, &dm));
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  ceed               = honee_bc->honee->ceed;
  num_comps_jac_data = honee_bc->num_comps_jac_data;

  PetscCall(DMGetDimension(dm, &dim));
  dim_sur = dim - height;
  PetscCall(QDataBoundaryGetNumComponents(dm, &q_data_size_sur));
  PetscCall(DMGetCoordinateNumComps(dm, &num_comp_x));
  {
    PetscSection section;

    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(PetscSectionGetFieldComponents(section, bc_def->dm_field, &num_comp_q));
  }

  PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, qf_func_ptr, qf_loc, qf_ifunc));
  PetscCallCeed(ceed, CeedQFunctionSetContext(*qf_ifunc, qfctx));
  PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(*qf_ifunc, 0));
  PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_ifunc, "q", num_comp_q, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_ifunc, "Grad_q", num_comp_q * dim_sur, CEED_EVAL_GRAD));
  PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_ifunc, "surface qdata", q_data_size_sur, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_ifunc, "x", num_comp_x, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(*qf_ifunc, "v", num_comp_q, CEED_EVAL_INTERP));
  if (num_comps_jac_data) PetscCallCeed(ceed, CeedQFunctionAddOutput(*qf_ifunc, "surface jacobian data", num_comps_jac_data, CEED_EVAL_NONE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create a QFunction matching the "standard" HONEE inputs for IJacobian

  This assumes the `bc_def` context is a `HoneeBCStruct` and inputs/outputs of the IJacobian QFunction are in the correct order.

  @param[in]  bc_def      `BCDefinition` that hosts the IJacobian
  @param[in]  qf_func_ptr `CeedQFunctionUser` for the IJacobian
  @param[in]  qf_loc      Absolute path to source of `CeedQFunctionUser`
  @param[in]  qfctx       `CeedQFunctionContext` for the IJacobian (also shared with the IFunction)
  @param[out] qf_ijac     QFunction for the IJacobian
**/
PetscErrorCode HoneeBCCreateIJacobianQF(BCDefinition bc_def, CeedQFunctionUser qf_func_ptr, const char *qf_loc, CeedQFunctionContext qfctx,
                                        CeedQFunction *qf_ijac) {
  Ceed          ceed;
  DM            dm;
  PetscInt      dim, dim_sur, height = 1, num_comp_x, num_comp_q;
  CeedInt       q_data_size_sur, num_comps_jac_data;
  HoneeBCStruct honee_bc;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetDM(bc_def, &dm));
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  ceed               = honee_bc->honee->ceed;
  num_comps_jac_data = honee_bc->num_comps_jac_data;

  PetscCall(DMGetDimension(dm, &dim));
  dim_sur = dim - height;
  PetscCall(QDataBoundaryGetNumComponents(dm, &q_data_size_sur));
  PetscCall(DMGetCoordinateNumComps(dm, &num_comp_x));
  {
    PetscSection section;

    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(PetscSectionGetFieldComponents(section, bc_def->dm_field, &num_comp_q));
  }

  PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, qf_func_ptr, qf_loc, qf_ijac));
  PetscCallCeed(ceed, CeedQFunctionSetContext(*qf_ijac, qfctx));
  PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(*qf_ijac, 0));
  PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_ijac, "dq", num_comp_q, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_ijac, "Grad_dq", num_comp_q * dim_sur, CEED_EVAL_GRAD));
  PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_ijac, "surface qdata", q_data_size_sur, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_ijac, "x", num_comp_x, CEED_EVAL_INTERP));
  if (num_comps_jac_data) PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_ijac, "surface jacobian data", num_comps_jac_data, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(*qf_ijac, "v", num_comp_q, CEED_EVAL_INTERP));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Setups and adds IFunction operator to given composite operator

  @param[in]     bc_def       `BCDefinition` for the boundary condition IFunction
  @param[in]     domain_label `DMLabel` for the face orientation label
  @param[in]     label_value  Orientation value
  @param[in]     qf_ifunc     QFunction for the IFunction
  @param[in,out] op_ifunc     Composite operator to be added to
  @param[out]    sub_op_ifunc Non-composite operator that is added in this function. To be passed to `HoneeBCAddIJacobianOp()`
**/
PetscErrorCode HoneeBCAddIFunctionOp(BCDefinition bc_def, DMLabel domain_label, PetscInt label_value, CeedQFunction qf_ifunc, CeedOperator op_ifunc,
                                     CeedOperator *sub_op_ifunc) {
  Honee               honee;
  Ceed                ceed;
  DM                  dm, dm_coord;
  HoneeBCStruct       honee_bc;
  PetscInt            dim, height = 1, num_comp_x, num_comp_q;
  CeedInt             q_data_size, num_comps_jac_data;
  CeedVector          q_data, jac_data;
  CeedOperator        sub_op_ifunc_;
  CeedElemRestriction elem_restr_qd, elem_restr_q, elem_restr_x, elem_restr_jac;
  CeedBasis           basis_x, basis_q;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetDM(bc_def, &dm));
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  honee              = honee_bc->honee;
  ceed               = honee_bc->honee->ceed;
  num_comps_jac_data = honee_bc->num_comps_jac_data;

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(QDataBoundaryGetNumComponents(dm, &q_data_size));
  PetscCall(DMGetCoordinateNumComps(dm, &num_comp_x));

  {
    PetscSection section;

    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(PetscSectionGetFieldComponents(section, bc_def->dm_field, &num_comp_q));
  }

  PetscCall(DMGetCoordinateDM(dm, &dm_coord));
  PetscCall(DMPlexCeedBasisCreate(ceed, dm, domain_label, label_value, height, bc_def->dm_field, &basis_q));
  PetscCall(DMPlexCeedBasisCreate(ceed, dm_coord, domain_label, label_value, height, bc_def->dm_field, &basis_x));
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, dm, domain_label, label_value, height, bc_def->dm_field, &elem_restr_q));
  PetscCall(DMPlexCeedElemRestrictionCoordinateCreate(ceed, dm, domain_label, label_value, height, &elem_restr_x));
  if (num_comps_jac_data > 0) {
    PetscCall(DMPlexCeedElemRestrictionQDataCreate(ceed, dm, domain_label, label_value, height, num_comps_jac_data, &elem_restr_jac));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_jac, &jac_data, NULL));
  }

  PetscCall(QDataBoundaryGet(ceed, dm, domain_label, label_value, elem_restr_x, basis_x, honee->x_coord, &elem_restr_qd, &q_data, &q_data_size));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_ifunc, NULL, NULL, &sub_op_ifunc_));
  PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ifunc_, "q", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ifunc_, "Grad_q", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ifunc_, "surface qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ifunc_, "x", elem_restr_x, basis_x, honee->x_coord));
  PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ifunc_, "v", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  if (num_comps_jac_data > 0)
    PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ifunc_, "surface jacobian data", elem_restr_jac, CEED_BASIS_NONE, jac_data));

  PetscCallCeed(ceed, CeedOperatorCompositeAddSub(op_ifunc, sub_op_ifunc_));
  *sub_op_ifunc = sub_op_ifunc_;

  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_q));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_x));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_x));
  if (num_comps_jac_data > 0) {
    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_jac));
    PetscCallCeed(ceed, CeedVectorDestroy(&jac_data));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Setups and adds IJacobian operator to given composite operator

  The field data (element restriction, vector, etc) is read from `sub_op_ifunc` and used for the IJacobian.

  @param[in]     bc_def       `BCDefinition` for the boundary condition IJacobian
  @param[out]    sub_op_ifunc IFunction operator corresponding to this IJacobian operator.
  @param[in]     domain_label `DMLabel` for the face orientation label
  @param[in]     label_value  Orientation value
  @param[in]     qf_ijac      QFunction for the IJacobian
  @param[in,out] op_ijac      Composite operator to be added to
**/
PetscErrorCode HoneeBCAddIJacobianOp(BCDefinition bc_def, CeedOperator sub_op_ifunc, DMLabel domain_label, PetscInt label_value,
                                     CeedQFunction qf_ijac, CeedOperator op_ijac) {
  Honee               honee;
  Ceed                ceed;
  DM                  dm, dm_coord;
  HoneeBCStruct       honee_bc;
  PetscInt            dim, height = 1, num_comp_x, num_comp_q;
  CeedInt             q_data_size;
  CeedVector          q_data, jac_data;
  CeedElemRestriction elem_restr_qd, elem_restr_q, elem_restr_x, elem_restr_jac;
  CeedOperator        sub_op_ijac;
  CeedBasis           basis_x, basis_q;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  PetscBool use_jac_data = honee_bc->num_comps_jac_data > 0;
  PetscCall(BCDefinitionGetDM(bc_def, &dm));
  honee = honee_bc->honee;
  ceed  = honee_bc->honee->ceed;

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(QDataBoundaryGetNumComponents(dm, &q_data_size));
  PetscCall(DMGetCoordinateNumComps(dm, &num_comp_x));

  {
    PetscSection section;

    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(PetscSectionGetFieldComponents(section, bc_def->dm_field, &num_comp_q));
  }

  PetscCall(DMGetCoordinateDM(dm, &dm_coord));
  PetscCall(DMPlexCeedBasisCreate(ceed, dm, domain_label, label_value, height, bc_def->dm_field, &basis_q));
  PetscCall(DMPlexCeedBasisCreate(ceed, dm_coord, domain_label, label_value, height, bc_def->dm_field, &basis_x));

  {  // Get restriction and basis from the RHS function
    CeedOperatorField op_field;

    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_op_ifunc, "q", &op_field));
    PetscCallCeed(ceed, CeedOperatorFieldGetElemRestriction(op_field, &elem_restr_q));

    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_op_ifunc, "x", &op_field));
    PetscCallCeed(ceed, CeedOperatorFieldGetElemRestriction(op_field, &elem_restr_x));

    if (use_jac_data) {
      PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_op_ifunc, "surface jacobian data", &op_field));
      PetscCallCeed(ceed, CeedOperatorFieldGetData(op_field, NULL, &elem_restr_jac, NULL, &jac_data));
    }
  }

  PetscCall(QDataBoundaryGet(ceed, dm, domain_label, label_value, elem_restr_x, basis_x, honee->x_coord, &elem_restr_qd, &q_data, &q_data_size));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_ijac, NULL, NULL, &sub_op_ijac));
  PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ijac, "dq", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ijac, "Grad_dq", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ijac, "surface qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ijac, "x", elem_restr_x, basis_x, honee->x_coord));
  if (use_jac_data) PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ijac, "surface jacobian data", elem_restr_jac, CEED_BASIS_NONE, jac_data));
  PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ijac, "v", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedOperatorCompositeAddSub(op_ijac, sub_op_ijac));

  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_x));
  if (use_jac_data) {
    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_jac));
    PetscCallCeed(ceed, CeedVectorDestroy(&jac_data));
  }
  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_q));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_x));
  PetscCallCeed(ceed, CeedOperatorDestroy(&sub_op_ijac));
  PetscFunctionReturn(PETSC_SUCCESS);
}
