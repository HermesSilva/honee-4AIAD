// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include <navierstokes.h>

#define ESSENTIAL_BC_MAX_COMPONENTS 16

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

static PetscErrorCode HoneeBCCreatePrefix_Private(const char name[], char **cl_prefix) {
  PetscSizeT name_len, prefix_len;

  PetscFunctionBeginUser;
  PetscCall(PetscStrlen(name, &name_len));
  prefix_len = name_len + 5;
  PetscCall(PetscCalloc1(prefix_len, cl_prefix));
  PetscCall(PetscSNPrintf(*cl_prefix, prefix_len, "bc_%s_", name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create command line prefix for bc_def

  @param[in]  bc_def    `BCDefinition`
  @param[out] cl_prefix Prefex to be used with command line arguments
**/
PetscErrorCode HoneeBCCreatePrefix(BCDefinition bc_def, char **cl_prefix) {
  const char *name;
  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetInfo(bc_def, &name, NULL, NULL));
  PetscCall(HoneeBCCreatePrefix_Private(name, cl_prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
   @brief Create and setup `BCDefinition`s from commandline options

   @param[in]     honee   `Honee`
   @param[in,out] problem `ProblemData`
   @param[in]     app_ctx `AppCtx`
**/
PetscErrorCode BoundaryConditionSetUp(Honee honee, ProblemData problem, AppCtx app_ctx) {
  MPI_Comm       comm = honee->comm;
  PetscBool      flg;
  PetscSegBuffer wall_face_ids_sb;
  char          *bc_names[HONEE_MAX_BCDEFS] = {NULL};

  PetscFunctionBeginUser;

  PetscOptionsBegin(comm, NULL, "Boundary Condition Options", NULL);
  problem->num_bc_defs = HONEE_MAX_BCDEFS;
  PetscCall(PetscOptionsStringArray("-bc_names", "Name of boundary condition names to be read", __func__, bc_names, &problem->num_bc_defs, &flg));
  PetscOptionsEnd();

  PetscCall(PetscSegBufferCreate(sizeof(PetscInt), 4, &wall_face_ids_sb));
  PetscCall(PetscCalloc1(problem->num_bc_defs, &problem->bc_defs));
  for (PetscInt b = 0; b < problem->num_bc_defs; b++) {
    const char  *name   = bc_names[b];
    BCDefinition bc_def = NULL;
    size_t       len;
    char         buffer[PETSC_MAX_OPTION_NAME];
    char        *cl_prefix;
    PetscInt     essential_comps[ESSENTIAL_BC_MAX_COMPONENTS], num_essential_comps = PETSC_STATIC_ARRAY_LENGTH(essential_comps);
    PetscScalar  essential_values[ESSENTIAL_BC_MAX_COMPONENTS];
    PetscInt     num_essential_values = PETSC_STATIC_ARRAY_LENGTH(essential_values);
    PetscBool    essential_comps_set, essential_values_set;

    if (name[0] == '\0') continue;  // skip empty names
    for (PetscInt i = 0; i < b; i++) {
      PetscBool match;
      PetscCall(PetscStrcmp(name, problem->bc_defs[i]->name, &match));
      PetscCheck(!match, comm, PETSC_ERR_USER_INPUT, "Cannot have repeat values in -bc_names found duplicate of %s", name);
    }

    PetscCall(PetscSNPrintf(buffer, PETSC_STATIC_ARRAY_LENGTH(buffer), "Boundary Condition Options: %s", name));
    PetscCall(HoneeBCCreatePrefix_Private(name, &cl_prefix));
    PetscOptionsBegin(comm, cl_prefix, buffer, NULL);
    PetscCall(PetscFree(cl_prefix));

    PetscCall(PetscOptionsBCDefinition("-face_ids", "Face IDs to apply boundary condition to", NULL, name, &problem->bc_defs[b], NULL));
    PetscCheck(problem->bc_defs[b], comm, PETSC_ERR_USER_INPUT, "Could not find face_ids for bc_name \"%s\"", name);
    bc_def = problem->bc_defs[b];
    len    = PETSC_STATIC_ARRAY_LENGTH(buffer);
    PetscCall(PetscOptionsString("-natural_type", "Name of natural BC to apply", NULL, buffer, buffer, len, &flg));
    if (flg) {
      PetscCall(PetscStrlen(buffer, &len));
      PetscCheck(len < PETSC_STATIC_ARRAY_LENGTH(buffer), comm, PETSC_ERR_USER_INPUT, "Name in -%s_natural_type longer than %zu, only recieved %s",
                 name, PETSC_STATIC_ARRAY_LENGTH(buffer), buffer);
      PetscCall(BCDefinitionSetNaturalType(bc_def, buffer));
    }

    PetscCall(PetscOptionsIntArray("-essential_comps", "An array of constrained component numbers", NULL, essential_comps, &num_essential_comps,
                                   &essential_comps_set));
    PetscCall(PetscOptionsScalarArray("-essential_values", "Values for the constrained components", NULL, essential_values, &num_essential_values,
                                      &essential_values_set));
    PetscCheck(!essential_values_set || essential_comps_set, comm, PETSC_ERR_USER_INPUT, "-bc_%s_essential_values requires -bc_%s_essential_comps",
               name, name);
    if (essential_values_set) {
      PetscCheck(num_essential_values == num_essential_comps, comm, PETSC_ERR_USER_INPUT,
                 "-bc_%s_essential_values has %" PetscInt_FMT " entries, but -bc_%s_essential_comps has %" PetscInt_FMT, name, num_essential_values,
                 name, num_essential_comps);
    }
    if (essential_comps_set) {
      PetscCall(BCDefinitionSetEssential(bc_def, num_essential_comps, essential_comps, essential_values_set ? essential_values : NULL));
    }

    PetscBool is_viscous_wall = PETSC_FALSE;
    PetscCall(PetscOptionsBool("-is_viscous_wall", "Whether to treat the boundaries labeled here as a viscous wall", NULL, is_viscous_wall,
                               &is_viscous_wall, NULL));
    if (is_viscous_wall) {
      const PetscInt *label_values;
      PetscInt       *label_values_seg, num_label_values;

      PetscCall(BCDefinitionGetInfo(bc_def, NULL, &num_label_values, &label_values));
      PetscCall(PetscSegBufferGetInts(wall_face_ids_sb, num_label_values, &label_values_seg));
      PetscCall(PetscArraycpy(label_values_seg, label_values, num_label_values));
    }

    {  // Add HoneeBCStruct to bc_def
      HoneeBCStruct honee_bc;

      PetscCall(PetscNew(&honee_bc));
      honee_bc->honee = honee;
      PetscCall(BCDefinitionSetContext(bc_def, honee_bc, (PetscCtxDestroyFn *)HoneeBCDestroy));
    }
    PetscOptionsEnd();
    PetscCall(PetscFree(bc_names[b]));
  }

  {  // Setup wall distance
    PetscInt  *wall_face_ids;
    PetscCount num_wall_face_ids_count;

    PetscCall(PetscSegBufferGetSize(wall_face_ids_sb, &num_wall_face_ids_count));
    PetscCall(PetscSegBufferExtractAlloc(wall_face_ids_sb, &wall_face_ids));
    PetscCall(PetscSegBufferDestroy(&wall_face_ids_sb));
    if (num_wall_face_ids_count) {
      PetscCall(HoneeWallDistanceSetup(honee, num_wall_face_ids_count, wall_face_ids));
      PetscCall(HoneeWallForceSetup(honee, num_wall_face_ids_count, wall_face_ids));
    }

    PetscCall(PetscFree(wall_face_ids));
  }

  // Use legacy boundary condition options only if new options are not used at all
  if (problem->num_bc_defs == 0) {
    BCDefinition   bc_def;
    PetscSegBuffer bc_defs_seg;

    PetscCall(PetscSegBufferCreate(sizeof(BCDefinition), 4, &bc_defs_seg));
    PetscOptionsBegin(comm, NULL, "Boundary Condition Options", NULL);
    PetscCall(PetscOptionsBCDefinition("-bc_wall", "Face IDs to apply wall BC", NULL, "wall", &bc_def, NULL));
    PetscCall(AddBCDefinitionToSegBuffer(bc_def, bc_defs_seg));

    if (bc_def) {  // Setup wall distance based on wall bc
      const PetscInt *label_values;
      PetscInt        num_label_values;
      PetscCall(BCDefinitionGetInfo(bc_def, NULL, &num_label_values, &label_values));
      PetscCall(HoneeWallDistanceSetup(honee, num_label_values, label_values));
      PetscCall(HoneeWallForceSetup(honee, num_label_values, label_values));
    }
    if (bc_def) {
      PetscInt num_essential_comps = 16, essential_comps[16];

      PetscCall(PetscOptionsIntArray("-wall_comps", "An array of constrained component numbers", NULL, essential_comps, &num_essential_comps, &flg));
      PetscCall(BCDefinitionSetEssential(bc_def, num_essential_comps, essential_comps, NULL));
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

          PetscCall(BCDefinitionSetEssential(bc_def, 1, essential_comps, NULL));
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

    PetscCount num_bc_defs_count;
    PetscCall(PetscSegBufferGetSize(bc_defs_seg, &num_bc_defs_count));
    problem->num_bc_defs = num_bc_defs_count;
    PetscCall(PetscSegBufferExtractAlloc(bc_defs_seg, &problem->bc_defs));
    PetscCall(PetscSegBufferDestroy(&bc_defs_seg));

    if (problem->num_bc_defs) {
      PetscCall(PetscPrintf(comm, "WARNING: Legacy style of setting boundary conditions is deprecated and will be removed in the future"));
    }

    // TEMPORARY before major BC refactoring.
    // In future, each natural BC (including example BCs) will have it's own name independent from the bc_def->name.
    for (PetscInt i = 0; i < problem->num_bc_defs; i++) {
      BCDefinition bc_def = problem->bc_defs[i];
      PetscBool    has_no_natural_bc;

      PetscCall(PetscStrcmpAny(bc_def->name, &has_no_natural_bc, "wall", "symmetry", ""));
      if (has_no_natural_bc) continue;
      PetscCall(BCDefinitionSetNaturalType(bc_def, bc_def->name));
    }
    for (PetscInt i = 0; i < problem->num_bc_defs; i++) {
      HoneeBCStruct honee_bc;
      BCDefinition  bc_def = problem->bc_defs[i];

      PetscCall(PetscNew(&honee_bc));
      honee_bc->honee = honee;
      PetscCall(BCDefinitionSetContext(bc_def, honee_bc, (PetscCtxDestroyFn *)HoneeBCDestroy));
    }
  }

  // TODO: Verify that the BCDefinition don't have overlapping claims to boundary faces
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
  PetscInt      dim, dim_sur, num_comp_x, num_comp_q;
  CeedInt       q_data_size_sur, num_comps_jac_data;
  HoneeBCStruct honee_bc;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetDM(bc_def, &dm));
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  ceed               = honee_bc->honee->ceed;
  num_comps_jac_data = honee_bc->num_comps_jac_data;

  PetscCall(DMGetDimension(dm, &dim));
  dim_sur = dim - PLEXHEIGHT_FACE;
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
  PetscInt      dim, dim_sur, num_comp_x, num_comp_q;
  CeedInt       q_data_size_sur, num_comps_jac_data;
  HoneeBCStruct honee_bc;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetDM(bc_def, &dm));
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  ceed               = honee_bc->honee->ceed;
  num_comps_jac_data = honee_bc->num_comps_jac_data;

  PetscCall(DMGetDimension(dm, &dim));
  dim_sur = dim - PLEXHEIGHT_FACE;
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
  Ceed                ceed;
  DM                  dm;
  HoneeBCStruct       honee_bc;
  PetscInt            dim, num_comp_x, num_comp_q;
  CeedInt             q_data_size, num_comps_jac_data;
  CeedVector          q_data, jac_data, x_coord;
  CeedOperator        sub_op_ifunc_;
  CeedElemRestriction elem_restr_qd, elem_restr_q, elem_restr_x, elem_restr_jac;
  CeedBasis           basis_x, basis_q;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetDM(bc_def, &dm));
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  ceed               = honee_bc->honee->ceed;
  num_comps_jac_data = honee_bc->num_comps_jac_data;

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateNumComps(dm, &num_comp_x));
  PetscCall(DMGetFieldNumComps(dm, bc_def->dm_field, &num_comp_q));

  PetscCall(DMPlexCeedBasisCreate(ceed, dm, domain_label, label_value, PLEXHEIGHT_FACE, bc_def->dm_field, &basis_q));
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, dm, domain_label, label_value, PLEXHEIGHT_FACE, bc_def->dm_field, &elem_restr_q));
  PetscCall(DMPlexCeedCoordinateCreateField(ceed, dm, domain_label, label_value, PLEXHEIGHT_FACE, &elem_restr_x, &basis_x, &x_coord));
  PetscCall(QDataBoundaryGet(ceed, dm, domain_label, label_value, &elem_restr_qd, &q_data, &q_data_size));
  if (num_comps_jac_data > 0) {
    PetscCall(DMPlexCeedElemRestrictionQDataCreate(ceed, dm, domain_label, label_value, PLEXHEIGHT_FACE, num_comps_jac_data, &elem_restr_jac));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_jac, &jac_data, NULL));
  }

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_ifunc, NULL, NULL, &sub_op_ifunc_));
  PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ifunc_, "q", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ifunc_, "Grad_q", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ifunc_, "surface qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ifunc_, "x", elem_restr_x, basis_x, x_coord));
  PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ifunc_, "v", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  if (num_comps_jac_data > 0) {
    PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ifunc_, "surface jacobian data", elem_restr_jac, CEED_BASIS_NONE, jac_data));
  }

  PetscCallCeed(ceed, CeedOperatorCompositeAddSub(op_ifunc, sub_op_ifunc_));
  *sub_op_ifunc = sub_op_ifunc_;

  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedVectorDestroy(&x_coord));
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
  Ceed                ceed;
  DM                  dm;
  HoneeBCStruct       honee_bc;
  PetscInt            dim, num_comp_x, num_comp_q;
  CeedInt             q_data_size;
  CeedVector          q_data, jac_data, x_coord;
  CeedElemRestriction elem_restr_qd, elem_restr_q, elem_restr_x, elem_restr_jac;
  CeedOperator        sub_op_ijac;
  CeedBasis           basis_x, basis_q;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  PetscBool use_jac_data = honee_bc->num_comps_jac_data > 0;
  PetscCall(BCDefinitionGetDM(bc_def, &dm));
  ceed = honee_bc->honee->ceed;

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateNumComps(dm, &num_comp_x));
  PetscCall(DMGetFieldNumComps(dm, bc_def->dm_field, &num_comp_q));

  PetscCall(DMPlexCeedCoordinateCreateField(ceed, dm, domain_label, label_value, PLEXHEIGHT_FACE, &elem_restr_x, &basis_x, &x_coord));
  PetscCall(QDataBoundaryGet(ceed, dm, domain_label, label_value, &elem_restr_qd, &q_data, &q_data_size));

  {  // Get restriction and basis from the IFunction function
    CeedOperatorField op_field;

    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_op_ifunc, "q", &op_field));
    PetscCallCeed(ceed, CeedOperatorFieldGetData(op_field, NULL, &elem_restr_q, &basis_q, NULL));
    if (use_jac_data) {
      PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_op_ifunc, "surface jacobian data", &op_field));
      PetscCallCeed(ceed, CeedOperatorFieldGetData(op_field, NULL, &elem_restr_jac, NULL, &jac_data));
    }
  }

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_ijac, NULL, NULL, &sub_op_ijac));
  PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ijac, "dq", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ijac, "Grad_dq", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ijac, "surface qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(sub_op_ijac, "x", elem_restr_x, basis_x, x_coord));
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
  PetscCallCeed(ceed, CeedVectorDestroy(&x_coord));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_q));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_x));
  PetscCallCeed(ceed, CeedOperatorDestroy(&sub_op_ijac));
  PetscFunctionReturn(PETSC_SUCCESS);
}
