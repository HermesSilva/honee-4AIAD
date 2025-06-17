// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include <bc_definition.h>
#include <dm-utils.h>
#include <petsc-ceed.h>
#include <petsc/private/petscimpl.h>

/**
   @brief Create `BCDefinition`

   @param[in]  name             Name of the boundary condition
   @param[in]  num_label_values Number of `DMLabel` values
   @param[in]  label_values     Array of label values that define the boundaries controlled by the `BCDefinition`, size `num_label_values`
   @param[out] bc_def           The new `BCDefinition`
**/
PetscErrorCode BCDefinitionCreate(const char *name, PetscInt num_label_values, PetscInt label_values[], BCDefinition *bc_def) {
  PetscFunctionBeginUser;
  PetscCall(PetscNew(bc_def));

  PetscCall(PetscStrallocpy(name, &(*bc_def)->name));
  (*bc_def)->num_label_values = num_label_values;
  PetscCall(PetscMalloc1(num_label_values, &(*bc_def)->label_values));
  for (PetscInt i = 0; i < num_label_values; i++) (*bc_def)->label_values[i] = label_values[i];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
   @brief Get base information for `BCDefinition`

   @param[in]  bc_def           `BCDefinition` to get information from
   @param[out] name             Name of the `BCDefinition`
   @param[out] num_label_values Number of `DMLabel` values
   @param[out] label_values     Array of label values that define the boundaries controlled by the `BCDefinition`, size `num_label_values`
**/
PetscErrorCode BCDefinitionGetInfo(BCDefinition bc_def, const char *name[], PetscInt *num_label_values, const PetscInt *label_values[]) {
  PetscFunctionBeginUser;
  if (name) {
    PetscAssertPointer(name, 2);
    *name = bc_def->name;
  }
  if (label_values) {
    PetscAssertPointer(num_label_values, 3);
    PetscAssertPointer(label_values, 4);
    *num_label_values = bc_def->num_label_values;
    *label_values     = bc_def->label_values;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
   @brief Destory a `BCDefinition` object

   @param[in,out] bc_def `BCDefinition` to be destroyed
**/
PetscErrorCode BCDefinitionDestroy(BCDefinition *bc_def) {
  BCDefinition bc_def_ = *bc_def;
  PetscFunctionBeginUser;
  if (bc_def_->name) PetscCall(PetscFree(bc_def_->name));
  if (bc_def_->label_values) PetscCall(PetscFree(bc_def_->label_values));
  if (bc_def_->essential_comps) PetscCall(PetscFree(bc_def_->essential_comps));
  if (bc_def_->dm) PetscCall(DMDestroy(&bc_def_->dm));
  if (bc_def_->DestroyCtx) PetscCall((*(bc_def_->DestroyCtx))(&bc_def_->ctx));
  PetscCall(PetscFree(bc_def_));
  *bc_def = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
   @brief Set `DM_BC_ESSENTIAL` boundary condition values

   @param[in,out] bc_def              `BCDefinition` to set values to
   @param[in]     num_essential_comps Number of components to set
   @param[in]     essential_comps     Array of components to set, size `num_essential_comps`
**/
PetscErrorCode BCDefinitionSetEssential(BCDefinition bc_def, PetscInt num_essential_comps, PetscInt essential_comps[]) {
  PetscFunctionBeginUser;
  bc_def->num_essential_comps = num_essential_comps;
  PetscCall(PetscMalloc1(num_essential_comps, &bc_def->essential_comps));
  PetscCall(PetscArraycpy(bc_def->essential_comps, essential_comps, num_essential_comps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
   @brief Get `DM_BC_ESSENTIAL` boundary condition values

   @param[in]  bc_def              `BCDefinition` to set values to
   @param[out] num_essential_comps Number of components to set
   @param[out] essential_comps     Array of components to set, size `num_essential_comps`
**/
PetscErrorCode BCDefinitionGetEssential(BCDefinition bc_def, PetscInt *num_essential_comps, const PetscInt *essential_comps[]) {
  PetscFunctionBeginUser;
  PetscAssertPointer(num_essential_comps, 2);
  PetscAssertPointer(essential_comps, 3);
  *num_essential_comps = bc_def->num_essential_comps;
  *essential_comps     = bc_def->essential_comps;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define LABEL_ARRAY_SIZE 256

// @brief See `PetscOptionsBCDefinition`
PetscErrorCode PetscOptionsBCDefinition_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[],
                                                const char name[], BCDefinition *bc_def, PetscBool *set) {
  PetscInt num_label_values = LABEL_ARRAY_SIZE, label_values[LABEL_ARRAY_SIZE] = {0};

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsIntArray(opt, text, man, label_values, &num_label_values, set));
  if (num_label_values > 0) {
    PetscCall(BCDefinitionCreate(name, num_label_values, label_values, bc_def));
  } else {
    *bc_def = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Set `DM` for BCDefinition

  @param[in,out] bc_def `BCDefinition` to add `dm` to
  @param[in]     dm     `DM` to assign to BCDefinition, or `NULL` to remove `DM`
**/
PetscErrorCode BCDefinitionSetDM(BCDefinition bc_def, DM dm) {
  PetscFunctionBeginUser;
  if (bc_def->dm) PetscCall(DMDestroy(&bc_def->dm));
  if (dm) {
    PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
    PetscCall(PetscObjectReference((PetscObject)dm));
    bc_def->dm = dm;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Get `DM` assigned to BCDefinition

  @param[in]  bc_def `BCDefinition` to get `dm` from
  @param[out] dm     `DM` assigned to BCDefinition
**/
PetscErrorCode BCDefinitionGetDM(BCDefinition bc_def, DM *dm) {
  PetscFunctionBeginUser;
  PetscAssertPointer(dm, 2);
  *dm = bc_def->dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Set custom context struct for use in BCDefinition

  @param[in,out] bc_def      `BCDefinition` to add `ctx` to
  @param[in]     destroy_ctx Optional function pointer that destroys the user context on `BCDefinitionDestroy()`
  @param[in]     ctx         Pointer to context struct
**/
PetscErrorCode BCDefinitionSetContext(BCDefinition bc_def, PetscCtxDestroyFn *destroy_ctx, void *ctx) {
  PetscFunctionBeginUser;
  if (bc_def->DestroyCtx) PetscCall((*(bc_def->DestroyCtx))(&bc_def->ctx));
  bc_def->ctx        = ctx;
  bc_def->DestroyCtx = destroy_ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Set custom context struct for use in BCDefinition

  @param[in]  bc_def `BCDefinition` to get `ctx` from
  @param[out] ctx    Pointer to context struct
**/
PetscErrorCode BCDefinitionGetContext(BCDefinition bc_def, void *ctx) {
  PetscFunctionBeginUser;
  PetscAssertPointer(ctx, 2);
  *(void **)ctx = bc_def->ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Add function pointers to create `CeedQFunction` and `CeedOperator` for IFunction of boundary condition

  @param[in,out] bc_def    `BCDefinition` to add function pointers to
  @param[in]     create_qf Function to create `CeedQFunction`
  @param[in]     add_op    Function to create and add `CeedOperator` to composite `CeedOperator`
**/
PetscErrorCode BCDefinitionSetIFunction(BCDefinition bc_def, BCDefinitionCreateQFunction create_qf, BCDefinitionAddIFunctionOperator add_op) {
  PetscFunctionBeginUser;
  bc_def->CreateIFunctionQF    = create_qf;
  bc_def->AddIFunctionOperator = add_op;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Add function pointers to create `CeedQFunction` and `CeedOperator` for IJacobian of boundary condition

  @param[in,out] bc_def    `BCDefinition` to add function pointers to
  @param[in]     create_qf Function to create `CeedQFunction`
  @param[in]     add_op    Function to create and add `CeedOperator` to composite `CeedOperator`
**/
PetscErrorCode BCDefinitionSetIJacobian(BCDefinition bc_def, BCDefinitionCreateQFunction create_qf, BCDefinitionAddIJacobianOperator add_op) {
  PetscFunctionBeginUser;
  bc_def->CreateIJacobianQF    = create_qf;
  bc_def->AddIJacobianOperator = add_op;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Add operators (IFunction, IJacobian) to composite operator

  This loops over orientations for each face label specified by `bc_def` and adds the IFunction and IJacobian operator to respective composite operator.

  @param[in]     bc_def   `BCDefinition` from which operators are created and added
  @param[in,out] op_ifunc Composite operator for IFunction operators to be added to
  @param[in,out] op_ijac  Composite operator for IJacobian operators to be added to
**/
PetscErrorCode BCDefinitionAddOperators(BCDefinition bc_def, CeedOperator op_ifunc, CeedOperator op_ijac) {
  Ceed            ceed = CeedOperatorReturnCeed(op_ifunc);
  CeedQFunction   qf_ifunction, qf_ijacobian;
  DMLabel         face_sets_label;
  PetscInt        num_face_set_values;
  const PetscInt *face_set_values;

  PetscFunctionBeginUser;
  if (!bc_def->CreateIFunctionQF || !bc_def->AddIFunctionOperator) PetscFunctionReturn(PETSC_SUCCESS);
  PetscBool add_ijac = (!bc_def->CreateIJacobianQF || !bc_def->AddIJacobianOperator || !op_ijac) ? PETSC_FALSE : PETSC_TRUE;
  PetscCheck(bc_def->dm, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "BCDefinition must have DM set using BCDefinitionSetDM()");

  PetscCall(bc_def->CreateIFunctionQF(bc_def, &qf_ifunction));
  if (add_ijac) PetscCall(bc_def->CreateIJacobianQF(bc_def, &qf_ijacobian));

  PetscCall(DMGetLabel(bc_def->dm, "Face Sets", &face_sets_label));
  PetscCall(BCDefinitionGetInfo(bc_def, NULL, &num_face_set_values, &face_set_values));
  for (PetscInt f = 0; f < num_face_set_values; f++) {
    DMLabel  face_orientation_label;
    PetscInt num_orientations_values, *orientation_values;

    {
      char *face_orientation_label_name;

      PetscCall(DMPlexCreateFaceLabel(bc_def->dm, face_set_values[f], &face_orientation_label_name));
      PetscCall(DMGetLabel(bc_def->dm, face_orientation_label_name, &face_orientation_label));
      PetscCall(PetscFree(face_orientation_label_name));
    }
    PetscCall(DMLabelCreateGlobalValueArray(bc_def->dm, face_orientation_label, &num_orientations_values, &orientation_values));
    for (PetscInt o = 0; o < num_orientations_values; o++) {
      CeedOperator sub_op_ifunc;
      PetscInt     orientation = orientation_values[o];

      PetscCall(bc_def->AddIFunctionOperator(bc_def, face_orientation_label, orientation, qf_ifunction, op_ifunc, &sub_op_ifunc));
      if (add_ijac) PetscCall(bc_def->AddIJacobianOperator(bc_def, sub_op_ifunc, face_orientation_label, orientation, qf_ijacobian, op_ijac));
      PetscCallCeed(ceed, CeedOperatorDestroy(&sub_op_ifunc));
    }
    PetscCall(PetscFree(orientation_values));
  }
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_ifunction));
  if (add_ijac) PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_ijacobian));
  PetscFunctionReturn(PETSC_SUCCESS);
  PetscFunctionReturn(PETSC_SUCCESS);
}
