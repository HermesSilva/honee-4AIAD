// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#pragma once

#include <ceed.h>
#include <petsc/private/petscimpl.h>
#include <petscdm.h>

typedef struct _p_BCDefinition *BCDefinition;

typedef PetscErrorCode (*BCDefinitionCreateQFunction)(BCDefinition bc_def, CeedQFunction *qf);
typedef PetscErrorCode (*BCDefinitionAddIFunctionOperator)(BCDefinition bc_def, DMLabel orientation_label, PetscInt orientation, CeedQFunction qf,
                                                           CeedOperator op, CeedOperator *sub_op);
typedef PetscErrorCode (*BCDefinitionAddIJacobianOperator)(BCDefinition bc_def, CeedOperator sub_op_ifunction, DMLabel orientation_label,
                                                           PetscInt orientation, CeedQFunction qf, CeedOperator op);

typedef struct _BCDefinitionOps *BCDefinitionOps;
struct _BCDefinitionOps {};

struct _p_BCDefinition {
  PETSCHEADER(struct _BCDefinitionOps);
  char *name;
  DM    dm;

  // Boundary ID information
  PetscInt num_label_values, *label_values, dm_field;

  // Essential Boundary information
  PetscInt num_essential_comps, *essential_comps;

  void                            *ctx;
  PetscCtxDestroyFn               *DestroyCtx;
  BCDefinitionCreateQFunction      CreateIFunctionQF, CreateIJacobianQF;
  BCDefinitionAddIFunctionOperator AddIFunctionOperator;
  BCDefinitionAddIJacobianOperator AddIJacobianOperator;

  char *natural_bc_type;
};

/**
   @brief Creates a `BCDefinition` from an array of integers in an option in the database

   Must be between `PetscOptionsBegin()` and `PetscOptionsEnd()`.

   @param[in]  opt    The option one is seeking
   @param[in]  text   Short string describing option
   @param[in]  man    Manual page for the option
   @param[in]  name   String that sets the name of the `BCDefinition`
   @param[out] bc_def Resulting `BCDefinition`, `NULL` if option is not set
   @param[out] set    `PETSC_TRUE` if found, else `PETSC_FALSE`
**/
#define PetscOptionsBCDefinition(opt, text, man, name, bc_def, set) \
  PetscOptionsBCDefinition_Private(PetscOptionsObject, opt, text, man, name, bc_def, set)
PetscErrorCode PetscOptionsBCDefinition_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[],
                                                const char name[], BCDefinition *bc_def, PetscBool *set);

PetscErrorCode BCDefinitionCreate(MPI_Comm comm, const char *name, PetscInt num_label_values, PetscInt label_values[], BCDefinition *bc_def);
PetscErrorCode BCDefinitionGetInfo(BCDefinition bc_def, const char *name[], PetscInt *num_label_values, const PetscInt *label_values[]);
PetscErrorCode BCDefinitionView(BCDefinition bc_def, PetscViewer viewer);
PetscErrorCode BCDefinitionViewFromOptions(BCDefinition bc_def, PetscObject obj, const char name[]);
PetscErrorCode BCDefinitionDestroy(BCDefinition *bc_def);

PetscErrorCode BCDefinitionSetEssential(BCDefinition bc_def, PetscInt num_essential_comps, PetscInt essential_comps[]);
PetscErrorCode BCDefinitionGetEssential(BCDefinition bc_def, PetscInt *num_essential_comps, const PetscInt *essential_comps[]);

PetscErrorCode BCDefinitionSetDM(BCDefinition bc_def, DM dm);
PetscErrorCode BCDefinitionGetDM(BCDefinition bc_def, DM *dm);

PetscErrorCode BCDefinitionSetContext(BCDefinition bc_def, void *ctx, PetscCtxDestroyFn *destroy_ctx);
PetscErrorCode BCDefinitionGetContext(BCDefinition bc_def, void *ctx);

PetscErrorCode BCDefinitionSetIFunction(BCDefinition bc_def, BCDefinitionCreateQFunction create_qf, BCDefinitionAddIFunctionOperator add_op);
PetscErrorCode BCDefinitionSetIJacobian(BCDefinition bc_def, BCDefinitionCreateQFunction create_qf, BCDefinitionAddIJacobianOperator add_op);
PetscErrorCode BCDefinitionAddOperators(BCDefinition bc_def, CeedOperator op_ifunction, CeedOperator op_ijacobian);

typedef PetscErrorCode (*BCDefinitionSetupFn)(BCDefinition bc_def, void *ctx);
PetscErrorCode BCDefinitionRegisterNaturalSetup(const char name[], BCDefinitionSetupFn setup_fn);
PetscErrorCode BCDefinitionSetNaturalType(BCDefinition bc_def, const char natural_type[]);
PetscErrorCode BCDefinitionGetNaturalType(BCDefinition bc_def, const char *natural_type[]);
PetscErrorCode BCDefinitionSetup(BCDefinition bc_def);
