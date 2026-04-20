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
  @brief Get string version of libCEED version information

  @param[in,out] version Static `char` array to store string
  @param[in]     len     Length of `version` array
**/
PetscErrorCode HoneeGetCeedVersionString(char version[], size_t len) {
  int         major, minor, patch;
  PetscBool   release;
  const char *git_version;

  PetscFunctionBeginUser;
  PetscCall(CeedGetVersion(&major, &minor, &patch, &release));
  PetscCall(CeedGetGitVersion(&git_version));
  PetscCall(PetscSNPrintf(version, len, "libCEED version: %d.%d.%d (%s)", major, minor, patch, release ? "release" : git_version));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Get string version of HONEE version information

  @param[in,out] version Static `char` array to store string
  @param[in]     len     Length of `version` array
**/
PetscErrorCode HoneeGetVersionString(char version[], size_t len) {
  int         major, minor, patch;
  PetscBool   release;
  const char *git_version;

  PetscFunctionBeginUser;
  PetscCall(HoneeGetVersion(&major, &minor, &patch, &release));
  PetscCall(HoneeGetGitVersion(&git_version));
  PetscCall(PetscSNPrintf(version, len, "HONEE version: %d.%d.%d (%s)", major, minor, patch, release ? "release" : git_version));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HoneeVersionFunction(MPI_Comm comm) {
  char        version[256];
  PetscViewer viewer;

  PetscFunctionBeginUser;
  PetscCall(PetscViewerASCIIGetStdout(comm, &viewer));
  PetscCall(HoneeGetVersionString(version, PETSC_STATIC_ARRAY_LENGTH(version)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%s\n", version));
  PetscCall(HoneeGetCeedVersionString(version, PETSC_STATIC_ARRAY_LENGTH(version)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%s\n", version));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Setup functions before running `PetscInitialize()`
PetscErrorCode HoneePrePetscInitialize() {
  PetscFunctionBeginUser;
  PetscCall(PetscSetHelpVersionFunctions(NULL, HoneeVersionFunction));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Initialize `Honee` object

  @param[in]  comm  `MPI_Comm` for the object
  @param[out] honee The initialized `Honee` object
**/
PetscErrorCode HoneeInit(MPI_Comm comm, Honee *honee) {
  Honee honee_;

  PetscFunctionBeginUser;
  PetscCall(HoneeInitClass_Private());
  PetscCall(PetscHeaderCreate(honee_, HONEE_CLASSID, "Honee", "HONEE", "Honee", comm, HoneeDestroy, NULL));
  PetscCall(PetscCalloc4(1, &honee_->app_ctx, 1, &honee_->problem_data, 1, &honee_->phys, 1, &honee_->units));
  honee_->problem_data->set_bc_from_ics = PETSC_TRUE;
  honee_->comm                          = PETSC_COMM_WORLD;
  honee_->start_time                    = time(NULL);
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

  PetscCall(QDataClearStoredData());
  PetscCall(DivDiffFluxProjectionDataDestroy(honee_->diff_flux_proj));

  // -- Vectors
  PetscCallCeed(ceed, CeedVectorDestroy(&honee_->q_ceed));
  PetscCallCeed(ceed, CeedVectorDestroy(&honee_->q_dot_ceed));
  PetscCallCeed(ceed, CeedVectorDestroy(&honee_->g_ceed));

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

  // -- Structs
  for (PetscInt i = 0; i < problem->num_bc_defs; i++) {
    PetscCall(BCDefinitionDestroy(&problem->bc_defs[i]));
  }
  PetscCall(PetscFree(problem->bc_defs));
  for (PetscInt i = 0; i < problem->num_components; i++) {
    PetscCall(PetscFree(problem->component_names[i]));
  }
  PetscCall(PetscFree(problem->component_names));

  PetscCall(PetscFree4(honee_->app_ctx, honee_->problem_data, honee_->phys, honee_->units));
  PetscCall(PetscHeaderDestroy(&honee_));
  *honee = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Saves a pointer-to-data in a `Honee` object by key

  This also checks whether `key` has been used previously and will error out if it has.

  @param[in] honee             `Honee` object to save pointer to
  @param[in] key               String used to identify the pointer
  @param[in] container         Pointer to the data
  @param[in] container_destroy Function to destroy the struct after it is used
**/
PetscErrorCode HoneeSetContainer(Honee honee, const char key[], void *container, PetscCtxDestroyFn *container_destroy) {
  PetscFunctionBeginUser;
  {  // Verify that key is not already taken
    void *test_data;
    PetscCall(PetscObjectContainerQuery((PetscObject)honee, key, &test_data));
    PetscCheck(test_data == NULL, PetscObjectComm((PetscObject)honee), PETSC_ERR_SUP, "Cannot set container with key '%s'; key is already in use.",
               key);
  }
  PetscCall(PetscObjectContainerCompose((PetscObject)honee, key, container, container_destroy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Retrieve a pointer-to-data in a `Honee` object by key

  This will error out if `honee` does not have a container identified by `key`.

  @param[in]  honee     `Honee` object to retrieve pointer from
  @param[in]  key       String used to identify the pointer
  @param[out] container Pointer to the data
**/
PetscErrorCode HoneeGetContainer(Honee honee, const char key[], void *container) {
  PetscFunctionBeginUser;
  PetscCall(PetscObjectContainerQuery((PetscObject)honee, key, container));
  PetscCheck(*(void **)container != NULL, PetscObjectComm((PetscObject)honee), PETSC_ERR_SUP, "Container with key '%s' not found.", key);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Check if `Honee` object as pointer-to-data identified by key

  @param[in]  honee   `Honee` object to query for key
  @param[in]  key     String to search for
  @param[out] has_key Whether key has been set
**/
PetscErrorCode HoneeHasContainer(Honee honee, const char key[], PetscBool *has_key) {
  void *test_data;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectContainerQuery((PetscObject)honee, key, &test_data));
  *has_key = test_data == NULL ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

const DMLabel  DMLABEL_DEFAULT       = NULL;
const PetscInt DMLABEL_DEFAULT_VALUE = 0;
