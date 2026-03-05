// SPDX-FileCopyrightText: Copyright (c) 2017-2025, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include "../qfunctions/honee-wall-distance.h"
#include <honee.h>
#include <navierstokes.h>

#define WALLDIST_PREFIX "walldistance_"
#define WALLDIST_KEY "Wall Distance Context"

typedef struct HoneeWallDistance_ *HoneeWallDistance;
struct HoneeWallDistance_ {
  IS                   is_face_label_values;
  Vec                  Phi, Walldist;
  DM                   dm_laplacian;
  PetscReal            p_values[6], p_value_final;
  PetscInt             num_p_values;
  CeedQFunctionContext laplacian_qfctx;
};

static PetscErrorCode HoneeWallDistanceDestroy(HoneeWallDistance *walldist_ctx) {
  HoneeWallDistance ctx_ = *walldist_ctx;

  PetscFunctionBeginUser;
  PetscCall(ISDestroy(&ctx_->is_face_label_values));
  PetscCall(VecDestroy(&ctx_->Phi));
  PetscCall(VecDestroy(&ctx_->Walldist));
  PetscCall(DMDestroy(&ctx_->dm_laplacian));
  if (ctx_->laplacian_qfctx) {
    Ceed ceed = CeedObjectReturnCeed((CeedObject)ctx_->laplacian_qfctx);
    PetscCallCeed(ceed, CeedQFunctionContextDestroy(&ctx_->laplacian_qfctx));
  }
  PetscCall(PetscFree(ctx_));
  *walldist_ctx = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Setup wall distance calculation struct in HONEE

  @param[in] honee            `Honee`
  @param[in] num_label_values Number of label values in `label_values`
  @param[in] label_values     Label values in "Face Sets" that will be used to define the wall
**/
PetscErrorCode HoneeWallDistanceSetup(Honee honee, PetscInt num_label_values, const PetscInt *label_values) {
  HoneeWallDistance walldist_ctx;
  PetscBool         already_has_walldist, p_values_given;

  PetscFunctionBeginUser;
  PetscCall(HoneeHasContainer(honee, WALLDIST_KEY, &already_has_walldist));
  PetscCheck(!already_has_walldist, PetscObjectComm((PetscObject)honee), PETSC_ERR_ARG_WRONGSTATE, "Wall distance context is already setup.");
  PetscCall(PetscNew(&walldist_ctx));
  PetscOptionsBegin(PetscObjectComm((PetscObject)honee), NULL, "HONEE Wall Distance Calculation", NULL);
  walldist_ctx->num_p_values = PETSC_STATIC_ARRAY_LENGTH(walldist_ctx->p_values);
  PetscCall(PetscOptionsRealArray("-" WALLDIST_PREFIX "p_values",
                                  "Array of values to use in computing p-Laplacian (result of former used an initial guess of latter)", NULL,
                                  walldist_ctx->p_values, &walldist_ctx->num_p_values, &p_values_given));
  if (!p_values_given) {
    walldist_ctx->p_values[0]  = 2;
    walldist_ctx->num_p_values = 1;
  } else {
    for (PetscInt p = 0; p < walldist_ctx->num_p_values; p++) {
      PetscCheck(walldist_ctx->p_values[p] >= 2, PetscObjectComm((PetscObject)honee), PETSC_ERR_USER_INPUT,
                 "p-Laplacian value should be greater or equal to two, recieved %g", walldist_ctx->p_values[p]);
      if (p != 0)
        PetscCheck(walldist_ctx->p_values[p] > walldist_ctx->p_values[p - 1], PetscObjectComm((PetscObject)honee), PETSC_ERR_USER_INPUT,
                   "Sequence of p-Laplacian values should be strictly increasing, recieved %g followed by %g", walldist_ctx->p_values[p - 1],
                   walldist_ctx->p_values[p]);
    }
  }
  PetscOptionsEnd();
  PetscCall(HoneeSetContainer(honee, WALLDIST_KEY, walldist_ctx, (PetscCtxDestroyFn *)HoneeWallDistanceDestroy));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, num_label_values, label_values, PETSC_COPY_VALUES, &walldist_ctx->is_face_label_values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct HoneePLaplacian_ *HoneePLaplacian;
struct HoneePLaplacian_ {
  OperatorApplyContext laplacian_residual;
};

static PetscErrorCode HoneePLaplacianContextDestroy(HoneePLaplacian *pLaplacian_ctx) {
  HoneePLaplacian ctx_ = *pLaplacian_ctx;

  PetscFunctionBeginUser;
  PetscCall(OperatorApplyContextDestroy(ctx_->laplacian_residual));
  PetscCall(PetscFree(ctx_));
  *pLaplacian_ctx = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief SNES Function to solve the p-Laplacian
static PetscErrorCode HoneeWallDistance_SNESFunction(SNES snes, Vec x, Vec f, void *ctx) {
  HoneePLaplacian context = (HoneePLaplacian)ctx;

  PetscFunctionBeginUser;
  PetscCall(ApplyCeedOperatorGlobalToGlobal(x, f, context->laplacian_residual));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Computing the solution to the p-Laplacian
static PetscErrorCode HoneeWallDistance_ComputePLaplacian(Honee honee) {
  Ceed              ceed = honee->ceed;
  DM                dm;
  PetscInt          num_comps    = 1, dim;
  PetscBool         setup_coords = PETSC_TRUE;  // Use coordinate space already determined by the solution DM
  HoneeWallDistance walldist_ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(HONEE_WallDistPLaplacian, honee->dm, NULL, NULL, NULL));
  PetscCall(HoneeGetContainer(honee, WALLDIST_KEY, &walldist_ctx));
  PetscCall(DMGetDimension(honee->dm, &dim));
  {  // Setup DM for p-Laplacian
    DMLabel         label;
    PetscInt        essential_comps = 0, num_label_values;
    const PetscInt *label_values;
    PetscSection    section;

    PetscCall(DMClone(honee->dm, &dm));
    PetscCall(DMSetMatrixPreallocateSkip(dm, PETSC_FALSE));  // Need this to always be set for MFFD Jacobian
    PetscCall(DMSetupByOrderBegin_FEM(PETSC_TRUE, setup_coords, honee->app_ctx->degree, PETSC_DECIDE, honee->app_ctx->q_extra, 1, &num_comps, dm));

    // Add strong boundary conditions to DM
    PetscCall(DMGetLabel(dm, "Face Sets", &label));
    PetscCheck(label, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "DM does not have 'Face Sets' label");
    PetscCall(DMPlexLabelComplete(dm, label));
    PetscCall(ISGetSize(walldist_ctx->is_face_label_values, &num_label_values));
    PetscCall(ISGetIndices(walldist_ctx->is_face_label_values, &label_values));
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "Wall", label, num_label_values, label_values, 0, 1, &essential_comps, NULL, NULL, NULL, NULL));
    PetscCall(ISRestoreIndices(walldist_ctx->is_face_label_values, &label_values));

    PetscCall(DMSetupByOrderEnd_FEM(setup_coords, dm));
    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(PetscSectionSetFieldName(section, 0, ""));
    PetscCall(PetscSectionSetComponentName(section, 0, 0, "WallDistanceLaplacian"));
  }

  CeedContextFieldLabel p_label;
  HoneePLaplacian       plaplacian_ctx;
  PetscCall(PetscNew(&plaplacian_ctx));
  {
    PetscInt             height = 0, dm_field = 0;
    CeedQFunction        qf_laplacian;
    CeedOperator         op_laplacian;
    CeedQFunctionContext laplacian_qfctx;
    CeedBasis            basis_laplacian;
    CeedElemRestriction  elem_restr_laplacian, elem_restr_qd;
    CeedVector           q_data;
    CeedInt              q_data_size;

    PetscCall(QDataGet(ceed, dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, &elem_restr_qd, &q_data, &q_data_size));
    PetscCall(DMPlexCeedElemRestrictionCreate(ceed, dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, height, dm_field, &elem_restr_laplacian));
    PetscCall(DMPlexCeedBasisCreate(ceed, dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, height, dm_field, &basis_laplacian));

    {
      PLaplacianContext laplacian_ctx;
      PetscCall(PetscNew(&laplacian_ctx));
      PetscCallCeed(ceed, CeedQFunctionContextCreate(ceed, &laplacian_qfctx));
      PetscCallCeed(ceed, CeedQFunctionContextSetData(laplacian_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*laplacian_ctx), laplacian_ctx));
      PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(laplacian_qfctx, CEED_MEM_HOST, FreeContextPetsc));
      PetscCallCeed(ceed,
                    CeedQFunctionContextRegisterDouble(laplacian_qfctx, "p", offsetof(struct PLaplacianContext_, p), 1, "p value for p-Laplacian"));
      PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(laplacian_qfctx, &walldist_ctx->laplacian_qfctx));
    }

    switch (dim) {
      case 2:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, PLaplacian_2D, PLaplacian_2D_loc, &qf_laplacian));
        break;
      case 3:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, PLaplacian_3D, PLaplacian_3D_loc, &qf_laplacian));
        break;
      default:
        SETERRQ(honee->comm, PETSC_ERR_ARG_WRONG, "Dimension %" PetscInt_FMT "not supported for p-Laplacian", dim);
    }
    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_laplacian, laplacian_qfctx));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_laplacian, "Grad phi", dim, CEED_EVAL_GRAD));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_laplacian, "qdata", q_data_size, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_laplacian, "v", 1, CEED_EVAL_INTERP));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_laplacian, "Grad v", dim, CEED_EVAL_GRAD));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_laplacian, NULL, NULL, &op_laplacian));
    PetscCallCeed(ceed, CeedOperatorSetField(op_laplacian, "Grad phi", elem_restr_laplacian, basis_laplacian, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_laplacian, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_laplacian, "v", elem_restr_laplacian, basis_laplacian, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_laplacian, "Grad v", elem_restr_laplacian, basis_laplacian, CEED_VECTOR_ACTIVE));

    PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(op_laplacian, "p", &p_label));

    Vec Phi_loc;
    PetscCall(DMCreateLocalVector(dm, &Phi_loc));
    PetscCall(VecZeroEntries(Phi_loc));
    PetscCall(OperatorApplyContextCreate(dm, dm, ceed, op_laplacian, NULL, NULL, Phi_loc, NULL, &plaplacian_ctx->laplacian_residual));
    PetscCall(VecDestroy(&Phi_loc));

    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_laplacian));
    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
    PetscCallCeed(ceed, CeedBasisDestroy(&basis_laplacian));
    PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
    PetscCallCeed(ceed, CeedQFunctionContextDestroy(&laplacian_qfctx));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_laplacian));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_laplacian));
  }

  SNES     snes;
  MPI_Comm comm = honee->comm;
  Vec      Phi;
  PetscCall(DMCreateGlobalVector(dm, &Phi));
  PetscCall(VecZeroEntries(Phi));

  PetscCall(SNESCreate(comm, &snes));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(SNESSetApplicationContext(snes, plaplacian_ctx));
  PetscCall(SNESSetType(snes, SNESNEWTONLS));
  PetscCall(SNESSetOptionsPrefix(snes, WALLDIST_PREFIX));
  PetscCall(SNESSetFunction(snes, NULL, HoneeWallDistance_SNESFunction, plaplacian_ctx));
  PetscCall(SNESSetFromOptions(snes));

  // NOTE: This would probably be more efficient if we also did an L^2 projection of the wall distance between each p-Laplacian solve.
  // For large values of p, the wall distance is probably a better initial guess than the solution to a lower valued p-Laplacian.
  for (PetscInt p = 0; p < walldist_ctx->num_p_values; p++) {
    CeedScalar new_p_value = walldist_ctx->p_values[p];
    PetscCallCeed(ceed, CeedOperatorSetContextDouble(plaplacian_ctx->laplacian_residual->op, p_label, &new_p_value));
    PetscCall(SNESSolve(snes, NULL, Phi));
  }
  PetscCall(VecViewFromOptions(Phi, NULL, "-" WALLDIST_PREFIX "laplacian_view"));

  walldist_ctx->dm_laplacian = dm;
  walldist_ctx->Phi          = Phi;
  {
    size_t        num_values;
    const double *values;
    PetscCallCeed(ceed, CeedOperatorGetContextDoubleRead(plaplacian_ctx->laplacian_residual->op, p_label, &num_values, &values));
    PetscCheck(num_values == 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "Recieved more than 1 value for the p-Laplacian, which shouldn't happen");
    walldist_ctx->p_value_final = *values;
    PetscCallCeed(ceed, CeedOperatorRestoreContextDoubleRead(plaplacian_ctx->laplacian_residual->op, p_label, &values));
  }

  PetscCall(SNESDestroy(&snes));
  PetscCall(HoneePLaplacianContextDestroy(&plaplacian_ctx));
  PetscCall(PetscLogEventEnd(HONEE_WallDistPLaplacian, honee->dm, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Get wall distance at quadrature points

  This returns a `CeedVector` of the wall distance field evaluated at quadrature points.
  To pass this into a `CeedQFunction`, use `elem_restr_walldist` and `CEED_BASIS_NONE`.

  @param[in]  honee               `Honee`
  @param[in]  domain_label        `DMLabel` to get quadrature points at
  @param[in]  label_value         Label values in `domain_label`
  @param[in]  height              Topological element height to get quadrature points for
  @param[out] elem_restr_walldist Element restriction for the wall distance
  @param[out] vec_walldist        Vector of wall distance evaluated at quadrature points
**/
PetscErrorCode HoneeWallDistanceGetQuadratureValues(Honee honee, DMLabel domain_label, PetscInt label_value, PetscInt height,
                                                    CeedElemRestriction *elem_restr_walldist, CeedVector *vec_walldist) {
  Ceed                 ceed = honee->ceed;
  PetscInt             dim;
  HoneeWallDistance    walldist_ctx;
  PetscInt             dm_field = 0;
  CeedElemRestriction  elem_restr_qd;
  CeedVector           q_data;
  CeedInt              q_data_size;
  CeedQFunction        qf_walldist;
  CeedOperator         op_walldist;
  OperatorApplyContext calc_walldist;
  CeedElemRestriction  elem_restr_laplacian;
  CeedBasis            basis_laplacian;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(HONEE_WallDistQuadraturePoints, honee->dm, NULL, NULL, NULL));
  PetscCall(HoneeGetContainer(honee, WALLDIST_KEY, &walldist_ctx));
  if (walldist_ctx->Phi == NULL) PetscCall(HoneeWallDistance_ComputePLaplacian(honee));
  DM dm = walldist_ctx->dm_laplacian;

  PetscCall(QDataGet(ceed, dm, domain_label, label_value, &elem_restr_qd, &q_data, &q_data_size));
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, dm, domain_label, label_value, height, dm_field, &elem_restr_laplacian));
  PetscCall(DMPlexCeedBasisCreate(ceed, dm, domain_label, label_value, height, dm_field, &basis_laplacian));
  PetscCall(DMPlexCeedElemRestrictionQDataCreate(ceed, honee->dm, domain_label, label_value, height, 1, elem_restr_walldist));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(*elem_restr_walldist, vec_walldist, NULL));

  PetscCall(DMGetDimension(walldist_ctx->dm_laplacian, &dim));
  switch (dim) {
    case 2:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, WallDistanceFromPLaplacian_2D, WallDistanceFromPLaplacian_3D_loc, &qf_walldist));
      break;
    case 3:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, WallDistanceFromPLaplacian_3D, WallDistanceFromPLaplacian_3D_loc, &qf_walldist));
      break;
    default:
      SETERRQ(honee->comm, PETSC_ERR_ARG_WRONG, "Dimension %" PetscInt_FMT "not supported for p-Laplacian", dim);
  }
  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_walldist, walldist_ctx->laplacian_qfctx));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_walldist, "phi", 1, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_walldist, "Grad phi", dim, CEED_EVAL_GRAD));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_walldist, "qdata", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_walldist, "walldist", 1, CEED_EVAL_NONE));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_walldist, NULL, NULL, &op_walldist));
  PetscCallCeed(ceed, CeedOperatorSetField(op_walldist, "phi", elem_restr_laplacian, basis_laplacian, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_walldist, "Grad phi", elem_restr_laplacian, basis_laplacian, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_walldist, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_walldist, "walldist", *elem_restr_walldist, CEED_BASIS_NONE, *vec_walldist));

  Vec Phi_loc;
  PetscCall(DMCreateLocalVector(dm, &Phi_loc));
  PetscCall(VecZeroEntries(Phi_loc));
  PetscCall(OperatorApplyContextCreate(dm, NULL, ceed, op_walldist, NULL, CEED_VECTOR_NONE, Phi_loc, NULL, &calc_walldist));
  PetscCall(VecDestroy(&Phi_loc));

  PetscCall(ApplyCeedOperatorGlobalToLocal(walldist_ctx->Phi, NULL, calc_walldist));

  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_laplacian));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_laplacian));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_walldist));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_walldist));
  PetscCallCeed(ceed, OperatorApplyContextDestroy(calc_walldist));

  PetscCall(PetscLogEventEnd(HONEE_WallDistQuadraturePoints, honee->dm, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Projects the wall distance field onto the finite element grid

  @param[in]  honee               `Honee`
  @param[out] elem_restr_walldist Element restriction for the wall distance
  @param[out] basis_walldist      Basis for projected wall distance field
  @param[out] vec_walldist        Vector of wall distance coefficients
**/
static PetscErrorCode HoneeWallDistanceGetProjectedField(Honee honee, CeedElemRestriction *elem_restr_walldist, CeedBasis *basis_walldist,
                                                         CeedVector *vec_walldist) {
  MPI_Comm             comm = honee->comm;
  Ceed                 ceed = honee->ceed;
  HoneeWallDistance    walldist_ctx;
  NodalProjectionData  projection;
  PetscInt             dim;
  CeedElemRestriction  elem_restr_qd;
  CeedVector           q_data;
  CeedInt              q_data_size;
  CeedElemRestriction  elem_restr_laplacian;
  CeedBasis            basis_laplacian;
  CeedQFunctionContext laplacian_qfctx;
  const DMLabel        domain_label = DMLABEL_DEFAULT;
  const PetscInt       label_value = DMLABEL_DEFAULT_VALUE, dm_field = 0, height = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(HONEE_WallDistL2Projection, honee->dm, NULL, NULL, NULL));
  PetscCall(HoneeGetContainer(honee, WALLDIST_KEY, &walldist_ctx));
  if (walldist_ctx->Phi == NULL) PetscCall(HoneeWallDistance_ComputePLaplacian(honee));
  DM dm = walldist_ctx->dm_laplacian;
  {  // Reset component name since we're reusing the p-Laplacian DM
    PetscSection section;

    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(PetscSectionSetComponentName(section, 0, 0, "WallDistance"));
  }
  PetscCall(DMGetDimension(dm, &dim));

  PetscCall(QDataGet(ceed, dm, domain_label, label_value, &elem_restr_qd, &q_data, &q_data_size));
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, dm, domain_label, label_value, height, dm_field, &elem_restr_laplacian));
  PetscCall(DMPlexCeedBasisCreate(ceed, dm, domain_label, label_value, height, dm_field, &basis_laplacian));
  PetscCall(PetscNew(&projection));
  projection->num_comp = 1;
  {  // -- Build Mass operator
    CeedQFunction qf_mass;
    CeedOperator  op_mass;

    PetscCall(HoneeMassQFunctionCreate(ceed, 1, q_data_size, &qf_mass));
    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass));
    PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "u", elem_restr_laplacian, basis_laplacian, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "v", elem_restr_laplacian, basis_laplacian,
                                             CEED_VECTOR_ACTIVE));  // Use the same FE space for the projected wall distance field

    {  // -- Setup KSP for L^2 projection
      Mat mat_mass;
      PetscCall(MatCreateCeed(dm, dm, op_mass, NULL, &mat_mass));
      PetscCall(KSPCreate(comm, &projection->ksp));
      PetscCall(KSPSetOptionsPrefix(projection->ksp, WALLDIST_PREFIX "projection_"));
      PetscCall(KSPSetFromOptions_WithMatCeed(projection->ksp, mat_mass));
      PetscCall(MatDestroy(&mat_mass));
    }

    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_mass));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_mass));
  }

  {  // Create RHS function for projection
    CeedQFunction        qf_rhs;
    CeedOperator         op_rhs;
    OperatorApplyContext calc_rhs;
    CeedElemRestriction  elem_restr_walldistq;
    CeedVector           vec_walldistq;

    PetscCall(HoneeWallDistanceGetQuadratureValues(honee, domain_label, label_value, height, &elem_restr_walldistq, &vec_walldistq));
    switch (dim) {
      case 2:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, WallDistance_ProjectionRHS_2D, WallDistance_ProjectionRHS_2D_loc, &qf_rhs));
        break;
      case 3:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, WallDistance_ProjectionRHS_3D, WallDistance_ProjectionRHS_3D_loc, &qf_rhs));
        break;
      default:
        SETERRQ(honee->comm, PETSC_ERR_ARG_WRONG, "Dimension %" PetscInt_FMT "not supported for p-Laplacian", dim);
    }
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs, "walldist", 1, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs, "qdata", q_data_size, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_rhs, "v", 1, CEED_EVAL_INTERP));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_rhs, NULL, NULL, &op_rhs));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs, "walldist", elem_restr_walldistq, CEED_BASIS_NONE, vec_walldistq));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs, "v", elem_restr_laplacian, basis_laplacian, CEED_VECTOR_ACTIVE));

    PetscCall(OperatorApplyContextCreate(dm, dm, ceed, op_rhs, CEED_VECTOR_NONE, NULL, NULL, NULL, &calc_rhs));
    projection->l2_rhs_ctx = calc_rhs;

    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_walldistq));
    PetscCallCeed(ceed, CeedVectorDestroy(&vec_walldistq));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_rhs));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_rhs));
  }

  PetscCall(DMCreateGlobalVector(dm, &walldist_ctx->Walldist));
  PetscCall(VecZeroEntries(walldist_ctx->Walldist));
  PetscCall(ApplyCeedOperatorLocalToGlobal(NULL, walldist_ctx->Walldist, projection->l2_rhs_ctx));
  PetscCall(KSPSolve(projection->ksp, walldist_ctx->Walldist, walldist_ctx->Walldist));

  if (elem_restr_walldist) {
    *elem_restr_walldist = NULL;
    PetscCallCeed(ceed, CeedElemRestrictionReferenceCopy(elem_restr_laplacian, elem_restr_walldist));
  }
  if (basis_walldist) {
    *basis_walldist = NULL;
    PetscCallCeed(ceed, CeedBasisReferenceCopy(basis_laplacian, basis_walldist));
  }
  if (vec_walldist) {
    Vec Walldist_loc;

    PetscCall(DMCreateLocalVector(dm, &Walldist_loc));
    PetscCall(DMGlobalToLocal(dm, walldist_ctx->Walldist, INSERT_VALUES, Walldist_loc));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_laplacian, vec_walldist, NULL));
    PetscCall(VecCopyPetscToCeed(Walldist_loc, *vec_walldist));
    PetscCall(VecDestroy(&Walldist_loc));
  }

  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&laplacian_qfctx));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_laplacian));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_laplacian));
  PetscCall(NodalProjectionDataDestroy(&projection));
  PetscCall(PetscLogEventEnd(HONEE_WallDistL2Projection, honee->dm, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief View the wall distance field projected onto the solution mesh

  Use `-walldistance_projection_view` to visualize the field.

  @param[in] honee              `Honee`
  @param[in] error_if_not_setup Whether to return an error if the wall distance struct hasn't been setup
**/
PetscErrorCode HoneeWallDistanceViewProjectedField(Honee honee, PetscBool error_if_not_setup) {
  HoneeWallDistance walldist_ctx;
  PetscBool         has_walldist, has_projection_view_option;

  PetscFunctionBeginUser;
  PetscCall(HoneeHasContainer(honee, WALLDIST_KEY, &has_walldist));
  if (!has_walldist && error_if_not_setup)
    SETERRQ(PetscObjectComm((PetscObject)honee), PETSC_ERR_ARG_WRONGSTATE,
            "Wall distance context has not been setup, so cannot view the projected wall distance field.");
  else if (!has_walldist) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscOptionsHasName(NULL, NULL, "-" WALLDIST_PREFIX "projection_view", &has_projection_view_option));
  if (!has_projection_view_option) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(HoneeWallDistanceGetProjectedField(honee, NULL, NULL, NULL));
  PetscCall(HoneeGetContainer(honee, WALLDIST_KEY, &walldist_ctx));
  PetscCall(VecViewFromOptions(walldist_ctx->Walldist, NULL, "-" WALLDIST_PREFIX "projection_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Run regression test for wall distance calculation

  @param[in] honee `Honee`
**/
PetscErrorCode HoneeWallDistanceTest(Honee honee) {
  Ceed                ceed = honee->ceed;
  CeedElemRestriction elem_restr_walldist;
  CeedVector          vec_walldist;
  Vec                 walldist_qpts;
  CeedSize            vec_walldist_size;

  PetscFunctionBeginUser;
  PetscCall(HoneeWallDistanceGetQuadratureValues(honee, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, 0, &elem_restr_walldist, &vec_walldist));

  PetscCallCeed(ceed, CeedVectorGetLength(vec_walldist, &vec_walldist_size));
  PetscCall(VecCreate(PETSC_COMM_SELF, &walldist_qpts));
  PetscCall(VecSetType(walldist_qpts, VECSEQ));
  PetscCall(VecSetSizes(walldist_qpts, vec_walldist_size, vec_walldist_size));
  {
    const CeedScalar *vec_walldist_ptr;
    PetscScalar      *walldist_qpts_ptr;

    PetscCallCeed(ceed, CeedVectorGetArrayRead(vec_walldist, CEED_MEM_HOST, &vec_walldist_ptr));
    PetscCall(VecGetArray(walldist_qpts, &walldist_qpts_ptr));
    for (CeedSize i = 0; i < vec_walldist_size; i++) walldist_qpts_ptr[i] = vec_walldist_ptr[i];
    PetscCall(VecRestoreArray(walldist_qpts, &walldist_qpts_ptr));
    PetscCallCeed(ceed, CeedVectorRestoreArrayRead(vec_walldist, &vec_walldist_ptr));
  }
  PetscCall(VecViewFromOptions(walldist_qpts, NULL, "-" WALLDIST_PREFIX "qpts_test_view"));

  PetscCall(RegressionTest(honee->app_ctx, walldist_qpts));
  honee->app_ctx->has_test_run = PETSC_TRUE;

  PetscCall(VecDestroy(&walldist_qpts));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_walldist));
  PetscCallCeed(ceed, CeedVectorDestroy(&vec_walldist));
  PetscFunctionReturn(PETSC_SUCCESS);
}
