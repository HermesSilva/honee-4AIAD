// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include <navierstokes.h>

#include <petscsection.h>
#include "../qfunctions/setupgeo.h"
#include "../qfunctions/setupgeo2d.h"

static CeedVector          *q_data_vecs;
static CeedElemRestriction *q_data_restrictions;
static PetscInt             num_q_data_stored;

/**
  @brief Get stored QData objects that match created objects, if present

  If created objects are not present, they are added to the storage and returned in the output

  Not Collective

  @param[in]  q_data_created        Vector with created QData
  @param[in]  elem_restr_qd_created Restriction for created QData
  @param[out] q_data_stored         Vector from storage matching QData
  @param[out] elem_restr_qd_stored  Restriction from storage matching QData
**/
static PetscErrorCode QDataGetStored(CeedVector q_data_created, CeedElemRestriction elem_restr_qd_created, CeedVector *q_data_stored,
                                     CeedElemRestriction *elem_restr_qd_stored) {
  Ceed     ceed = CeedVectorReturnCeed(q_data_created);
  CeedSize created_length, stored_length;
  PetscInt q_data_stored_index = -1;

  PetscFunctionBeginUser;
  PetscCallCeed(ceed, CeedVectorGetLength(q_data_created, &created_length));
  for (PetscInt i = 0; i < num_q_data_stored; i++) {
    CeedVector difference_cvec;
    CeedScalar max_difference;

    PetscCallCeed(ceed, CeedVectorGetLength(q_data_vecs[0], &stored_length));
    if (created_length != stored_length) continue;
    PetscCallCeed(ceed, CeedVectorCreate(ceed, stored_length, &difference_cvec));
    PetscCallCeed(ceed, CeedVectorCopy(q_data_vecs[i], difference_cvec));
    PetscCallCeed(ceed, CeedVectorAXPY(difference_cvec, -1, q_data_created));
    PetscCallCeed(ceed, CeedVectorNorm(difference_cvec, CEED_NORM_MAX, &max_difference));
    PetscCallCeed(ceed, CeedVectorDestroy(&difference_cvec));
    if (max_difference < 100 * CEED_EPSILON) {
      q_data_stored_index = i;
      break;
    }
  }

  if (q_data_stored_index == -1) {
    q_data_stored_index = num_q_data_stored++;

    PetscCall(PetscRealloc(num_q_data_stored * sizeof(CeedVector), &q_data_vecs));
    PetscCall(PetscRealloc(num_q_data_stored * sizeof(CeedElemRestriction), &q_data_restrictions));
    q_data_vecs[q_data_stored_index]         = NULL;  // Must set to NULL for ReferenceCopy
    q_data_restrictions[q_data_stored_index] = NULL;  // Must set to NULL for ReferenceCopy
    PetscCallCeed(ceed, CeedVectorReferenceCopy(q_data_created, &q_data_vecs[q_data_stored_index]));
    PetscCallCeed(ceed, CeedElemRestrictionReferenceCopy(elem_restr_qd_created, &q_data_restrictions[q_data_stored_index]));
  }
  *q_data_stored        = NULL;  // Must set to NULL for ReferenceCopy
  *elem_restr_qd_stored = NULL;  // Must set to NULL for ReferenceCopy
  PetscCallCeed(ceed, CeedVectorReferenceCopy(q_data_vecs[q_data_stored_index], q_data_stored));
  PetscCallCeed(ceed, CeedElemRestrictionReferenceCopy(q_data_restrictions[q_data_stored_index], elem_restr_qd_stored));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Clear stored QData objects
**/
PetscErrorCode QDataClearStoredData() {
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < num_q_data_stored; i++) {
    Ceed ceed = CeedVectorReturnCeed(q_data_vecs[i]);

    PetscCallCeed(ceed, CeedVectorDestroy(&q_data_vecs[i]));
    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&q_data_restrictions[i]));
  }
  PetscCall(PetscFree(q_data_vecs));
  PetscCall(PetscFree(q_data_restrictions));
  q_data_vecs         = NULL;
  q_data_restrictions = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Get number of components of quadrature data for domain
 *
 * @param[in]  dm          DM where quadrature data would be used
 * @param[out] q_data_size Number of components of quadrature data
 */
PetscErrorCode QDataGetNumComponents(DM dm, CeedInt *q_data_size) {
  PetscInt num_comp_x, dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateNumComps(dm, &num_comp_x));
  *q_data_size = 0;
  switch (dim) {
    case 2:
      switch (num_comp_x) {
        case 2:
          *q_data_size = 5;
          break;
        case 3:
          *q_data_size = 7;
          break;
      }
      break;
    case 3:
      *q_data_size = 10;
      break;
  }
  PetscCheck(*q_data_size, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP,
             "QData not valid for DM of dimension %" PetscInt_FMT " and coordinates with dimension %" PetscInt_FMT, dim, num_comp_x);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Create quadrature data for domain
 *
 * @param[in]  ceed          Ceed object quadrature data will be used with
 * @param[in]  dm            DM where quadrature data would be used
 * @param[in]  domain_label  DMLabel that quadrature data would be used one
 * @param[in]  label_value   Value of label
 * @param[in]  elem_restr_x  CeedElemRestriction of the coordinates (must match `domain_label` and `label_value` selections)
 * @param[in]  basis_x       CeedBasis of the coordinates
 * @param[in]  x_coord       CeedVector of the coordinates
 * @param[out] elem_restr_qd CeedElemRestriction of the quadrature data
 * @param[out] q_data        CeedVector of the quadrature data
 * @param[out] q_data_size   Number of components of quadrature data
 */
PetscErrorCode QDataGet(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, CeedElemRestriction elem_restr_x, CeedBasis basis_x,
                        CeedVector x_coord, CeedElemRestriction *elem_restr_qd, CeedVector *q_data, CeedInt *q_data_size) {
  CeedQFunction       qf_setup = NULL;
  CeedOperator        op_setup;
  CeedVector          q_data_created;
  CeedElemRestriction elem_restr_qd_created;
  CeedInt             num_comp_x;
  PetscInt            dim, height = 0;

  PetscFunctionBeginUser;
  PetscCall(QDataGetNumComponents(dm, q_data_size));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_x, &num_comp_x));
  PetscCall(DMGetDimension(dm, &dim));
  switch (dim) {
    case 2:
      switch (num_comp_x) {
        case 2:
          PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Setup2d, Setup2d_loc, &qf_setup));
          break;
        case 3:
          PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Setup2D_3Dcoords, Setup2D_3Dcoords_loc, &qf_setup));
          break;
      }
      break;
    case 3:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Setup, Setup_loc, &qf_setup));
      break;
  }
  PetscCheck(qf_setup, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "%s not valid for DM of dimension %" PetscInt_FMT, __func__, dim);

  // -- Create QFunction for quadrature data
  PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(qf_setup, 0));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_setup, "dx", num_comp_x * (dim - height), CEED_EVAL_GRAD));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_setup, "surface qdata", *q_data_size, CEED_EVAL_NONE));

  PetscCall(DMPlexCeedElemRestrictionQDataCreate(ceed, dm, domain_label, label_value, height, *q_data_size, &elem_restr_qd_created));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_qd_created, &q_data_created, NULL));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "dx", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "surface qdata", elem_restr_qd_created, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedOperatorApply(op_setup, x_coord, q_data_created, CEED_REQUEST_IMMEDIATE));

  PetscCall(QDataGetStored(q_data_created, elem_restr_qd_created, q_data, elem_restr_qd));

  PetscCallCeed(ceed, CeedVectorDestroy(&q_data_created));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd_created));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_setup));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_setup));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Get number of components of quadrature data for boundary of domain
 *
 * @param[in]  dm          DM where quadrature data would be used
 * @param[out] q_data_size Number of components of quadrature data
 */
PetscErrorCode QDataBoundaryGetNumComponents(DM dm, CeedInt *q_data_size) {
  PetscInt dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  *q_data_size = 0;
  switch (dim) {
    case 2:
      *q_data_size = 3;
      break;
    case 3:
      *q_data_size = 10;
      break;
  }
  PetscCheck(*q_data_size, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "%s not valid for DM of dimension %" PetscInt_FMT, __func__, dim);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Create quadrature data for boundary of domain
 *
 * @param[in]  ceed          Ceed object quadrature data will be used with
 * @param[in]  dm            DM where quadrature data would be used
 * @param[in]  domain_label  DMLabel that quadrature data would be used one
 * @param[in]  label_value   Value of label
 * @param[in]  elem_restr_x  CeedElemRestriction of the coordinates (must match `domain_label` and `label_value` selections)
 * @param[in]  basis_x       CeedBasis of the coordinates
 * @param[in]  x_coord       CeedVector of the coordinates
 * @param[out] elem_restr_qd CeedElemRestriction of the quadrature data
 * @param[out] q_data        CeedVector of the quadrature data
 * @param[out] q_data_size   Number of components of quadrature data
 */
PetscErrorCode QDataBoundaryGet(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, CeedElemRestriction elem_restr_x, CeedBasis basis_x,
                                CeedVector x_coord, CeedElemRestriction *elem_restr_qd, CeedVector *q_data, CeedInt *q_data_size) {
  CeedQFunction       qf_setup_sur = NULL;
  CeedOperator        op_setup_sur;
  CeedVector          q_data_created;
  CeedElemRestriction elem_restr_qd_created;
  CeedInt             num_comp_x;
  PetscInt            dim, height = 1;

  PetscFunctionBeginUser;
  PetscCall(QDataBoundaryGetNumComponents(dm, q_data_size));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_x, &num_comp_x));
  PetscCall(DMGetDimension(dm, &dim));
  switch (dim) {
    case 2:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, SetupBoundary2d, SetupBoundary2d_loc, &qf_setup_sur));
      break;
    case 3:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, SetupBoundary, SetupBoundary_loc, &qf_setup_sur));
      break;
  }
  PetscCheck(qf_setup_sur, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "%s not valid for DM of dimension %" PetscInt_FMT, __func__, dim);

  // -- Create QFunction for quadrature data
  PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(qf_setup_sur, 0));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_setup_sur, "dx", num_comp_x * (dim - height), CEED_EVAL_GRAD));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_setup_sur, "weight", 1, CEED_EVAL_WEIGHT));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_setup_sur, "surface qdata", *q_data_size, CEED_EVAL_NONE));

  PetscCall(DMPlexCeedElemRestrictionQDataCreate(ceed, dm, domain_label, label_value, height, *q_data_size, &elem_restr_qd_created));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_qd_created, &q_data_created, NULL));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_setup_sur, NULL, NULL, &op_setup_sur));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup_sur, "dx", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup_sur, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup_sur, "surface qdata", elem_restr_qd_created, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedOperatorApply(op_setup_sur, x_coord, q_data_created, CEED_REQUEST_IMMEDIATE));

  PetscCall(QDataGetStored(q_data_created, elem_restr_qd_created, q_data, elem_restr_qd));

  PetscCallCeed(ceed, CeedVectorDestroy(&q_data_created));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd_created));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_setup_sur));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_setup_sur));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Get number of components of quadrature data for boundary of domain
 *
 * @param[in]  dm          DM where quadrature data would be used
 * @param[out] q_data_size Number of components of quadrature data
 */
PetscErrorCode QDataBoundaryGradientGetNumComponents(DM dm, CeedInt *q_data_size) {
  PetscInt dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  *q_data_size = 0;
  switch (dim) {
    case 2:
      *q_data_size = 7;
      break;
    case 3:
      *q_data_size = 13;
      break;
  }
  PetscCheck(*q_data_size, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "%s not valid for DM of dimension %" PetscInt_FMT, __func__, dim);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Compute `CeedOperator` surface gradient QData

  Collective across MPI processes.

  @param[in]  ceed          `Ceed` object
  @param[in]  dm            `DMPlex` grid
  @param[in]  domain_label  `DMLabel` for surface
  @param[in]  label_value   `DMPlex` label value for surface
  @param[in]  x_coord       `CeedVector` for coordinates
  @param[out] elem_restr_qd `CeedElemRestriction` for QData
  @param[out] q_data        `CeedVector` holding QData
  @param[out] q_data_size   Number of QData components per quadrature point
**/
PetscErrorCode QDataBoundaryGradientGet(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, CeedVector x_coord,
                                        CeedElemRestriction *elem_restr_qd, CeedVector *q_data, CeedInt *q_data_size) {
  PetscInt            dim;
  const PetscInt      height_cell = 0, height_face = 1;
  CeedInt             num_comp_x;
  CeedElemRestriction elem_restr_x_cell, elem_restr_x_face, elem_restr_qd_created;
  CeedVector          q_data_created;
  CeedBasis           basis_x_cell_to_face, basis_x_face;
  CeedQFunction       qf_setup_sur = NULL;
  CeedOperator        op_setup_sur;

  PetscFunctionBeginUser;
  PetscCall(CreateCoordinateBasisFromPlex(ceed, dm, domain_label, label_value, height_face, &basis_x_face));
  PetscCall(DMPlexCeedBasisCellToFaceCoordinateCreate(ceed, dm, domain_label, label_value, label_value, &basis_x_cell_to_face));
  PetscCall(DMPlexCeedElemRestrictionCoordinateCreate(ceed, dm, domain_label, label_value, height_cell, &elem_restr_x_cell));
  PetscCall(DMPlexCeedElemRestrictionCoordinateCreate(ceed, dm, domain_label, label_value, height_face, &elem_restr_x_face));

  PetscCall(QDataBoundaryGradientGetNumComponents(dm, q_data_size));
  PetscCall(DMPlexCeedElemRestrictionQDataCreate(ceed, dm, domain_label, label_value, height_face, *q_data_size, &elem_restr_qd_created));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_qd_created, &q_data_created, NULL));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_x_face, &num_comp_x));

  PetscCall(DMGetDimension(dm, &dim));
  switch (dim) {
    case 2:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Setup2DBoundaryGradient, Setup2DBoundaryGradient_loc, &qf_setup_sur));
      break;
    case 3:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, SetupBoundaryGradient, SetupBoundaryGradient_loc, &qf_setup_sur));
      break;
  }
  PetscCheck(qf_setup_sur, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "%s not valid for DM of dimension %" PetscInt_FMT, __func__, dim);

  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_setup_sur, "dx/dX cell", num_comp_x * (dim - height_cell), CEED_EVAL_GRAD));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_setup_sur, "dx/dX face", num_comp_x * (dim - height_face), CEED_EVAL_GRAD));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_setup_sur, "weight", 1, CEED_EVAL_WEIGHT));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_setup_sur, "q data", *q_data_size, CEED_EVAL_NONE));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_setup_sur, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_sur));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup_sur, "dx/dX cell", elem_restr_x_cell, basis_x_cell_to_face, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup_sur, "dx/dX face", elem_restr_x_face, basis_x_face, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup_sur, "weight", CEED_ELEMRESTRICTION_NONE, basis_x_face, CEED_VECTOR_NONE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup_sur, "q data", elem_restr_qd_created, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedOperatorApply(op_setup_sur, x_coord, q_data_created, CEED_REQUEST_IMMEDIATE));

  PetscCall(QDataGetStored(q_data_created, elem_restr_qd_created, q_data, elem_restr_qd));

  PetscCallCeed(ceed, CeedOperatorDestroy(&op_setup_sur));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_setup_sur));
  PetscCallCeed(ceed, CeedVectorDestroy(&q_data_created));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd_created));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_x_face));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_x_cell));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_x_face));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_x_cell_to_face));
  PetscFunctionReturn(PETSC_SUCCESS);
}
