// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Utilities for setting up DM and PetscFE

#include <ceed.h>
#include <dm-utils.h>
#include <petsc-ceed-utils.h>
#include <petscdmplex.h>
#include <petscds.h>

/**
  @brief Convert `DM` field to `DS` field.

  @param[in]   dm            `DM` holding mesh
  @param[in]   domain_label  Label for `DM` domain
  @param[in]   dm_field      Index of `DM` field
  @param[out]  ds_field      Index of `DS` field

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode DMFieldToDSField(DM dm, DMLabel domain_label, PetscInt dm_field, PetscInt *ds_field) {
  PetscDS         ds;
  IS              field_is;
  const PetscInt *fields;
  PetscInt        num_fields;

  PetscFunctionBeginUser;
  PetscCall(DMGetRegionDS(dm, domain_label, &field_is, &ds, NULL));
  PetscCall(ISGetIndices(field_is, &fields));
  PetscCall(ISGetSize(field_is, &num_fields));
  for (PetscInt i = 0; i < num_fields; i++) {
    if (dm_field == fields[i]) {
      *ds_field = i;
      break;
    }
  }
  PetscCall(ISRestoreIndices(field_is, &fields));

  PetscCheck(*ds_field != -1, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Could not find dm_field %" PetscInt_FMT " in DS", dm_field);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Destroy `ElemRestriction` in a `PetscContainer`.

  Not collective across MPI processes.

  @param[in,out]  ctx  `CeedElemRestriction`

  @return An error code: 0 - success, otherwise - failure
**/
static PetscErrorCode DMPlexCeedElemRestrictionDestroy(void **ctx) {
  CeedElemRestriction rstr = *(CeedElemRestriction *)ctx;

  PetscFunctionBegin;
  PetscCallCeed(CeedElemRestrictionReturnCeed(rstr), CeedElemRestrictionDestroy(&rstr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create `CeedElemRestriction` from `DMPlex`.

 The `CeedElemRestriction` is stored in the `DMPlex` object for reuse;
 if the routine is called again with the same arguments, the same `CeedElemRestriction` is returned.

  Not collective across MPI processes.

  @param[in]   ceed          `Ceed` context
  @param[in]   dm            `DMPlex` holding mesh
  @param[in]   domain_label  `DMLabel` for `DMPlex` domain
  @param[in]   label_value   Stratum value
  @param[in]   height        Height of `DMPlex` topology
  @param[in]   dm_field      Index of `DMPlex` field
  @param[out]  restriction   `CeedElemRestriction` for `DMPlex`

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode DMPlexCeedElemRestrictionCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt dm_field,
                                               CeedElemRestriction *restriction) {
  char                container_name[1024];
  CeedElemRestriction container_restriction;

  PetscFunctionBeginUser;
  {
    const char *label_name = NULL;

    if (domain_label) PetscCall(PetscObjectGetName((PetscObject)domain_label, &label_name));
    PetscCall(PetscSNPrintf(container_name, sizeof(container_name),
                            "DM CeedElemRestriction - label: %s label value: %" PetscInt_FMT " height: %" PetscInt_FMT " DM field: %" PetscInt_FMT,
                            label_name ? label_name : "NONE", label_value, height, dm_field));
  }
  PetscCall(PetscObjectContainerQuery((PetscObject)dm, container_name, (void **)&container_restriction));

  if (container_restriction) {
    // Copy existing restriction
    *restriction = NULL;
    PetscCallCeed(ceed, CeedElemRestrictionReferenceCopy(container_restriction, restriction));
  } else {
    PetscInt num_elem, elem_size, num_dof, num_comp, *restriction_offsets_petsc;
    CeedInt *restriction_offsets_ceed = NULL;

    // Build restriction
    PetscCall(DMPlexGetLocalOffsets(dm, domain_label, label_value, height, dm_field, &num_elem, &elem_size, &num_comp, &num_dof,
                                    &restriction_offsets_petsc));
    PetscCall(IntArrayPetscToCeed(num_elem * elem_size, &restriction_offsets_petsc, &restriction_offsets_ceed));
    PetscCallCeed(ceed, CeedElemRestrictionCreate(ceed, num_elem, elem_size, num_comp, 1, num_dof, CEED_MEM_HOST, CEED_COPY_VALUES,
                                                  restriction_offsets_ceed, restriction));
    PetscCall(PetscFree(restriction_offsets_ceed));

    // Set in container
    PetscCallCeed(ceed, CeedElemRestrictionReferenceCopy(*restriction, &container_restriction));
    PetscCall(PetscObjectContainerCompose((PetscObject)dm, container_name, container_restriction,
                                          (PetscCtxDestroyFn *)DMPlexCeedElemRestrictionDestroy));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create `CeedElemRestriction` from `DMPlex` domain for mesh coordinates.

 The `CeedElemRestriction` is stored in the coordinate `DMPlex` object for reuse;
 if the routine is called again with the same arguments, the same `CeedElemRestriction` is returned.

  Not collective across MPI processes.

  @param[in]   ceed          `Ceed` context
  @param[in]   dm            `DMPlex` holding mesh
  @param[in]   domain_label  Label for `DMPlex` domain
  @param[in]   label_value   Stratum value
  @param[in]   height        Height of `DMPlex` topology
  @param[out]  restriction   `CeedElemRestriction` for mesh

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode DMPlexCeedElemRestrictionCoordinateCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height,
                                                         CeedElemRestriction *restriction) {
  DM dm_coord;

  PetscFunctionBeginUser;
  PetscCall(DMGetCellCoordinateDM(dm, &dm_coord));
  if (!dm_coord) {
    PetscCall(DMGetCoordinateDM(dm, &dm_coord));
  }
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, dm_coord, domain_label, label_value, height, 0, restriction));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create `CeedElemRestriction` from `DMPlex` domain for auxilury `QFunction` data.

  Not collective across MPI processes.

  @param[in]   ceed           `Ceed` context
  @param[in]   dm             `DMPlex` holding mesh
  @param[in]   domain_label   Label for `DMPlex` domain
  @param[in]   label_value    Stratum value
  @param[in]   height         Height of `DMPlex` topology
  @param[in]   q_data_size    Number of components for `QFunction` data
  @param[in]   is_collocated  Boolean flag indicating if the data is collocated on the nodes (`PETSC_TRUE`) or on quadrature points (`PETSC_FALSE`)
  @param[out]  restriction    Strided `CeedElemRestriction` for `QFunction` data

  @return An error code: 0 - success, otherwise - failure
**/
static PetscErrorCode DMPlexCeedElemRestrictionStridedCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height,
                                                             PetscInt q_data_size, PetscBool is_collocated, CeedElemRestriction *restriction) {
  PetscInt num_elem, num_qpts, dm_field = 0;

  PetscFunctionBeginUser;
  {  // Get number of elements
    PetscInt depth;
    DMLabel  depth_label;
    IS       point_is, depth_is;

    PetscCall(DMPlexGetDepth(dm, &depth));
    PetscCall(DMPlexGetDepthLabel(dm, &depth_label));
    PetscCall(DMLabelGetStratumIS(depth_label, depth - height, &depth_is));
    if (domain_label) {
      IS domain_is;

      PetscCall(DMLabelGetStratumIS(domain_label, label_value, &domain_is));
      if (domain_is) {
        PetscCall(ISIntersect(depth_is, domain_is, &point_is));
        PetscCall(ISDestroy(&domain_is));
      } else {
        point_is = NULL;
      }
      PetscCall(ISDestroy(&depth_is));
    } else {
      point_is = depth_is;
    }
    if (point_is) {
      PetscCall(ISGetLocalSize(point_is, &num_elem));
    } else {
      num_elem = 0;
    }
    PetscCall(ISDestroy(&point_is));
  }

  {  // Get number of quadrature points
    PetscDS  ds;
    PetscFE  fe;
    PetscInt ds_field = -1;

    PetscCall(DMGetRegionDS(dm, domain_label, NULL, &ds, NULL));
    PetscCall(DMFieldToDSField(dm, domain_label, dm_field, &ds_field));
    PetscCall(PetscDSGetDiscretization(ds, ds_field, (PetscObject *)&fe));
    PetscCall(PetscFEGetHeightSubspace(fe, height, &fe));
    if (is_collocated) {
      PetscDualSpace dual_space;
      PetscInt       num_dual_basis_vectors, dim, num_comp;

      PetscCall(PetscFEGetSpatialDimension(fe, &dim));
      PetscCall(PetscFEGetNumComponents(fe, &num_comp));
      PetscCall(PetscFEGetDualSpace(fe, &dual_space));
      PetscCall(PetscDualSpaceGetDimension(dual_space, &num_dual_basis_vectors));
      num_qpts = num_dual_basis_vectors / num_comp;
    } else {
      PetscQuadrature quadrature;

      PetscCall(DMGetRegionDS(dm, domain_label, NULL, &ds, NULL));
      PetscCall(DMFieldToDSField(dm, domain_label, dm_field, &ds_field));
      PetscCall(PetscDSGetDiscretization(ds, ds_field, (PetscObject *)&fe));
      PetscCall(PetscFEGetHeightSubspace(fe, height, &fe));
      PetscCall(PetscFEGetQuadrature(fe, &quadrature));
      PetscCall(PetscQuadratureGetData(quadrature, NULL, NULL, &num_qpts, NULL, NULL));
    }
  }

  // Create the restriction
  PetscCallCeed(ceed, CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, q_data_size, num_elem * num_qpts * q_data_size, CEED_STRIDES_BACKEND,
                                                       restriction));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create `CeedElemRestriction` from `DMPlex` domain for auxilury `QFunction` data.

  Not collective across MPI processes.

  @param[in]   ceed           `Ceed` context
  @param[in]   dm             `DMPlex` holding mesh
  @param[in]   domain_label   Label for `DMPlex` domain
  @param[in]   label_value    Stratum value
  @param[in]   height         Height of `DMPlex` topology
  @param[in]   q_data_size    Number of components for `QFunction` data
  @param[out]  restriction    Strided `CeedElemRestriction` for `QFunction` data

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode DMPlexCeedElemRestrictionQDataCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height,
                                                    PetscInt q_data_size, CeedElemRestriction *restriction) {
  PetscFunctionBeginUser;
  PetscCall(DMPlexCeedElemRestrictionStridedCreate(ceed, dm, domain_label, label_value, height, q_data_size, PETSC_FALSE, restriction));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create `CeedElemRestriction` from `DMPlex` domain for nodally collocated auxilury `QFunction` data.

  Not collective across MPI processes.

  @param[in]   ceed           `Ceed` context
  @param[in]   dm             `DMPlex` holding mesh
  @param[in]   domain_label   Label for `DMPlex` domain
  @param[in]   label_value    Stratum value
  @param[in]   height         Height of `DMPlex` topology
  @param[in]   q_data_size    Number of components for `QFunction` data
  @param[out]  restriction    Strided `CeedElemRestriction` for `QFunction` data

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode DMPlexCeedElemRestrictionCollocatedCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height,
                                                         PetscInt q_data_size, CeedElemRestriction *restriction) {
  PetscFunctionBeginUser;
  PetscCall(DMPlexCeedElemRestrictionStridedCreate(ceed, dm, domain_label, label_value, height, q_data_size, PETSC_TRUE, restriction));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Creates a tensor-product uniform quadrature.
         This is only for comparison studies and generally should not be used.

  @param[in]   dim         Spatial dimension
  @param[in]   num_comp    Number of components
  @param[in]   num_points  Number of points in one dimension
  @param[in]   a           Left end of interval (often -1)
  @param[in]   b           Right end of interval (often +1)
  @param[out]  q           `PetscQuadrature` object

  @return An error code: 0 - success, otherwise - failure
**/
static PetscErrorCode PetscDTUniformTensorQuadrature(PetscInt dim, PetscInt num_comp, PetscInt num_points, PetscReal a, PetscReal b,
                                                     PetscQuadrature *q) {
  DMPolytopeType ct;
  PetscInt       num_total_points = dim > 1 ? (dim > 2 ? (num_points * num_points * num_points) : (num_points * num_points)) : num_points;
  PetscReal     *coords, *weights, *coords_1d;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(num_total_points * dim, &coords));
  PetscCall(PetscMalloc1(num_total_points * num_comp, &weights));
  // Compute weights, points
  switch (dim) {
    case 0:
      ct = DM_POLYTOPE_POINT;
      PetscCall(PetscFree(coords));
      PetscCall(PetscFree(weights));
      PetscCall(PetscMalloc1(1, &coords));
      PetscCall(PetscMalloc1(num_comp, &weights));
      num_total_points = 1;
      coords[0]        = 0.0;
      for (PetscInt c = 0; c < num_comp; c++) weights[c] = 1.0;
      break;
    case 1: {
      ct             = DM_POLYTOPE_SEGMENT;
      PetscReal step = (b - a) / num_points;

      for (PetscInt i = 0; i < num_points; i++) {
        coords[i] = step * (i + 0.5) + a;
        for (PetscInt c = 0; c < num_comp; c++) weights[i * num_comp + c] = 1.0;
      }
    } break;
    case 2: {
      ct = DM_POLYTOPE_QUADRILATERAL;
      PetscCall(PetscMalloc1(num_points, &coords_1d));
      PetscReal step = (b - a) / num_points;

      for (PetscInt i = 0; i < num_points; i++) coords_1d[i] = step * (i + 0.5) + a;
      for (PetscInt i = 0; i < num_points; i++) {
        for (PetscInt j = 0; j < num_points; j++) {
          coords[(i * num_points + j) * dim + 0] = coords_1d[i];
          coords[(i * num_points + j) * dim + 1] = coords_1d[j];
          for (PetscInt c = 0; c < num_comp; c++) weights[(i * num_points + j) * num_comp + c] = 1.0;
        }
      }
      PetscCall(PetscFree(coords_1d));
    } break;
    case 3: {
      ct = DM_POLYTOPE_HEXAHEDRON;
      PetscCall(PetscMalloc1(num_points, &coords_1d));
      PetscReal step = (b - a) / num_points;

      for (PetscInt i = 0; i < num_points; i++) coords_1d[i] = step * (i + 0.5) + a;
      for (PetscInt i = 0; i < num_points; i++) {
        for (PetscInt j = 0; j < num_points; j++) {
          for (PetscInt k = 0; k < num_points; k++) {
            coords[((i * num_points + j) * num_points + k) * dim + 0] = coords_1d[i];
            coords[((i * num_points + j) * num_points + k) * dim + 1] = coords_1d[j];
            coords[((i * num_points + j) * num_points + k) * dim + 2] = coords_1d[k];
            for (PetscInt c = 0; c < num_comp; c++) weights[((i * num_points + j) * num_points + k) * num_comp + c] = 1.0;
          }
        }
      }
      PetscCall(PetscFree(coords_1d));
    } break;
    // LCOV_EXCL_START
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot construct quadrature rule for dimension %" PetscInt_FMT, dim);
      // LCOV_EXCL_STOP
  }
  PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, q));
  PetscCall(PetscQuadratureSetCellType(*q, ct));
  PetscCall(PetscQuadratureSetOrder(*q, num_points));
  PetscCall(PetscQuadratureSetData(*q, dim, num_comp, num_total_points, coords, weights));
  PetscCall(PetscObjectChangeTypeName((PetscObject)*q, "UniformTensor"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Setup `DM` with FE space of appropriate degree

  @param[in]   comm        MPI communicator
  @param[in]   dim         Spatial dimension
  @param[in]   num_comp    Number of components
  @param[in]   is_simplex  Flag for simplex or tensor product element
  @param[in]   order       Polynomial order of space
  @param[in]   q_order     Quadrature order
  @param[in]   prefix      The options prefix, or `NULL`
  @param[out]  fem         `PetscFE` object

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode PetscFECreateLagrangeFromOptions(MPI_Comm comm, PetscInt dim, PetscInt num_comp, PetscBool is_simplex, PetscInt order,
                                                PetscInt q_order, const char prefix[], PetscFE *fem) {
  PetscBool       is_tensor     = !is_simplex;
  DMPolytopeType  polytope_type = DMPolytopeTypeSimpleShape(dim, is_simplex);
  PetscSpace      fe_space;
  PetscDualSpace  fe_dual_space;
  PetscQuadrature quadrature, face_quadrature;

  PetscFunctionBeginUser;
  // Create space
  PetscCall(PetscSpaceCreate(comm, &fe_space));
  PetscCall(PetscSpaceSetType(fe_space, PETSCSPACEPOLYNOMIAL));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)fe_space, prefix));
  PetscCall(PetscSpacePolynomialSetTensor(fe_space, is_tensor));
  PetscCall(PetscSpaceSetNumComponents(fe_space, num_comp));
  PetscCall(PetscSpaceSetNumVariables(fe_space, dim));
  if (order >= 0) {
    PetscCall(PetscSpaceSetDegree(fe_space, order, PETSC_DETERMINE));
    if (polytope_type == DM_POLYTOPE_TRI_PRISM || polytope_type == DM_POLYTOPE_TRI_PRISM_TENSOR) {
      PetscSpace fe_space_end, fe_space_side;

      PetscCall(PetscSpaceSetNumComponents(fe_space, 1));
      PetscCall(PetscSpaceCreate(comm, &fe_space_end));
      PetscCall(PetscSpaceSetType(fe_space_end, PETSCSPACEPOLYNOMIAL));
      PetscCall(PetscSpacePolynomialSetTensor(fe_space_end, PETSC_FALSE));
      PetscCall(PetscSpaceSetNumComponents(fe_space_end, 1));
      PetscCall(PetscSpaceSetNumVariables(fe_space_end, dim - 1));
      PetscCall(PetscSpaceSetDegree(fe_space_end, order, PETSC_DETERMINE));
      PetscCall(PetscSpaceCreate(comm, &fe_space_side));
      PetscCall(PetscSpaceSetType(fe_space_side, PETSCSPACEPOLYNOMIAL));
      PetscCall(PetscSpacePolynomialSetTensor(fe_space_side, PETSC_FALSE));
      PetscCall(PetscSpaceSetNumComponents(fe_space_side, 1));
      PetscCall(PetscSpaceSetNumVariables(fe_space_side, 1));
      PetscCall(PetscSpaceSetDegree(fe_space_side, order, PETSC_DETERMINE));
      PetscCall(PetscSpaceSetType(fe_space, PETSCSPACETENSOR));
      PetscCall(PetscSpaceTensorSetNumSubspaces(fe_space, 2));
      PetscCall(PetscSpaceTensorSetSubspace(fe_space, 0, fe_space_end));
      PetscCall(PetscSpaceTensorSetSubspace(fe_space, 1, fe_space_side));
      PetscCall(PetscSpaceDestroy(&fe_space_end));
      PetscCall(PetscSpaceDestroy(&fe_space_side));

      if (num_comp > 1) {
        PetscSpace scalar_fe_space = fe_space;

        PetscCall(PetscSpaceCreate(comm, &fe_space));
        PetscCall(PetscSpaceSetNumVariables(fe_space, dim));
        PetscCall(PetscSpaceSetNumComponents(fe_space, num_comp));
        PetscCall(PetscSpaceSetType(fe_space, PETSCSPACESUM));
        PetscCall(PetscSpaceSumSetNumSubspaces(fe_space, num_comp));
        PetscCall(PetscSpaceSumSetConcatenate(fe_space, PETSC_TRUE));
        PetscCall(PetscSpaceSumSetInterleave(fe_space, PETSC_TRUE, PETSC_FALSE));
        for (PetscInt i = 0; i < num_comp; i++) PetscCall(PetscSpaceSumSetSubspace(fe_space, i, scalar_fe_space));
        PetscCall(PetscSpaceDestroy(&scalar_fe_space));
      }
    }
  }
  PetscCall(PetscSpaceSetFromOptions(fe_space));
  PetscCall(PetscSpaceSetUp(fe_space));
  PetscCall(PetscSpaceGetDegree(fe_space, &order, NULL));
  PetscCall(PetscSpacePolynomialGetTensor(fe_space, &is_tensor));
  PetscCall(PetscSpaceGetNumComponents(fe_space, &num_comp));

  // Create dual space
  PetscCall(PetscDualSpaceCreate(comm, &fe_dual_space));
  PetscCall(PetscDualSpaceSetType(fe_dual_space, PETSCDUALSPACELAGRANGE));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)fe_dual_space, prefix));
  {
    DM dual_space_dm;

    PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_SELF, polytope_type, &dual_space_dm));
    PetscCall(PetscDualSpaceSetDM(fe_dual_space, dual_space_dm));
    PetscCall(DMDestroy(&dual_space_dm));
  }
  PetscCall(PetscDualSpaceSetNumComponents(fe_dual_space, num_comp));
  PetscCall(PetscDualSpaceSetOrder(fe_dual_space, order));
  PetscCall(PetscDualSpaceLagrangeSetTensor(fe_dual_space, (is_tensor || (polytope_type == DM_POLYTOPE_TRI_PRISM)) ? PETSC_TRUE : PETSC_FALSE));
  PetscCall(PetscDualSpaceSetFromOptions(fe_dual_space));
  PetscCall(PetscDualSpaceSetUp(fe_dual_space));

  // Create quadrature
  q_order               = q_order >= 0 ? q_order : order;
  PetscBool use_uniform = PETSC_FALSE;

  PetscOptionsBegin(comm, NULL, "Uniform quadrature check", NULL);
  PetscCall(PetscOptionsBool("-petscdt_uniform_tensor_quadrature", "", NULL, use_uniform, &use_uniform, NULL));
  PetscOptionsEnd();
  PetscCheck(!use_uniform || (is_tensor || dim == 1), comm, PETSC_ERR_SUP, "Can only use uniform quadrature on tensor product elements");
  if (use_uniform) {
    PetscCall(PetscDTUniformTensorQuadrature(dim, 1, q_order + 1, -1.0, 1.0, &quadrature));
    PetscCall(PetscDTUniformTensorQuadrature(dim - 1, 1, q_order + 1, -1.0, 1.0, &face_quadrature));
  } else {
    PetscCall(PetscDTCreateDefaultQuadrature(polytope_type, q_order, &quadrature, &face_quadrature));
  }

  // Create finite element
  PetscCall(PetscFECreateFromSpaces(fe_space, fe_dual_space, quadrature, face_quadrature, fem));
  PetscCall(PetscFESetFromOptions(*fem));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Get global `DMPlex` topology type.

  Collective across MPI processes.

  @param[in]   dm            `DMPlex` holding mesh
  @param[in]   domain_label  `DMLabel` for `DMPlex` domain
  @param[in]   label_value   Stratum value
  @param[in]   height        Height of `DMPlex` topology
  @param[out]  cell_type     `DMPlex` topology type
**/
static inline PetscErrorCode GetGlobalDMPlexPolytopeType(DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height,
                                                         DMPolytopeType *cell_type) {
  PetscInt first_point;
  PetscInt ids[1] = {label_value};
  DMLabel  depth_label;

  PetscFunctionBeginUser;
  // Use depth label if no domain label present
  if (!domain_label) {
    PetscInt depth;

    PetscCall(DMPlexGetDepth(dm, &depth));
    PetscCall(DMPlexGetDepthLabel(dm, &depth_label));
    ids[0] = depth - height;
  }

  // Get cell interp, grad, and quadrature data
  PetscCall(DMGetFirstLabeledPoint(dm, dm, domain_label ? domain_label : depth_label, 1, ids, height, &first_point, NULL));
  if (first_point != -1) PetscCall(DMPlexGetCellType(dm, first_point, cell_type));

  {  // Form agreement about CellType
    PetscInt cell_type_local = -1, cell_type_global = -1;

    if (first_point != -1) cell_type_local = (PetscInt)*cell_type;
    PetscCallMPI(MPIU_Allreduce(&cell_type_local, &cell_type_global, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm)));
    *cell_type = (DMPolytopeType)cell_type_global;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Get the permutation and field offset for a given depth.

  Not collective across MPI processes.

  @param[in]   dm            `DMPlex` holding mesh
  @param[in]   depth         Depth of `DMPlex` topology
  @param[in]   field         Index of `DMPlex` field
  @param[out]  permutation   Permutation for `DMPlex` field
  @param[out]  field_offset  Offset to `DMPlex` field
**/
static inline PetscErrorCode GetClosurePermutationAndFieldOffsetAtDepth(DM dm, PetscInt depth, PetscInt field, IS *permutation,
                                                                        PetscInt *field_offset) {
  PetscBool    is_field_continuous = PETSC_TRUE;
  PetscInt     dim, num_fields, offset = 0, size = 0;
  PetscSection section;

  PetscFunctionBeginUser;
  *field_offset = 0;

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionGetNumFields(section, &num_fields));
  // ---- Permutation size and offset to current field
  for (PetscInt f = 0; f < num_fields; f++) {
    PetscBool      is_continuous;
    PetscInt       num_components, num_dof_1d, dual_space_size, field_size;
    PetscObject    obj;
    PetscFE        fe = NULL;
    PetscDualSpace dual_space;

    PetscCall(PetscSectionGetFieldComponents(section, f, &num_components));
    PetscCall(DMGetField(dm, f, NULL, &obj));
    fe = (PetscFE)obj;
    PetscCall(PetscFEGetDualSpace(fe, &dual_space));
    PetscCall(PetscDualSpaceLagrangeGetContinuity(dual_space, &is_continuous));
    if (f == field) is_field_continuous = is_continuous;
    if (!is_continuous) continue;
    PetscCall(PetscDualSpaceGetDimension(dual_space, &dual_space_size));
    num_dof_1d = (PetscInt)PetscCeilReal(PetscPowReal(dual_space_size / num_components, 1.0 / dim));
    field_size = PetscPowInt(num_dof_1d, depth) * num_components;
    if (f < field) offset += field_size;
    size += field_size;
  }

  if (is_field_continuous) {
    *field_offset = offset;
    PetscCall(PetscSectionGetClosurePermutation(section, (PetscObject)dm, depth, size, permutation));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Compute field tabulation from `PetscTabulation`.

  Not collective across MPI processes.

  @param[in]   dm          `DMPlex` holding mesh
  @param[in]   field       Index of `DMPlex` field
  @param[in]   face        Index of basis face, or 0
  @param[in]   tabulation  PETSc basis tabulation
  @param[out]  interp      Interpolation matrix in libCEED orientation
  @param[out]  grad        Gradient matrix in libCEED orientation

  @note `interp` and `grad` are allocated using `PetscCalloc1` and must be freed by the user.
**/
static inline PetscErrorCode ComputeFieldTabulationP2C(DM dm, PetscInt field, PetscInt face, PetscTabulation tabulation, CeedScalar **interp,
                                                       CeedScalar **grad) {
  PetscBool       is_simplex, has_permutation = PETSC_FALSE;
  PetscInt        field_offset = 0, num_comp, P, Q, dim;
  const PetscInt *permutation_indices;
  IS              permutation = NULL;

  PetscFunctionBeginUser;
  num_comp = tabulation->Nc;
  P        = tabulation->Nb / num_comp;
  Q        = tabulation->Np;
  dim      = tabulation->cdim;

  PetscCall(PetscCalloc1(P * Q, interp));
  PetscCall(PetscCalloc1(P * Q * dim, grad));

  PetscCall(DMPlexIsSimplex(dm, &is_simplex));
  if (!is_simplex) {
    PetscCall(GetClosurePermutationAndFieldOffsetAtDepth(dm, dim, field, &permutation, &field_offset));
    has_permutation = !!permutation;
    if (has_permutation) PetscCall(ISGetIndices(permutation, &permutation_indices));
  }

  const CeedInt c = 0;
  for (CeedInt q = 0; q < Q; q++) {
    for (CeedInt p_ceed = 0; p_ceed < P; p_ceed++) {
      CeedInt p_petsc           = has_permutation ? (permutation_indices[field_offset + p_ceed * num_comp] - field_offset) : (p_ceed * num_comp);
      (*interp)[q * P + p_ceed] = tabulation->T[0][((face * Q + q) * P * num_comp + p_petsc) * num_comp + c];
      for (CeedInt d = 0; d < dim; d++) {
        (*grad)[(d * Q + q) * P + p_ceed] = tabulation->T[1][(((face * Q + q) * P * num_comp + p_petsc) * num_comp + c) * dim + d];
      }
    }
  }

  if (has_permutation) PetscCall(ISRestoreIndices(permutation, &permutation_indices));
  PetscCall(ISDestroy(&permutation));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Get quadrature data from `PetscQuadrature` for use with libCEED.

  Not collective across MPI processes.

  @param[in]   fe          `PetscFE` object
  @param[in]   quadrature  PETSc basis quadrature
  @param[out]  q_dim       Quadrature dimension, or NULL if not needed
  @param[out]  num_comp    Number of components, or NULL if not needed
  @param[out]  Q           Number of quadrature points, or NULL if not needed
  @param[out]  q_points    Quadrature points in libCEED orientation, or NULL if not needed
  @param[out]  q_weights   Quadrature weights in libCEED orientation, or NULL if not needed

  @note `q_weights` and `q_points` are allocated using `PetscCalloc1` and must be freed by the user.
**/
static inline PetscErrorCode GetQuadratureDataP2C(PetscFE fe, PetscQuadrature quadrature, PetscInt *q_dim, PetscInt *num_comp, PetscInt *Q,
                                                  CeedScalar **q_points, CeedScalar **q_weights) {
  PetscInt         spatial_dim, dim, num_comp_quadrature, num_q_points;
  const PetscReal *q_points_petsc, *q_weights_petsc;

  PetscFunctionBeginUser;
  PetscCall(PetscFEGetSpatialDimension(fe, &spatial_dim));
  PetscCall(PetscQuadratureGetData(quadrature, &dim, &num_comp_quadrature, &num_q_points, &q_points_petsc, &q_weights_petsc));
  if (q_dim) *q_dim = dim;
  if (num_comp) *num_comp = num_comp_quadrature;
  if (Q) *Q = num_q_points;
  if (q_weights) {
    PetscSizeT q_weights_size = num_q_points;

    PetscCall(PetscCalloc1(q_weights_size, q_weights));
    for (CeedInt i = 0; i < num_q_points; i++) (*q_weights)[i] = q_weights_petsc[i];
  }
  if (q_points) {
    PetscSizeT q_points_size = num_q_points * spatial_dim;

    PetscCall(PetscCalloc1(q_points_size, q_points));
    for (CeedInt i = 0; i < num_q_points; i++) {
      for (CeedInt d = 0; d < dim; d++) (*q_points)[i * spatial_dim + d] = q_points_petsc[i * dim + d];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create 1D tabulation from `PetscFE`.

  Collective across MPI processes.

  @note `q_weights` and `q_points` are allocated using `PetscCalloc1` and must be freed by the user.

  @param[in]   comm          `MPI_Comm` for creating `FE` object from options
  @param[in]   fe            PETSc basis `FE` object
  @param[out]  tabulation    PETSc basis tabulation
  @param[out]  q_points_1d   Quadrature points in libCEED orientation
  @param[out]  q_weights_1d  Quadrature weights in libCEED orientation

  @return An error code: 0 - success, otherwise - failure
**/
static inline PetscErrorCode Create1DTabulation_Tensor(MPI_Comm comm, PetscFE fe, PetscTabulation *tabulation, PetscReal **q_points_1d,
                                                       CeedScalar **q_weights_1d) {
  PetscFE         fe_1d;
  PetscInt        dim, num_comp, Q = -1, q_dim = -1, num_derivatives = 1;
  PetscQuadrature quadrature;

  PetscFunctionBeginUser;
  // Create 1D FE
  PetscCall(PetscFEGetNumComponents(fe, &num_comp));
  PetscCall(PetscFEGetSpatialDimension(fe, &dim));
  {
    const char     *prefix;
    PetscBool       is_tensor;
    PetscInt        num_comp, order, q_order;
    PetscDualSpace  dual_space;
    PetscQuadrature quadrature;

    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)fe, &prefix));
    PetscCall(PetscFEGetDualSpace(fe, &dual_space));
    PetscCall(PetscFEGetNumComponents(fe, &num_comp));
    PetscCall(PetscDualSpaceLagrangeGetTensor(dual_space, &is_tensor));
    PetscCall(PetscDualSpaceGetOrder(dual_space, &order));
    PetscCall(PetscFEGetQuadrature(fe, &quadrature));
    PetscCall(PetscQuadratureGetData(quadrature, NULL, NULL, &q_order, NULL, NULL));
    PetscCall(PetscFECreateLagrangeFromOptions(comm, 1, num_comp, !is_tensor, order,
                                               (PetscInt)PetscCeilReal(PetscPowReal(1.0 * q_order, 1.0 / dim)) - 1, prefix, &fe_1d));
  }

  // Get quadrature data
  PetscCall(PetscFEGetQuadrature(fe_1d, &quadrature));
  PetscCall(GetQuadratureDataP2C(fe_1d, quadrature, &q_dim, NULL, &Q, q_points_1d, q_weights_1d));

  // Compute 1D tabulation
  PetscCall(PetscFECreateTabulation(fe_1d, 1, Q, *q_points_1d, num_derivatives, tabulation));

  // Cleanup
  PetscCall(PetscFEDestroy(&fe_1d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Destroy `CeedBasis` in a `PetscContainer`.

  Not collective across MPI processes.

  @param[in,out]  ctx  `CeedBasis`

  @return An error code: 0 - success, otherwise - failure
**/
static PetscErrorCode DMPlexCeedBasisDestroy(void **ctx) {
  CeedBasis basis = *(CeedBasis *)ctx;

  PetscFunctionBegin;
  PetscCallCeed(CeedBasisReturnCeed(basis), CeedBasisDestroy(&basis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create `CeedBasis` from `DMPlex`.

 The `CeedBasis` is stored in the `DMPlex` object for reuse;
 if the routine is called again with the same arguments, the same `CeedBasis` is returned.

  Collective across MPI processes.

  @param[in]   ceed          `Ceed` context
  @param[in]   dm            `DMPlex` holding mesh
  @param[in]   domain_label  `DMLabel` for `DMPlex` domain
  @param[in]   label_value   Stratum value
  @param[in]   height        Height of `DMPlex` topology
  @param[in]   dm_field      Index of `DMPlex` field
  @param[out]  basis         `CeedBasis` for `DMPlex`

  @return An error code: 0 - success, otherwise - failure
  **/
PetscErrorCode DMPlexCeedBasisCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt dm_field,
                                     CeedBasis *basis) {
  MPI_Comm  comm = PetscObjectComm((PetscObject)dm);
  char      container_name[1024];
  CeedBasis container_basis;

  PetscFunctionBeginUser;
  {
    const char *label_name = NULL;

    if (domain_label) PetscCall(PetscObjectGetName((PetscObject)domain_label, &label_name));
    PetscCall(PetscSNPrintf(container_name, sizeof(container_name),
                            "DM CeedBasis - label: %s label value: %" PetscInt_FMT " height: %" PetscInt_FMT " DM field: %" PetscInt_FMT,
                            label_name ? label_name : "NONE", label_value, height, dm_field));
  }
  PetscCall(PetscObjectContainerQuery((PetscObject)dm, container_name, (void **)&container_basis));

  if (container_basis) {
    // Copy existing basis
    *basis = NULL;
    PetscCallCeed(ceed, CeedBasisReferenceCopy(container_basis, basis));
  } else {
    PetscDS         ds;
    PetscFE         fe;
    PetscDualSpace  dual_space;
    PetscQuadrature quadrature;
    PetscBool       is_tensor = PETSC_TRUE;
    PetscInt        ds_field  = -1;

    // Get element information
    PetscCall(DMGetRegionDS(dm, domain_label, NULL, &ds, NULL));
    PetscCall(DMFieldToDSField(dm, domain_label, dm_field, &ds_field));
    PetscCall(PetscDSGetDiscretization(ds, ds_field, (PetscObject *)&fe));
    PetscCall(PetscFEGetHeightSubspace(fe, height, &fe));
    PetscCall(PetscFEGetQuadrature(fe, &quadrature));
    PetscCall(PetscFEGetDualSpace(fe, &dual_space));
    PetscCall(PetscDualSpaceLagrangeGetTensor(dual_space, &is_tensor));

    // Build libCEED basis
    PetscBool use_nontensor = PETSC_FALSE;

    PetscOptionsBegin(comm, NULL, "Tensor Basis as Nontensor Check", NULL);
    PetscCall(PetscOptionsBool("-ceed_basis_as_nontensor", "", NULL, use_nontensor, &use_nontensor, NULL));
    PetscOptionsEnd();

    if (!is_tensor || use_nontensor) {
      PetscTabulation  basis_tabulation;
      PetscInt         num_derivatives = 1, face = 0;
      CeedScalar      *q_points, *q_weights, *interp, *grad;
      CeedElemTopology elem_topo;

      {
        DMPolytopeType cell_type;

        PetscCall(GetGlobalDMPlexPolytopeType(dm, domain_label, label_value, height, &cell_type));
        elem_topo = PolytopeTypePetscToCeed(cell_type);
        PetscCheck(elem_topo, comm, PETSC_ERR_SUP, "DMPlex topology not supported: %s", DMPolytopeTypes[cell_type]);
      }

      // Compute basis tabulation
      PetscCall(GetQuadratureDataP2C(fe, quadrature, NULL, NULL, NULL, &q_points, &q_weights));
      PetscCall(PetscFEGetCellTabulation(fe, num_derivatives, &basis_tabulation));
      PetscCall(ComputeFieldTabulationP2C(dm, dm_field, face, basis_tabulation, &interp, &grad));
      {
        PetscInt num_comp = basis_tabulation->Nc, P = basis_tabulation->Nb / num_comp, Q = basis_tabulation->Np;

        PetscCallCeed(ceed, CeedBasisCreateH1(ceed, elem_topo, num_comp, P, Q, interp, grad, q_points, q_weights, basis));
      }

      PetscCall(PetscFree(q_points));
      PetscCall(PetscFree(q_weights));
      PetscCall(PetscFree(interp));
      PetscCall(PetscFree(grad));
    } else {
      PetscInt        P_1d, Q_1d, num_comp, dim;
      PetscTabulation basis_tabulation;
      CeedScalar     *q_points_1d, *q_weights_1d, *interp_1d, *grad_1d;

      PetscCall(PetscFEGetSpatialDimension(fe, &dim));
      PetscCall(Create1DTabulation_Tensor(comm, fe, &basis_tabulation, &q_points_1d, &q_weights_1d));
      num_comp = basis_tabulation->Nc;
      P_1d     = basis_tabulation->Nb / num_comp;
      Q_1d     = basis_tabulation->Np;
      PetscCall(ComputeFieldTabulationP2C(dm, dm_field, 0, basis_tabulation, &interp_1d, &grad_1d));
      PetscCallCeed(ceed, CeedBasisCreateTensorH1(ceed, dim, num_comp, P_1d, Q_1d, interp_1d, grad_1d, q_points_1d, q_weights_1d, basis));

      // Cleanup
      PetscCall(PetscFree(q_points_1d));
      PetscCall(PetscFree(q_weights_1d));
      PetscCall(PetscFree(interp_1d));
      PetscCall(PetscFree(grad_1d));
      PetscCall(PetscTabulationDestroy(&basis_tabulation));
    }
    // Set in container
    PetscCallCeed(ceed, CeedBasisReferenceCopy(*basis, &container_basis));
    PetscCall(PetscObjectContainerCompose((PetscObject)dm, container_name, container_basis, (PetscCtxDestroyFn *)DMPlexCeedBasisDestroy));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create `CeedBasis` for coordinate space of `DMPlex`.

 The `CeedBasis` is stored in the coordinate `DMPlex` object for reuse;
 if the routine is called again with the same arguments, the same `CeedBasis` is returned.

  Collective across MPI processes.

  @param[in]   ceed          `Ceed` context
  @param[in]   dm            `DMPlex` holding mesh
  @param[in]   domain_label  `DMLabel` for `DMPlex` domain
  @param[in]   label_value   Stratum value
  @param[in]   height        Height of `DMPlex` topology
  @param[out]  basis         `CeedBasis` for coordinate space of `DMPlex`
**/
PetscErrorCode DMPlexCeedBasisCoordinateCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, CeedBasis *basis) {
  DM dm_coord = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinateDM(dm, &dm_coord));
  PetscCall(DMPlexCeedBasisCreate(ceed, dm_coord, domain_label, label_value, height, 0, basis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct CeedVectorAndState_ *CeedVectorAndState;
struct CeedVectorAndState_ {
  CeedVector       vector;  // !< `CeedVector` containing copy of data from a `Vec`
  PetscObjectState state;   // !< Object state of the `Vec` when the data was copied
};

/**
  @brief Destroy `CeedVectorAndState` in a `PetscContainer`.

  Not collective across MPI processes.

  @param[in,out]  ctx  `CeedVectorAndState`

  @return An error code: 0 - success, otherwise - failure
**/
static PetscErrorCode CeedVectorAndStateDestroy(void **ctx) {
  CeedVectorAndState vec_and_state = *(CeedVectorAndState *)ctx;

  PetscFunctionBegin;
  PetscCallCeed(CeedVectorReturnCeed(vec_and_state->vector), CeedVectorDestroy(&vec_and_state->vector));
  PetscCall(PetscFree(vec_and_state));
  *ctx = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Get Ceed objects for coordinate field of DM

  @param[in]  ceed         `Ceed` context
  @param[in]  dm           `DMPlex` holding mesh
  @param[in]  domain_label `DMLabel` for `DMPlex` domain
  @param[in]  label_value  Stratum value
  @param[in]  height       Height of `DMPlex` topology
  @param[out] elem_restr   `CeedElemRestriction` for coodinates of `DMPlex`, or `NULL`
  @param[out] basis        `CeedBasis` for coordinate space of `DMPlex`, or `NULL`
  @param[out] vector       `CeedVector` for coordinates of `DMPlex`, or `NULL`
**/
PetscErrorCode DMPlexCeedCoordinateCreateField(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height,
                                               CeedElemRestriction *elem_restr, CeedBasis *basis, CeedVector *vector) {
  CeedElemRestriction elem_restr_ = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMPlexCeedElemRestrictionCoordinateCreate(ceed, dm, domain_label, label_value, height, &elem_restr_));
  if (elem_restr) {
    *elem_restr = NULL;
    PetscCallCeed(ceed, CeedElemRestrictionReferenceCopy(elem_restr_, elem_restr));
  }
  if (basis) PetscCall(DMPlexCeedBasisCoordinateCreate(ceed, dm, domain_label, label_value, height, basis));
  if (vector) {
    DM                 cdm;
    char               container_name[] = "DM Coordinate CeedVectorAndState";
    CeedVectorAndState vec_and_state;
    Vec                X_loc;
    PetscObjectState   X_loc_state;

    PetscCall(DMGetCellCoordinateDM(dm, &cdm));
    if (cdm) {
      PetscCall(DMGetCellCoordinatesLocal(dm, &X_loc));
    } else {
      PetscCall(DMGetCoordinatesLocal(dm, &X_loc));
    }
    PetscCall(VecGetState(X_loc, &X_loc_state));
    PetscCall(PetscObjectContainerQuery((PetscObject)X_loc, container_name, (void **)&vec_and_state));

    if (vec_and_state && vec_and_state->state == X_loc_state) {
      // Copy existing vector
      *vector = NULL;
      PetscCallCeed(ceed, CeedVectorReferenceCopy(vec_and_state->vector, vector));
    } else {
      PetscCall(CeedElemRestrictionCreateVector(elem_restr_, vector, NULL));
      PetscCall(VecCopyPetscToCeed(X_loc, *vector));

      // Set in container
      PetscCall(PetscNew(&vec_and_state));
      PetscCallCeed(ceed, CeedVectorReferenceCopy(*vector, &vec_and_state->vector));
      vec_and_state->state = X_loc_state;
      PetscCall(PetscObjectContainerCompose((PetscObject)X_loc, container_name, vec_and_state, (PetscCtxDestroyFn *)CeedVectorAndStateDestroy));
    }
  }
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create `CeedBasis` for cell to face quadrature space evaluation from `DMPlex`.

  Collective across MPI processes.

  @param[in]   ceed          `Ceed` context
  @param[in]   dm            `DMPlex` holding mesh
  @param[in]   domain_label  `DMLabel` for `DMPlex` domain
  @param[in]   label_value   Stratum value
  @param[in]   face          Index of face
  @param[in]   dm_field      Index of `DMPlex` field
  @param[out]  basis         `CeedBasis` for cell to face evaluation
**/
PetscErrorCode DMPlexCeedBasisCellToFaceCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt face, PetscInt dm_field,
                                               CeedBasis *basis) {
  PetscInt        ds_field = -1, height = 0;
  PetscDS         ds;
  PetscFE         fe;
  PetscQuadrature face_quadrature;

  PetscFunctionBeginUser;
  // Get element information
  PetscCall(DMGetRegionDS(dm, domain_label, NULL, &ds, NULL));
  PetscCall(DMFieldToDSField(dm, domain_label, dm_field, &ds_field));
  PetscCall(PetscDSGetDiscretization(ds, ds_field, (PetscObject *)&fe));
  PetscCall(PetscFEGetFaceQuadrature(fe, &face_quadrature));

  {  // Build libCEED basis
    PetscInt         num_derivatives = 1, num_comp, P, Q = -1;
    PetscTabulation  basis_tabulation;
    CeedScalar      *q_points, *q_weights, *interp, *grad;
    CeedElemTopology elem_topo;

    {  // Verify compatible element topology
      DMPolytopeType cell_type;

      PetscCall(GetGlobalDMPlexPolytopeType(dm, domain_label, label_value, height, &cell_type));
      elem_topo = PolytopeTypePetscToCeed(cell_type);
      PetscCheck(elem_topo, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "DMPlex topology not supported: %s", DMPolytopeTypes[cell_type]);
    }

    // Compute basis tabulation
    PetscCall(GetQuadratureDataP2C(fe, face_quadrature, NULL, NULL, &Q, &q_points, &q_weights));
    PetscCall(PetscFEGetFaceTabulation(fe, num_derivatives, &basis_tabulation));
    num_comp = basis_tabulation->Nc;
    P        = basis_tabulation->Nb / num_comp;
    PetscCall(ComputeFieldTabulationP2C(dm, dm_field, face, basis_tabulation, &interp, &grad));
    PetscCallCeed(ceed, CeedBasisCreateH1(ceed, elem_topo, num_comp, P, Q, interp, grad, q_points, q_weights, basis));

    PetscCall(PetscFree(q_points));
    PetscCall(PetscFree(q_weights));
    PetscCall(PetscFree(interp));
    PetscCall(PetscFree(grad));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create `CeedBasis` for cell to face quadrature space evaluation on coordinate space of `DMPlex`.

  Collective across MPI processes.

  @param[in]   ceed          `Ceed` context
  @param[in]   dm            `DMPlex` holding mesh
  @param[in]   domain_label  `DMLabel` for `DMPlex` domain
  @param[in]   label_value   Stratum value
  @param[in]   face          Index of face
  @param[out]  basis         `CeedBasis` for cell to face evaluation on coordinate space of `DMPlex`
**/
PetscErrorCode DMPlexCeedBasisCellToFaceCoordinateCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt face,
                                                         CeedBasis *basis) {
  DM dm_coord;

  PetscFunctionBeginUser;
  PetscCall(DMGetCellCoordinateDM(dm, &dm_coord));
  if (!dm_coord) {
    PetscCall(DMGetCoordinateDM(dm, &dm_coord));
  }
  PetscCall(DMPlexCeedBasisCellToFaceCreate(ceed, dm_coord, domain_label, label_value, face, 0, basis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Setup `DM` with FE space of appropriate degree

  Must be followed by `DMSetupByOrderEnd_FEM`

  @param[in]   setup_faces     Flag to setup face geometry
  @param[in]   setup_coords    Flag to setup coordinate spaces
  @param[in]   degree          Polynomial orders of field
  @param[in]   coord_order     Polynomial order of coordinate basis, or `PETSC_DECIDE` for default
  @param[in]   q_extra         Additional quadrature order
  @param[in]   num_fields      Number of fields in solution vector
  @param[in]   field_sizes     Array of number of components for each field
  @param[out]  dm              `DM` to setup

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode DMSetupByOrderBegin_FEM(PetscBool setup_faces, PetscBool setup_coords, PetscInt degree, PetscInt coord_order, PetscInt q_extra,
                                       PetscInt num_fields, const PetscInt *field_sizes, DM dm) {
  PetscInt  dim, q_order = degree + q_extra;
  PetscBool is_simplex = PETSC_TRUE;
  PetscFE   fe;
  MPI_Comm  comm = PetscObjectComm((PetscObject)dm);

  PetscFunctionBeginUser;
  PetscCall(DMPlexIsSimplex(dm, &is_simplex));

  // Setup DM
  PetscCall(DMGetDimension(dm, &dim));
  for (PetscInt i = 0; i < num_fields; i++) {
    PetscFE  fe_face;
    PetscInt q_order = degree + q_extra;

    PetscCall(PetscFECreateLagrange(comm, dim, field_sizes[i], is_simplex, degree, q_order, &fe));
    if (setup_faces) PetscCall(PetscFEGetHeightSubspace(fe, 1, &fe_face));
    PetscCall(DMAddField(dm, NULL, (PetscObject)fe));
    PetscCall(PetscFEDestroy(&fe));
  }
  PetscCall(DMCreateDS(dm));

  // Project coordinates to enrich quadrature space
  if (setup_coords) {
    DM             dm_coord;
    PetscDS        ds_coord;
    PetscFE        fe_coord_current, fe_coord_new, fe_coord_face_new;
    PetscDualSpace fe_coord_dual_space;
    PetscInt       fe_coord_order, num_comp_coord;

    PetscCall(DMGetCoordinateDM(dm, &dm_coord));
    PetscCall(DMGetCoordinateDim(dm, &num_comp_coord));
    PetscCall(DMGetRegionDS(dm_coord, NULL, NULL, &ds_coord, NULL));
    PetscCall(PetscDSGetDiscretization(ds_coord, 0, (PetscObject *)&fe_coord_current));
    PetscCall(PetscFEGetDualSpace(fe_coord_current, &fe_coord_dual_space));
    PetscCall(PetscDualSpaceGetOrder(fe_coord_dual_space, &fe_coord_order));

    // Create FE for coordinates
    if (coord_order != PETSC_DECIDE) fe_coord_order = coord_order;
    PetscCall(PetscFECreateLagrange(comm, dim, num_comp_coord, is_simplex, fe_coord_order, q_order, &fe_coord_new));
    if (setup_faces) PetscCall(PetscFEGetHeightSubspace(fe_coord_new, 1, &fe_coord_face_new));
    PetscCall(DMSetCoordinateDisc(dm, fe_coord_new, PETSC_FALSE, PETSC_TRUE));
    PetscCall(PetscFEDestroy(&fe_coord_new));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Finish setting up `DM`

  Must be called after `DMSetupByOrderBegin_FEM` and all strong BCs have be declared via `DMAddBoundaries`.

  @param[in]   setup_coords    Flag to setup coordinate spaces
  @param[out]  dm              `DM` to setup

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode DMSetupByOrderEnd_FEM(PetscBool setup_coords, DM dm) {
  PetscBool is_simplex;

  PetscFunctionBeginUser;
  PetscCall(DMPlexIsSimplex(dm, &is_simplex));
  // Set tensor permutation if needed
  if (!is_simplex) {
    PetscCall(DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL));
    if (setup_coords) {
      DM dm_coord;

      PetscCall(DMGetCoordinateDM(dm, &dm_coord));
      PetscCall(DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL));
    }
  }
  PetscCall(DMLocalizeCoordinates(dm));  // Must localize after tensor closure setting
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Setup `DM` with FE space of appropriate degree with no boundary conditions

  Calls `DMSetupByOrderBegin_FEM` and `DMSetupByOrderEnd_FEM` successively

  @param[in]   setup_faces     Flag to setup face geometry
  @param[in]   setup_coords    Flag to setup coordinate spaces
  @param[in]   degree          Polynomial orders of field
  @param[in]   coord_order     Polynomial order of coordinate basis, or `PETSC_DECIDE` for default
  @param[in]   q_extra         Additional quadrature order
  @param[in]   num_fields      Number of fields in solution vector
  @param[in]   field_sizes     Array of number of components for each field
  @param[out]  dm              `DM` to setup

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode DMSetupByOrder_FEM(PetscBool setup_faces, PetscBool setup_coords, PetscInt degree, PetscInt coord_order, PetscInt q_extra,
                                  PetscInt num_fields, const PetscInt *field_sizes, DM dm) {
  PetscFunctionBeginUser;
  PetscCall(DMSetupByOrderBegin_FEM(setup_faces, setup_coords, degree, coord_order, q_extra, num_fields, field_sizes, dm));
  PetscCall(DMSetupByOrderEnd_FEM(setup_coords, dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Get the points in a label stratum which have requested height

  Example, from the "Face Sets" label (which contains face, edge, and vertex points), return points that correspond to face points (height = 1) and have a label value of 3:
  ```
  PetscCall(DMGetStratumISAtHeight(dm, "Face Sets", 3, 1, &is_face_points));
  ```

  @param[in]  dm     DM which has the label
  @param[in]  name   Name fo the label
  @param[in]  value  Label value of points to return
  @param[in]  height Height of points to return
  @param[out] points `IS` of points, or `NULL` is there are no points in that stratum at height
**/
static PetscErrorCode DMGetStratumISAtHeight(DM dm, const char name[], PetscInt value, PetscInt height, IS *points) {
  PetscInt depth;
  DMLabel  depth_label;
  IS       label_is, depth_is;

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMPlexGetDepthLabel(dm, &depth_label));
  PetscCall(DMLabelGetStratumIS(depth_label, depth - height, &depth_is));
  PetscCall(DMGetStratumIS(dm, name, value, &label_is));
  if (label_is == NULL || depth_is == NULL) *points = NULL;
  else PetscCall(ISIntersect(depth_is, label_is, points));
  PetscCall(ISDestroy(&depth_is));
  PetscCall(ISDestroy(&label_is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Create a label with separate values for the `FE` orientations for all cells with this material for the `DMPlex` face.

  Collective across MPI processes.

  @param[in,out]  dm                `DMPlex` holding mesh
  @param[in]      dm_face           Face number on `DMPlex`
  @param[out]     face_label_name   Label name for newly created label, or `NULL` if no elements match the `material` and `dm_face`
**/
PetscErrorCode DMPlexCreateFaceLabel(DM dm, PetscInt dm_face, char **face_label_name) {
  PetscInt        num_face_points = 0, face_height = 1;
  const PetscInt *face_points;
  IS              is_face_points;
  DMLabel         face_label = NULL;
  MPI_Comm        comm       = PetscObjectComm((PetscObject)dm);

  PetscFunctionBeginUser;
  {  // Create label name and label
    PetscSizeT label_name_len = 64;
    PetscBool  has_label_already;

    // Face label
    PetscCall(PetscCalloc1(label_name_len, face_label_name));
    PetscCall(PetscSNPrintf(*face_label_name, label_name_len, "Orientation for DM face %" PetscInt_FMT, dm_face));
    PetscCall(DMHasLabel(dm, *face_label_name, &has_label_already));
    if (has_label_already) PetscFunctionReturn(PETSC_SUCCESS);

    PetscCall(DMCreateLabel(dm, *face_label_name));
    PetscCall(DMGetLabel(dm, *face_label_name, &face_label));
  }

  PetscCall(DMGetStratumISAtHeight(dm, "Face Sets", dm_face, face_height, &is_face_points));
  if (is_face_points) {
    PetscCall(ISGetLocalSize(is_face_points, &num_face_points));
    PetscCall(ISGetIndices(is_face_points, &face_points));
  }

  for (PetscInt face_index = 0; face_index < num_face_points; face_index++) {
    const PetscInt *face_support, *cell_cone;
    PetscInt        face_support_size = 0, cell_cone_size = 0, fe_face_on_cell = -1;
    PetscInt        face_point = face_points[face_index];

    // Get cell supporting face
    PetscCall(DMPlexGetSupport(dm, face_point, &face_support));
    PetscCall(DMPlexGetSupportSize(dm, face_point, &face_support_size));
    PetscCheck(face_support_size == 1, comm, PETSC_ERR_SUP, "Expected one cell in support of exterior face, but got %" PetscInt_FMT " cells",
               face_support_size);
    PetscInt cell_point = face_support[0];

    // Find face number
    PetscCall(DMPlexGetCone(dm, cell_point, &cell_cone));
    PetscCall(DMPlexGetConeSize(dm, cell_point, &cell_cone_size));
    for (PetscInt i = 0; i < cell_cone_size; i++) {
      if (cell_cone[i] == face_point) fe_face_on_cell = i;
    }
    PetscCheck(fe_face_on_cell != -1, comm, PETSC_ERR_SUP, "Could not find face %" PetscInt_FMT " in cone of its support", face_point);

    // Add to label
    PetscCall(DMLabelSetValue(face_label, face_point, fe_face_on_cell));
  }
  PetscCall(DMPlexLabelAddFaceCells(dm, face_label));
  PetscCall(ISDestroy(&is_face_points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
   @brief Creates an array of all the values set for a `DMLabel`

   Because `DMLabelGetValueIS` returns label values locally, not globally.

   Collective across MPI processes.

   @param[in]  dm          `DM` with the label
   @param[in]  label       `DMLabel` to get values of
   @param[out] num_values  Total number of values
   @param[out] value_array Array of label values, must be freed by user
**/
PetscErrorCode DMLabelCreateGlobalValueArray(DM dm, DMLabel label, PetscInt *num_values, PetscInt **value_array) {
  PetscInt       num_values_local, minmax_values[2], minmax_values_loc[2];
  PetscInt      *values_local = NULL;
  MPI_Comm       comm         = PetscObjectComm((PetscObject)dm);
  PetscSegBuffer seg;
  PetscCount     seg_size;

  PetscFunctionBeginUser;
  {  // Extract array of local DMLabel values so it can be sorted
    IS              is_values;
    const PetscInt *is_values_local = NULL;

    PetscCall(DMLabelGetValueIS(label, &is_values));
    PetscCall(ISGetLocalSize(is_values, &num_values_local));

    PetscCall(ISGetIndices(is_values, &is_values_local));
    PetscCall(PetscMalloc1(num_values_local, &values_local));
    PetscCall(PetscArraycpy(values_local, is_values_local, num_values_local));
    PetscCall(ISRestoreIndices(is_values, &is_values_local));
    PetscCall(ISDestroy(&is_values));
  }

  if (num_values_local) {
    PetscCall(PetscSortInt(num_values_local, values_local));
    minmax_values_loc[0] = values_local[0];
    minmax_values_loc[1] = values_local[num_values_local - 1];
  } else {
    // Rank has no set label values, therefore set local min/max to have no effect on global min/max
    minmax_values_loc[0] = PETSC_INT_MAX;
    minmax_values_loc[1] = PETSC_INT_MIN;
  }
  PetscCall(PetscGlobalMinMaxInt(comm, minmax_values_loc, minmax_values));

  PetscCall(PetscSegBufferCreate(sizeof(PetscInt), 16, &seg));
  for (PetscInt value = minmax_values[0]; value <= minmax_values[1]; value++) {
    PetscInt location_local = -1, location_global = -1;

    PetscCall(PetscFindInt(value, num_values_local, values_local, &location_local));
    PetscCallMPI(MPIU_Allreduce(&location_local, &location_global, 1, MPIU_INT, MPI_MAX, comm));
    if (location_global < 0) continue;
    else {
      PetscInt *seg_slot;

      PetscCall(PetscSegBufferGetInts(seg, 1, &seg_slot));
      *seg_slot = value;
    }
  }
  PetscCall(PetscSegBufferGetSize(seg, &seg_size));
  *num_values = (PetscInt)seg_size;
  PetscCall(PetscSegBufferExtractAlloc(seg, value_array));
  PetscCall(PetscSegBufferDestroy(&seg));
  PetscCall(PetscFree(values_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Get the number of components for `dm`'s coordinate field

  @param[in]  dm       DM
  @param[out] num_comp Number of components for the coordinate field
**/
PetscErrorCode DMGetCoordinateNumComps(DM dm, PetscInt *num_comp) {
  DM           dm_coord;
  PetscSection section_coord;
  PetscInt     field = 0;  // Default field has the coordinates

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinateDM(dm, &dm_coord));
  PetscCall(DMGetLocalSection(dm_coord, &section_coord));
  PetscCall(PetscSectionGetFieldComponents(section_coord, field, num_comp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Get the number of components for `dm`'s field

  @param[in]  dm       DM
  @param[in]  field    The field ID to get the number of components for
  @param[out] num_comp Number of components for the coordinate field
**/
PetscErrorCode DMGetFieldNumComps(DM dm, PetscInt field, PetscInt *num_comp) {
  PetscSection section;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionGetFieldComponents(section, field, num_comp));
  PetscFunctionReturn(PETSC_SUCCESS);
}
