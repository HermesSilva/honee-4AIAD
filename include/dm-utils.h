// SPDX-FileCopyrightText: Copyright (c) 2017-2025, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#pragma once

#include <ceed.h>
#include <petscdm.h>

PetscErrorCode DMPlexCeedElemRestrictionCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt dm_field,
                                               CeedElemRestriction *restriction);
PetscErrorCode DMPlexCeedElemRestrictionCoordinateCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height,
                                                         CeedElemRestriction *restriction);
PetscErrorCode DMPlexCeedElemRestrictionQDataCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height,
                                                    PetscInt q_data_size, CeedElemRestriction *restriction);
PetscErrorCode DMPlexCeedElemRestrictionCollocatedCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height,
                                                         PetscInt q_data_size, CeedElemRestriction *restriction);

PetscErrorCode DMPlexCeedBasisCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt dm_field,
                                     CeedBasis *basis);
PetscErrorCode DMPlexCeedBasisCoordinateCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, CeedBasis *basis);
PetscErrorCode DMPlexCeedCoordinateCreateField(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height,
                                               CeedElemRestriction *elem_restr, CeedBasis *basis, CeedVector *vector);
PetscErrorCode DMPlexCeedBasisCellToFaceCoordinateCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt face,
                                                         CeedBasis *basis);
PetscErrorCode DMPlexCeedBasisCellToFaceCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt face, PetscInt dm_field,
                                               CeedBasis *basis);
PetscErrorCode DMPlexCreateFaceLabel(DM dm, PetscInt dm_face, char **face_label_name);

PetscErrorCode DMGetCoordinateNumComps(DM dm, PetscInt *num_comp_x);
PetscErrorCode DMGetFieldNumComps(DM dm, PetscInt field, PetscInt *num_comp);

PetscErrorCode DMSetupByOrderBegin_FEM(PetscBool setup_faces, PetscBool setup_coords, PetscInt degree, PetscInt coord_order, PetscInt q_extra,
                                       PetscInt num_fields, const PetscInt *field_sizes, DM dm);
PetscErrorCode DMSetupByOrderEnd_FEM(PetscBool setup_coords, DM dm);
PetscErrorCode DMSetupByOrder_FEM(PetscBool setup_faces, PetscBool setup_coords, PetscInt degree, PetscInt coord_order, PetscInt q_extra,
                                  PetscInt num_fields, const PetscInt *field_sizes, DM dm);
