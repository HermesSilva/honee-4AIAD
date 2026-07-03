// SPDX-FileCopyrightText: Copyright (c) 2017-2025, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include <honee.h>

#include <petscdmplex.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static PetscErrorCode GetYNodeLocs(const MPI_Comm comm, const char path[PETSC_MAX_PATH_LEN], PetscReal **pynodes, PetscInt *nynodes) {
  int            ndims, dims[2] = {0.};
  FILE          *fp;
  const PetscInt char_array_len = 512;
  char           line[char_array_len];
  PetscReal     *node_locs;

  PetscFunctionBeginUser;
  PetscCall(PetscFOpen(comm, path, "r", &fp));
  PetscCall(PetscSynchronizedFGets(comm, fp, char_array_len, line));
  {
    char **array;

    PetscCall(PetscStrToArray(line, ' ', &ndims, &array));
    for (PetscInt i = 0; i < ndims; i++) dims[i] = atoi(array[i]);
    PetscCall(PetscStrToArrayDestroy(ndims, array));
  }
  if (ndims < 2) dims[1] = 1;  // Assume 1 column of data is not otherwise specified
  *nynodes = dims[0];
  PetscCall(PetscMalloc1(*nynodes, &node_locs));

  for (PetscInt i = 0; i < dims[0]; i++) {
    char **array;

    PetscCall(PetscSynchronizedFGets(comm, fp, char_array_len, line));
    PetscCall(PetscStrToArray(line, ' ', &ndims, &array));
    PetscCheck(ndims == dims[1], comm, PETSC_ERR_FILE_UNEXPECTED,
               "Line %" PetscInt_FMT " of %s does not contain correct number of columns (%d instead of %d)", i, path, ndims, dims[1]);

    node_locs[i] = (PetscReal)atof(array[0]);
    PetscCall(PetscStrToArrayDestroy(ndims, array));
  }
  PetscCall(PetscFClose(comm, fp));
  *pynodes = node_locs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* @brief Modify the domain and mesh for blasius
 *
 * Modifies mesh such that `N` elements are within `refine_height` with a geometric growth ratio of `growth`. Excess elements are then distributed
 * linearly in logspace to the top surface.
 *
 * The top surface is also angled downwards, so that it may be used as an outflow.
 * It's angle is controlled by `top_angle` (in units of degrees).
 *
 * If `node_locs` is not NULL, then the nodes will be placed at `node_locs` locations.
 * If it is NULL, then the modified coordinate values will be set in the array, along with `num_node_locs`.
 */
static PetscErrorCode HoneeMeshTransform_PlateMesh(MPI_Comm comm, DM dm, PetscReal growth, PetscInt N, PetscReal refine_height, PetscReal top_angle,
                                                   PetscReal *node_locs[], PetscInt *num_node_locs) {
  PetscInt     narr, ncoords, dim;
  PetscReal    domain_min[3], domain_max[3], domain_size[3];
  PetscScalar *arr_coords;
  Vec          vec_coords;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscReal angle_coeff = tan(top_angle * (M_PI / 180));
  // Get domain boundary information
  PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
  for (PetscInt i = 0; i < 3; i++) domain_size[i] = domain_max[i] - domain_min[i];

  // Get coords array from DM
  PetscCall(DMGetCoordinatesLocal(dm, &vec_coords));
  PetscCall(VecGetLocalSize(vec_coords, &narr));
  PetscCall(VecGetArray(vec_coords, &arr_coords));

  PetscScalar(*coords)[dim] = (PetscScalar(*)[dim])arr_coords;
  ncoords                   = narr / dim;

  // Get mesh information
  PetscInt nmax = 3, faces[3];
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &nmax, NULL));
  // Get element size of the box mesh, for indexing each node
  const PetscReal dybox = domain_size[1] / faces[1];

  if (!*node_locs) {
    // Calculate the first element height
    PetscReal dy1 = refine_height * (growth - 1) / (pow(growth, N) - 1);

    // Calculate log of sizing outside BL
    PetscReal logdy = (log(domain_max[1]) - log(refine_height)) / (faces[1] - N);

    *num_node_locs = faces[1] + 1;
    PetscReal *temp_node_locs;
    PetscCall(PetscMalloc1(*num_node_locs, &temp_node_locs));

    for (PetscInt i = 0; i < ncoords; i++) {
      PetscInt y_box_index = round(coords[i][1] / dybox);
      if (y_box_index <= N) {
        coords[i][1] =
            (1 - (coords[i][0] - domain_min[0]) * angle_coeff / domain_max[1]) * dy1 * (pow(growth, coords[i][1] / dybox) - 1) / (growth - 1);
      } else {
        PetscInt j   = y_box_index - N;
        coords[i][1] = (1 - (coords[i][0] - domain_min[0]) * angle_coeff / domain_max[1]) * exp(log(refine_height) + logdy * j);
      }
      if (coords[i][0] == domain_min[0] && coords[i][2] == domain_min[2]) temp_node_locs[y_box_index] = coords[i][1];
    }

    *node_locs = temp_node_locs;
  } else {
    PetscCheck(*num_node_locs >= faces[1] + 1, comm, PETSC_ERR_FILE_UNEXPECTED,
               "The y_node_locs_path has too few locations; There are %" PetscInt_FMT " + 1 nodes, but only %" PetscInt_FMT " locations given",
               faces[1] + 1, *num_node_locs);
    if (*num_node_locs > faces[1] + 1) {
      PetscCall(PetscPrintf(comm,
                            "WARNING: y_node_locs_path has more locations (%" PetscInt_FMT ") "
                            "than the mesh has nodes (%" PetscInt_FMT "). This maybe unintended.\n",
                            *num_node_locs, faces[1] + 1));
    }
    PetscScalar max_y = (*node_locs)[faces[1]];

    for (PetscInt i = 0; i < ncoords; i++) {
      // Determine which y-node we're at
      PetscInt y_box_index = round(coords[i][1] / dybox);
      coords[i][1]         = (1 - (coords[i][0] - domain_min[0]) * angle_coeff / max_y) * (*node_locs)[y_box_index];
    }
  }

  PetscCall(VecRestoreArray(vec_coords, &arr_coords));
  PetscCall(DMSetCoordinatesLocal(dm, vec_coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Transform the mesh based on pass options

  Use the option `-mesh_transform` to determine the type of mesh transformation to apply

  @param dm `DM` of mesh to be transformed
**/
PetscErrorCode HoneeMeshTransformFromOptions(DM dm) {
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  PetscFunctionBeginUser;
  PetscOptionsBegin(comm, NULL, "Options for HONEE Mesh Transformation", NULL);
  MeshTransform mesh_transform_type = MESH_TRANSFORM_NONE;
  PetscCall(PetscOptionsDeprecated("-mesh_transform", "-meshtransform_type", "HONEE 0.0", NULL));
  PetscCall(PetscOptionsEnum("-meshtransform_type", "Mesh transform to perform", NULL, MeshTransforms, (PetscEnum)mesh_transform_type,
                             (PetscEnum *)&mesh_transform_type, NULL));

  switch (mesh_transform_type) {
    case MESH_TRANSFORM_PLATEMESH: {
      PetscReal mesh_refine_height                   = 5.9e-4;  // m
      PetscReal mesh_growth                          = 1.08;    // [-]
      PetscInt  mesh_Ndelta                          = 45;      // [-]
      PetscReal mesh_top_angle                       = 5;       // degrees
      char      mesh_ynodes_path[PETSC_MAX_PATH_LEN] = "";

      PetscCall(PetscOptionsDeprecated("-platemesh_Ndelta", "-meshtransform_platemesh_Ndelta", "HONEE 0.0", NULL));
      PetscCall(PetscOptionsDeprecated("-platemesh_refine_height", "-meshtransform_platemesh_refine_height", "HONEE 0.0", NULL));
      PetscCall(PetscOptionsDeprecated("-platemesh_growth", "-meshtransform_platemesh_growth", "HONEE 0.0", NULL));
      PetscCall(PetscOptionsDeprecated("-platemesh_top_angle", "-meshtransform_platemesh_top_angle", "HONEE 0.0", NULL));
      PetscCall(PetscOptionsDeprecated("-platemesh_y_node_locs_path", "-meshtransform_platemesh_y_node_locs_path", "HONEE 0.0", NULL));

      PetscCall(PetscOptionsBoundedInt("-meshtransform_platemesh_Ndelta", "Velocity at boundary layer edge", NULL, mesh_Ndelta, &mesh_Ndelta, NULL,
                                       1));
      PetscCall(PetscOptionsScalar("-meshtransform_platemesh_refine_height", "Height of boundary layer mesh refinement", NULL, mesh_refine_height,
                                   &mesh_refine_height, NULL));
      PetscCall(PetscOptionsScalar("-meshtransform_platemesh_growth", "Geometric growth rate of boundary layer mesh", NULL, mesh_growth, &mesh_growth,
                                   NULL));
      PetscCall(PetscOptionsScalar("-meshtransform_platemesh_top_angle", "Geometric top_angle rate of boundary layer mesh", NULL, mesh_top_angle,
                                   &mesh_top_angle, NULL));
      PetscCall(PetscOptionsString("-meshtransform_platemesh_y_node_locs_path",
                                   "Path to file with y node locations. "
                                   "If empty, will use the algorithmic mesh warping.",
                                   NULL, mesh_ynodes_path, mesh_ynodes_path, sizeof(mesh_ynodes_path), NULL));
      PetscReal *mesh_ynodes  = NULL;
      PetscInt   mesh_nynodes = 0;
      if (strcmp(mesh_ynodes_path, "")) PetscCall(GetYNodeLocs(comm, mesh_ynodes_path, &mesh_ynodes, &mesh_nynodes));
      PetscCall(HoneeMeshTransform_PlateMesh(comm, dm, mesh_growth, mesh_Ndelta, mesh_refine_height, mesh_top_angle, &mesh_ynodes, &mesh_nynodes));
      PetscCall(PetscFree(mesh_ynodes));
    } break;
    case MESH_TRANSFORM_NONE:
      break;
  }
  PetscOptionsEnd();

  {  // Scale coordinates based on DM length scale
    // We want to make sure to do the scaling *after* the mesh transformation, so units in the transform are physical
    Vec       coords;
    PetscReal meter;
    PetscCall(DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &meter));
    PetscCall(DMGetCoordinatesLocal(dm, &coords));
    PetscCall(VecScale(coords, meter));
    PetscCall(DMSetCoordinatesLocal(dm, coords));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
