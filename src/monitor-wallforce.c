// SPDX-FileCopyrightText: Copyright (c) 2017-2026, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Monitor wall forces

#include <honee.h>
#include <navierstokes.h>
#include "../qfunctions/newtonian_state.h"

// Surface forces function setup
static PetscErrorCode ComputeSurfaceForces(DM dm, Vec G_loc, PetscInt num_walls, const PetscInt walls[], PetscScalar *reaction_force) {
  DMLabel            face_label;
  const PetscScalar *g_array;
  PetscInt           dim  = 3;
  MPI_Comm           comm = PetscObjectComm((PetscObject)dm);
  PetscSection       section;

  PetscFunctionBeginUser;
  PetscCall(DMGetLabel(dm, "Face Sets", &face_label));
  PetscCall(VecGetArrayRead(G_loc, &g_array));
  for (PetscInt w = 0; w < num_walls; w++) {
    const PetscInt wall = walls[w], *points;
    IS             wall_is;
    PetscInt       num_points, num_comp = 0;

    PetscCall(DMLabelGetStratumIS(face_label, wall, &wall_is));
    if (!wall_is) continue;  // No wall points on this process, skip

    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(PetscSectionGetFieldComponents(section, 0, &num_comp));
    PetscCall(ISGetSize(wall_is, &num_points));
    PetscCall(ISGetIndices(wall_is, &points));
    for (PetscInt i = 0; i < num_points; i++) {
      const PetscInt           p = points[i];
      const StateConservative *r;
      PetscInt                 dof;

      PetscCall(DMPlexPointLocalRead(dm, p, g_array, &r));
      PetscCall(PetscSectionGetDof(section, p, &dof));
      for (PetscInt node = 0; node < dof / num_comp; node++) {
        for (PetscInt j = 0; j < dim; j++) {
          reaction_force[w * dim + j] -= r[node].momentum[j];
        }
      }
    }
    PetscCall(ISRestoreIndices(wall_is, &points));
    PetscCall(ISDestroy(&wall_is));
  }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, reaction_force, dim * num_walls, MPIU_SCALAR, MPI_SUM, comm));
  PetscCall(VecRestoreArrayRead(G_loc, &g_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// CSV Monitor
PetscErrorCode TSMonitor_WallForce(TS ts, PetscInt step_no, PetscReal time, Vec Q, void *ctx) {
  Honee             honee = ctx;
  Vec               G_loc;
  PetscInt          num_wall = honee->app_ctx->wall_forces.num_wall, dim = 3;
  const PetscInt   *walls  = honee->app_ctx->wall_forces.walls;
  PetscViewer       viewer = honee->app_ctx->wall_forces.viewer;
  PetscViewerFormat format = honee->app_ctx->wall_forces.viewer_format;
  PetscScalar      *reaction_force;
  PetscBool         is_ascii;

  PetscFunctionBeginUser;
  if (!viewer) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetNamedLocalVector(honee->dm, "ResidualLocal", &G_loc));
  PetscCall(PetscCalloc1(num_wall * dim, &reaction_force));
  if (num_wall <= 0)
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts), "WARNING: Wall forces requested, but no boundaries are labeled as wall."));
  PetscCall(ComputeSurfaceForces(honee->dm, G_loc, num_wall, walls, reaction_force));
  PetscCall(DMRestoreNamedLocalVector(honee->dm, "ResidualLocal", &G_loc));

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &is_ascii));

  if (is_ascii) {
    if (format == PETSC_VIEWER_ASCII_CSV && !honee->app_ctx->wall_forces.header_written) {
      PetscCall(HoneeWriteCSVHeader(viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Step,Time,Wall,ForceX,ForceY,ForceZ\n"));
      honee->app_ctx->wall_forces.header_written = PETSC_TRUE;
    }
    for (PetscInt w = 0; w < num_wall; w++) {
      PetscInt wall = walls[w];
      if (format == PETSC_VIEWER_ASCII_CSV) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT ",%.17e,%" PetscInt_FMT ",%.17e,%.17e,%.17e\n", step_no, time, wall,
                                         (double)reaction_force[w * dim + 0], (double)reaction_force[w * dim + 1],
                                         (double)reaction_force[w * dim + 2]));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "Wall %" PetscInt_FMT " Forces: Force_x = %12g, Force_y = %12g, Force_z = %12g\n", wall,
                                         reaction_force[w * dim + 0], reaction_force[w * dim + 1], reaction_force[w * dim + 2]));
      }
    }
  }
  PetscCall(PetscFree(reaction_force));
  PetscFunctionReturn(PETSC_SUCCESS);
}
