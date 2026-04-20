// SPDX-FileCopyrightText: Copyright (c) 2017-2026, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Monitor wall forces

#include <honee.h>
#include <navierstokes.h>
#include "../qfunctions/newtonian_state.h"

typedef struct {
  PetscInt     num_walls;
  PetscInt    *walls;
  PetscScalar *reaction_forces;
} *HoneeWallForce;

static PetscErrorCode HoneeWallForceDestroy(HoneeWallForce *wallforce) {
  HoneeWallForce wallforce_ = *wallforce;

  PetscFunctionBeginUser;
  PetscCall(PetscFree(wallforce_->reaction_forces));
  PetscCall(PetscFree(wallforce_->walls));
  PetscCall(PetscFree(wallforce_));
  *wallforce = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define CALCWALLFORCE_KEY "Honee Wall Force Calculation"

/**
  @brief Setup wall force calculation in HONEE

  @param[in] honee     `Honee`
  @param[in] num_walls Number of walls to calculate the forces on
  @param[in] walls     Label values in "Face Sets" to calculate the forces on
**/
PetscErrorCode HoneeWallForceSetup(Honee honee, PetscInt num_walls, const PetscInt walls[]) {
  HoneeWallForce wallforce;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&wallforce));
  wallforce->num_walls = num_walls;
  PetscCall(PetscMalloc1(num_walls, &wallforce->walls));
  PetscCall(PetscArraycpy(wallforce->walls, walls, num_walls));
  PetscCall(PetscMalloc1(num_walls * 3, &wallforce->reaction_forces));
  PetscCall(HoneeSetContainer(honee, CALCWALLFORCE_KEY, wallforce, (PetscCtxDestroyFn *)HoneeWallForceDestroy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Compute the forces on the given walls

  This uses a consistent residual-based method for computing the forces.
  In short, the momentum residual calculated at nodes on the wall directly corresponds to the force being applied to that wall.
  Summing these residuals together results in the total force (e.g. pressure and viscous forces) on the surface.

  @param[in]  dm             `DM`
  @param[in]  G_loc          Local residual vector
  @param[in]  num_walls      Number of walls to calculate the forces on
  @param[in]  walls          Label values in "Face Sets" to calculate the forces on
  @param[out] reaction_force Array of the reaction forces applied to the walls in `[num_walls][direction]` indexing
**/
static PetscErrorCode ComputeSurfaceForces(DM dm, Vec G_loc, PetscInt num_walls, const PetscInt walls[], PetscScalar *reaction_force) {
  DMLabel            face_label;
  const PetscScalar *g_array;
  PetscInt           dim  = 3;
  MPI_Comm           comm = PetscObjectComm((PetscObject)dm);
  PetscSection       section;

  PetscFunctionBeginUser;
  PetscCall(DMGetLabel(dm, "Face Sets", &face_label));
  PetscCall(PetscArrayzero(reaction_force, num_walls * dim));
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

typedef struct {
  PetscInt  tab_level;
  PetscBool is_header_written;
} *MonitorWallForceCtx;

// @brief Setup monitoring for wall forces, passed to `TSMonitorSetFromOptions()`
PetscErrorCode SetupMontiorWallForce(TS ts, PetscViewerAndFormat *ctx) {
  Honee               honee;
  MonitorWallForceCtx monitor_ctx;
  PetscInt            tab_level;
  PetscBool           is_ascii;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectTypeCompare((PetscObject)ctx->viewer, PETSCVIEWERASCII, &is_ascii));
  PetscCheck(is_ascii, PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "Only supports ASCII viewers");
  PetscCall(TSGetApplicationContext(ts, &honee));

  PetscCall(PetscNew(&monitor_ctx));
  PetscCall(PetscObjectGetTabLevel((PetscObject)ts, &tab_level));
  monitor_ctx->tab_level = tab_level + 1;
  ctx->data              = monitor_ctx;
  ctx->data_destroy      = PetscCtxDestroyDefault;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Monitor for wall forces
PetscErrorCode TSMonitor_WallForce(TS ts, PetscInt step_no, PetscReal time, Vec Q, PetscViewerAndFormat *ctx) {
  Honee               honee;
  HoneeWallForce      wallforce;
  MonitorWallForceCtx monitor_ctx = ctx->data;
  Vec                 G_loc;
  PetscInt            dim = 3;
  PetscBool           is_ascii;

  PetscFunctionBeginUser;
  if (!ctx->viewer) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSGetApplicationContext(ts, &honee));
  PetscCall(DMGetNamedLocalVector(honee->dm, "ResidualLocal", &G_loc));
  PetscCall(HoneeGetContainer(honee, CALCWALLFORCE_KEY, &wallforce));
  if (wallforce->num_walls <= 0)
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts), "WARNING: Wall forces requested, but no boundaries are labeled as wall."));
  PetscCall(ComputeSurfaceForces(honee->dm, G_loc, wallforce->num_walls, wallforce->walls, wallforce->reaction_forces));
  PetscCall(DMRestoreNamedLocalVector(honee->dm, "ResidualLocal", &G_loc));

  PetscCall(PetscObjectTypeCompare((PetscObject)ctx->viewer, PETSCVIEWERASCII, &is_ascii));
  PetscCheck(is_ascii, PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "Only supports ASCII viewers");

  if (ctx->format == PETSC_VIEWER_ASCII_CSV && !monitor_ctx->is_header_written) {
    PetscCall(HoneeWriteCSVHeader(ctx->viewer));
    PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "Step,Time,Wall,ForceX,ForceY,ForceZ\n"));
    monitor_ctx->is_header_written = PETSC_TRUE;
  }
  for (PetscInt w = 0; w < wallforce->num_walls; w++) {
    PetscInt wall = wallforce->walls[w];
    if (ctx->format == PETSC_VIEWER_ASCII_CSV) {
      PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "%" PetscInt_FMT ",%.17e,%" PetscInt_FMT ",%.17e,%.17e,%.17e\n", step_no, time, wall,
                                       (double)wallforce->reaction_forces[w * dim + 0], (double)wallforce->reaction_forces[w * dim + 1],
                                       (double)wallforce->reaction_forces[w * dim + 2]));
    } else {
      PetscCall(PetscViewerASCIIAddTab(ctx->viewer, monitor_ctx->tab_level));
      PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "Wall %" PetscInt_FMT " Forces: Force_x = %12g, Force_y = %12g, Force_z = %12g\n", wall,
                                       wallforce->reaction_forces[w * dim + 0], wallforce->reaction_forces[w * dim + 1],
                                       wallforce->reaction_forces[w * dim + 2]));
      PetscCall(PetscViewerASCIISubtractTab(ctx->viewer, monitor_ctx->tab_level));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
