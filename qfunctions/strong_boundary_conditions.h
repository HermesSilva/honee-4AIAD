// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#include <ceed/types.h>

#include "setupgeo_helpers.h"

CEED_QFUNCTION(SetupStrongBC)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*coords)[CEED_Q_VLA]       = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*dxdX_q)[3][CEED_Q_VLA]    = (const CeedScalar(*)[3][CEED_Q_VLA])in[1];
  const CeedScalar(*multiplicity)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*coords_stored)[CEED_Q_VLA]      = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*scale_stored)                   = out[1];
  CeedScalar *dXdx_q                          = out[2];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    for (int j = 0; j < 3; j++) coords_stored[j][i] = coords[j][i];
    scale_stored[i] = 1.0 / multiplicity[0][i];
    CeedScalar dXdx[3][3];
    InvertMappingJacobian_3D(Q, i, dxdX_q, dXdx, NULL);

    StoredValuesPack(Q, i, 0, 9, (CeedScalar *)dXdx, dXdx_q);
  }
  return 0;
}

typedef struct {
  CeedScalar q[5];
  CeedInt    mask[5];
} StrongContext;

CEED_QFUNCTION(StrongQF)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*scale) = (const CeedScalar(*))in[0];
  CeedScalar(*bcval)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const StrongContext *c = (const StrongContext *)ctx;

  // Apply mask once to get the per component target in q space
  CeedScalar q_set[5];
  for (CeedInt j = 0; j < 5; j++) q_set[j] = c->mask[j] ? c->q[j] : 0.0;
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    for (CeedInt j = 0; j < 5; j++) bcval[j][i] = scale[i] * q_set[j];
  }
  return 0;
}
