// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Mass operator for Navier-Stokes example using PETSc
#include <ceed.h>
#include <math.h>

// *****************************************************************************
// This QFunction applies the mass matrix to N components
//
// Inputs:
//   u      - Input vector at quadrature points
//   q_data - Quadrature weights
//
// Output:
//   v - Output vector at quadrature points
//
// *****************************************************************************
CEED_QFUNCTION_HELPER int Mass_N(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, const CeedInt N) {
  const CeedScalar(*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data)        = in[1];
  CeedScalar(*v)[CEED_Q_VLA]       = (CeedScalar(*)[CEED_Q_VLA])out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedPragmaSIMD for (CeedInt j = 0; j < N; j++) { v[j][i] = q_data[i] * u[j][i]; }
  }
  return 0;
}

CEED_QFUNCTION(Mass_1)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) { return Mass_N(ctx, Q, in, out, 1); }

CEED_QFUNCTION(Mass_2)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) { return Mass_N(ctx, Q, in, out, 2); }

CEED_QFUNCTION(Mass_3)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) { return Mass_N(ctx, Q, in, out, 3); }

CEED_QFUNCTION(Mass_4)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) { return Mass_N(ctx, Q, in, out, 4); }

CEED_QFUNCTION(Mass_5)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) { return Mass_N(ctx, Q, in, out, 5); }

CEED_QFUNCTION(Mass_7)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) { return Mass_N(ctx, Q, in, out, 7); }

CEED_QFUNCTION(Mass_9)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) { return Mass_N(ctx, Q, in, out, 9); }

CEED_QFUNCTION(Mass_12)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) { return Mass_N(ctx, Q, in, out, 12); }

CEED_QFUNCTION(Mass_22)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) { return Mass_N(ctx, Q, in, out, 22); }
