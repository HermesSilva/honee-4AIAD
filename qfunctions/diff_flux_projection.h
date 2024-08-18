// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include <ceed.h>

#include "utils.h"

// @brief QFunction to calculate the divergence of the diffusive flux
CEED_QFUNCTION_HELPER int ComputeDivDiffusiveFluxGeneric(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, const CeedInt dim,
                                                         const CeedInt num_comps) {
  const CeedScalar *grad_q   = in[0];
  const CeedScalar(*q_data)  = in[1];
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar dXdx[9];

    QdataUnpack_ND(dim, Q, i, q_data, NULL, dXdx);
    CeedPragmaSIMD for (CeedInt n = 0; n < num_comps; n++) {
      CeedScalar grad_qn[9];

      // Get gradient into dim x dim matrix form, with orientation [flux_direction][gradient_direction]
      // Equivalent of GradUnpackN
      const CeedInt offset = Q * n * dim;  // offset to reach nth component flux gradients
      for (CeedInt g = 0; g < dim; g++) {
        for (CeedInt f = 0; f < dim; f++) {
          grad_qn[f * dim + g] = grad_q[offset + (Q * num_comps * dim) * g + Q * f + i];
        }
      }
      v[n][i] = 0;
      DivergenceND(grad_qn, dXdx, dim, &v[n][i]);
    }
  }
  return 0;
}

CEED_QFUNCTION(ComputeDivDiffusiveFlux3D_4)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ComputeDivDiffusiveFluxGeneric(ctx, Q, in, out, 3, 4);
}

CEED_QFUNCTION(ComputeDivDiffusiveFlux3D_1)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ComputeDivDiffusiveFluxGeneric(ctx, Q, in, out, 3, 1);
}

CEED_QFUNCTION(ComputeDivDiffusiveFlux2D_1)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ComputeDivDiffusiveFluxGeneric(ctx, Q, in, out, 2, 1);
}
