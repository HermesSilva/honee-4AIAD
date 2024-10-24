// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#include <ceed/types.h>

// @brief Calculate the inverse of the multiplicity, reducing to a single component
CEED_QFUNCTION(InverseMultiplicity)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*multiplicity)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  CeedScalar(*inv_multiplicity)               = (CeedScalar(*))out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) inv_multiplicity[i] = 1.0 / multiplicity[0][i];
  return 0;
}
