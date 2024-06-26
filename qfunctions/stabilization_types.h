// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#pragma once

typedef enum {
  STAB_NONE = 0,
  STAB_SU   = 1,  // Streamline Upwind
  STAB_SUPG = 2,  // Streamline Upwind Petrov-Galerkin
} StabilizationType;

typedef enum {
  STAB_TAU_CTAU           = 0,
  STAB_TAU_ADVDIFF_SHAKIB = 1,  // Approximation from Shakib's Thesis
} StabilizationTauType;
