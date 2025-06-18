// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#pragma once
#include <ceed/types.h>

typedef enum {
  STAB_NONE = 0,
  STAB_SU   = 1,  // Streamline Upwind
  STAB_SUPG = 2,  // Streamline Upwind Petrov-Galerkin
} StabilizationType;
#ifndef CEED_RUNNING_JIT_PASS
static const char *const StabilizationTypes[] = {"NONE", "SU", "SUPG", "StabilizationType", "STAB_", NULL};
#endif

typedef enum {
  STAB_TAU_CTAU           = 0,
  STAB_TAU_ADVDIFF_SHAKIB = 1,  // Approximation from Shakib's Thesis
} StabilizationTauType;
#ifndef CEED_RUNNING_JIT_PASS
static const char *const StabilizationTauTypes[] = {"CTAU", "ADVDIFF_SHAKIB", "StabilizationTauType", "STAB_TAU_", NULL};
#endif

typedef enum {
  DIV_DIFF_FLUX_PROJ_NONE     = 0,
  DIV_DIFF_FLUX_PROJ_DIRECT   = 1,
  DIV_DIFF_FLUX_PROJ_INDIRECT = 2,
} DivDiffFluxProjectionMethod;
#ifndef CEED_RUNNING_JIT_PASS
static const char *const DivDiffFluxProjectionMethods[] = {"NONE", "DIRECT", "INDIRECT", "DivDiffFluxProjectionMethod", "DIV_DIFF_FLUX_PROJ_", NULL};
#endif
