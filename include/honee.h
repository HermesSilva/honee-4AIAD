// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#pragma once
#include <petsc.h>
#include <stdbool.h>
typedef struct Honee_private *Honee;

#define HONEE_VERSION_MAJOR 0
#define HONEE_VERSION_MINOR 0
#define HONEE_VERSION_PATCH 0
#define HONEE_VERSION_RELEASE false
/// Compile-time check that the the current library version is at least as
/// recent as the specified version. This macro is typically used in
/// @code
/// #if HONEE_VERSION_GE(0, 1, 0)
///   code path that needs at least 0.1.0
/// #else
///   fallback code for older versions
/// #endif
/// @endcode
///
/// A non-release version always compares as positive infinity.
///
#define HONEE_VERSION_GE(major, minor, patch) \
  (!HONEE_VERSION_RELEASE ||                  \
   (HONEE_VERSION_MAJOR > major ||            \
    (HONEE_VERSION_MAJOR == major && (HONEE_VERSION_MINOR > minor || (HONEE_VERSION_MINOR == minor && HONEE_VERSION_PATCH >= patch)))))

PetscErrorCode HoneeGetVersion(int *major, int *minor, int *patch, PetscBool *release);
PetscErrorCode HoneeGetGitVersion(const char **git_version);
PetscErrorCode HoneeGetBuildConfiguration(const char **build_config);
