// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#pragma once
#include <ceed.h>
#include <petscdm.h>
#include <petscsection.h>
#include <petscts.h>
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
PetscErrorCode HoneeGetVersionString(char version[], size_t len);
PetscErrorCode HoneeGetCeedVersionString(char version[], size_t len);
PetscErrorCode HoneePrePetscInitialize();

PetscErrorCode HoneeSetContainer(Honee honee, const char key[], void *container, PetscCtxDestroyFn *container_destroy);
PetscErrorCode HoneeGetContainer(Honee honee, const char key[], void *container);
PetscErrorCode HoneeHasContainer(Honee honee, const char key[], PetscBool *has_key);

#define HONEE_MAX_BCDEFS 32
extern const DMLabel  DMLABEL_DEFAULT;
extern const PetscInt DMLABEL_DEFAULT_VALUE;

// Mesh Transformation
typedef enum {
  MESH_TRANSFORM_NONE      = 0,
  MESH_TRANSFORM_PLATEMESH = 1,
} MeshTransform;
static const char *const MeshTransforms[] = {"NONE", "PLATEMESH", "MeshTransform", "MESH_TRANSFORM_", NULL};
PetscErrorCode           HoneeMeshTransformFromOptions(DM dm);
