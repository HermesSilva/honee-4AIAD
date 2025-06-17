#include <honee.h>

const char *HoneeGitVersion         = HONEE_GIT_VERSION;
const char *HoneeBuildConfiguration = HONEE_BUILD_CONFIGURATION;

/**
  @brief Get output of `git describe --dirty` from build time.

  While @ref HoneeGetVersion() uniquely identifies the source code for release builds, it does not identify builds from other commits.

  Not collective across MPI processes.

  @param[out]  git_version  A static string containing the Git commit description.
                            If `git describe --always --dirty` fails, the string `"unknown"` will be provided.
                            This could occur if Git is not installed or if Honee is not being built from a repository, for example.`

  @sa HoneeGetVersion() HoneeGetBuildConfiguration()

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode HoneeGetGitVersion(const char **git_version) {
  *git_version = HoneeGitVersion;
  return PETSC_SUCCESS;
}

/**
  @brief Get build variables as a multi-line string.

  Each line of the string has the format `VARNAME = value`.

  Not collective across MPI processes.

  @param[out]  build_config  A static string containing build variables

  @sa HoneeGetVersion() HoneeGetGitVersion()

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode HoneeGetBuildConfiguration(const char **build_config) {
  *build_config = HoneeBuildConfiguration;
  return PETSC_SUCCESS;
}
