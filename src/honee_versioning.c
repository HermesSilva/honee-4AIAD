#include <honee.h>
/**
  @brief Get Honee library version info

  Honee version numbers have the form major.minor.patch. Non-release versions may contain unstable interfaces.

  @param[out]  major    Major version of the library
  @param[out]  minor    Minor version of the library
  @param[out]  patch    Patch (subminor) version of the library
  @param[out]  release  True for releases; false for development branches.

  The caller may pass NULL for any arguments that are not needed.

  @sa HONEE_VERSION_GE()

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode HoneeGetVersion(int *major, int *minor, int *patch, PetscBool *release) {
  if (major) *major = HONEE_VERSION_MAJOR;
  if (minor) *minor = HONEE_VERSION_MINOR;
  if (patch) *patch = HONEE_VERSION_PATCH;
  if (release) *release = HONEE_VERSION_RELEASE;
  return 0;
}
