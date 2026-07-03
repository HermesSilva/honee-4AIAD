// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include "utils.h"

#include <ceed/types.h>

/**
  @brief Calculate CFL for 3D spatial dimensions

  Defined as
  @f[
  \mathrm{CFL} = \Delta t \sqrt{\overline{g}_{jk} u_j u_k}
  @f]

  where \f$ \Delta t \f$ is the `timestep`, \f$ \overline{g}_{jk} \f$ is the element metric tensor `gij`, and \f$ u_j \f$ is the `velocity`.
  For best results, `gij` should be scaled to account for the reference element domain (often [-1,1] for Gaussian quadrature).

  @param[in] velocity Velocity
  @param[in] timestep Timestep
  @param[in] gij      Element metric tensor (should be scaled)
  @return    The CFL value for the element
**/
CEED_QFUNCTION_HELPER CeedScalar CalculateCFL_3D(const CeedScalar velocity[3], CeedScalar timestep, const CeedScalar gij[3][3]) {
  CeedScalar gij_uj[3] = {0.};

  MatVec3(gij, velocity, CEED_NOTRANSPOSE, gij_uj);
  return sqrt(Dot3(velocity, gij_uj)) * timestep;
}

/**
  @brief Calculate CFL for 2D spatial dimensions

  Defined as
  @f[
  \mathrm{CFL} = \Delta t \sqrt{\overline{g}_{jk} u_j u_k}
  @f]

  where \f$ \Delta t \f$ is the `timestep`, \f$ \overline{g}_{jk} \f$ is the element metric tensor `gij`, and \f$ u_j \f$ is the `velocity`.
  For best results, `gij` should be scaled to account for the reference element domain (often [-1,1] for Gaussian quadrature).

  @param[in] velocity Advection velocity
  @param[in] timestep Timestep
  @param[in] gij      Element metric tensor (should be scaled)
  @return    The CFL value for the element
**/
CEED_QFUNCTION_HELPER CeedScalar CalculateCFL_2D(const CeedScalar velocity[2], CeedScalar timestep, const CeedScalar gij[2][2]) {
  CeedScalar gij_uj[2] = {0.};

  MatVec2(gij, velocity, CEED_NOTRANSPOSE, gij_uj);
  return sqrt(Dot2(velocity, gij_uj)) * timestep;
}

/**
  @brief Calculate Peclet number for 3D spatial dimensions

  Defined as
  @f[
  \mathrm{Pe} = \frac{\sqrt{\overline{g}_{jk} u_j u_k}}{\kappa}
  @f]

  where \f$ \kappa \f$ is the `diffusion_coeff`, \f$ \overline{g}_{jk} \f$ is the element metric tensor `gij`, and \f$ u_j \f$ is the `velocity`.
  For best results, `gij` should be scaled to account for the reference element domain (often [-1,1] for Gaussian quadrature).

  @param[in] velocity        Advection velocity
  @param[in] diffusion_coeff Diffusion coefficient
  @param[in] gij             Element metric tensor (should be scaled)
  @return    The Peclet value for the element
**/
CEED_QFUNCTION_HELPER CeedScalar CalculatePe_3D(const CeedScalar velocity[3], CeedScalar diffusion_coeff, const CeedScalar gij[3][3]) {
  CeedScalar gij_uj[3] = {0.}, gij_inv_mat[3][3] = {{0.}};

  MatInv3(gij, gij_inv_mat, NULL);
  MatVec3(gij_inv_mat, velocity, CEED_NOTRANSPOSE, gij_uj);
  return sqrt(Dot3(velocity, gij_uj)) / diffusion_coeff;
}

/**
  @brief Calculate Peclet number for 2D spatial dimensions

  Defined as
  @f[
  \mathrm{Pe} = \frac{\sqrt{\overline{g}_{jk} u_j u_k}}{\kappa}
  @f]

  where \f$ \kappa \f$ is the `diffusion_coeff`, \f$ \overline{g}_{jk} \f$ is the element metric tensor `gij`, and \f$ u_j \f$ is the `velocity`.
  For best results, `gij` should be scaled to account for the reference element domain (often [-1,1] for Gaussian quadrature).

  @param[in] velocity        Advection velocity
  @param[in] diffusion_coeff Diffusion coefficient
  @param[in] gij             Element metric tensor (should be scaled)
  @return    The Peclet value for the element
**/
CEED_QFUNCTION_HELPER CeedScalar CalculatePe_2D(const CeedScalar velocity[2], CeedScalar diffusion_coeff, const CeedScalar gij[2][2]) {
  CeedScalar gij_uj[2] = {0.}, gij_inv_mat[2][2] = {{0.}};

  MatInv2(gij, gij_inv_mat, NULL);
  MatVec2(gij_inv_mat, velocity, CEED_NOTRANSPOSE, gij_uj);
  return sqrt(Dot2(velocity, gij_uj)) / diffusion_coeff;
}
