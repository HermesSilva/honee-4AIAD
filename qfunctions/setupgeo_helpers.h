// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Geometric factors (3D) for HONEE
#pragma once

#include "utils.h"

#include <ceed/types.h>

/**
 * @brief Calculate dXdx from dxdX for 3D elements
 *
 * Reference (parent) coordinates: X
 * Physical (current) coordinates: x
 * Change of coordinate matrix: dxdX_{i,j} = x_{i,j} (indicial notation)
 * Inverse of change of coordinate matrix: dXdx_{i,j} = (detJ^-1) * X_{i,j}
 *
 * @param[in]  Q        Number of quadrature points
 * @param[in]  i        Current quadrature point
 * @param[in]  dxdX_q   Mapping Jacobian (gradient of the coordinate space)
 * @param[out] dXdx     Inverse of mapping Jacobian at quadrature point i
 * @param[out] detJ_ptr Determinate of the Jacobian, may be NULL is not desired
 */
CEED_QFUNCTION_HELPER void InvertMappingJacobian_3D(CeedInt Q, CeedInt i, const CeedScalar (*dxdX_q)[3][CEED_Q_VLA], CeedScalar dXdx[3][3],
                                                    CeedScalar *detJ_ptr) {
  CeedScalar dxdX[3][3];

  GradUnpack3D(Q, i, 3, (CeedScalar *)dxdX_q, dxdX);
  MatInv3(dxdX, dXdx, detJ_ptr);
}

/**
 * @brief Calculate dXdx from dxdX for 2D elements
 *
 * Reference (parent) coordinates: X
 * Physical (current) coordinates: x
 * Change of coordinate matrix: dxdX_{i,j} = x_{i,j} (indicial notation)
 * Inverse of change of coordinate matrix: dXdx_{i,j} = (detJ^-1) * X_{i,j}
 *
 * @param[in]  Q        Number of quadrature points
 * @param[in]  i        Current quadrature point
 * @param[in]  dxdX_q   Mapping Jacobian (gradient of the coordinate space)
 * @param[out] dXdx     Inverse of mapping Jacobian at quadrature point i
 * @param[out] detJ_ptr Determinate of the Jacobian, may be NULL is not desired
 */
CEED_QFUNCTION_HELPER void InvertMappingJacobian_2D(CeedInt Q, CeedInt i, const CeedScalar (*dxdX_q)[2][CEED_Q_VLA], CeedScalar dXdx[2][2],
                                                    CeedScalar *detJ_ptr) {
  CeedScalar dxdX[2][2];

  GradUnpack2D(Q, i, 2, (CeedScalar *)dxdX_q, dxdX);
  MatInv2(dxdX, dXdx, detJ_ptr);
}

/**
 * @brief Calculate face element's normal vector from dxdX
 *
 * Reference (parent) 2D coordinates: X
 * Physical (current) 3D coordinates: x
 * Change of coordinate matrix:
 *   dxdX_{i,j} = dx_i/dX_j (indicial notation) [3 * 2]
 * Inverse change of coordinate matrix:
 *   dXdx_{i,j} = dX_i/dx_j (indicial notation) [2 * 3]
 *
 * (N1,N2,N3) is given by the cross product of the columns of dxdX_{i,j}
 *
 * detJb is the magnitude of (N1,N2,N3)
 *
 * Normal vector = (N1,N2,N3) / detJb
 *
 * @param[in]  Q        Number of quadrature points
 * @param[in]  i        Current quadrature point
 * @param[in]  dxdX_q   Mapping Jacobian (gradient of the coordinate space)
 * @param[out] normal   Inverse of mapping Jacobian at quadrature point i
 * @param[out] detJ_ptr Determinate of the Jacobian, may be NULL is not desired
 */
CEED_QFUNCTION_HELPER void NormalVectorFromdxdX_3D(CeedInt Q, CeedInt i, const CeedScalar (*dxdX_q)[3][CEED_Q_VLA], CeedScalar normal[3],
                                                   CeedScalar *detJ_ptr) {
  CeedScalar dxdX[3][2];

  GradUnpack2D(Q, i, 3, (CeedScalar *)dxdX_q, dxdX);
  // N1, N2, and N3 are given by the cross product of the columns of dxdX
  normal[0] = dxdX[1][0] * dxdX[2][1] - dxdX[2][0] * dxdX[1][1];
  normal[1] = dxdX[2][0] * dxdX[0][1] - dxdX[0][0] * dxdX[2][1];
  normal[2] = dxdX[0][0] * dxdX[1][1] - dxdX[1][0] * dxdX[0][1];

  const CeedScalar detJ = Norm3(normal);
  ScaleN(normal, 1 / detJ, 3);
  if (detJ_ptr) *detJ_ptr = detJ;
}

/**
 * This QFunction sets up the geometric factor required for integration when reference coordinates are in 1D and the physical coordinates are in 2D
 *
 * Reference (parent) 1D coordinates: X
 * Physical (current) 2D coordinates: x
 * Change of coordinate vector:
 *           N1 = dx_1/dX
 *           N2 = dx_2/dX
 *
 * detJb is the magnitude of (N1,N2)
 *
 * We require the determinant of the Jacobian to properly compute integrals of the form: int( u v )
 *
 * Normal vector is given by the cross product of (N1,N2)/detJ and ẑ
 *
 * @param[in]  Q        Number of quadrature points
 * @param[in]  i        Current quadrature point
 * @param[in]  dxdX_q   Mapping Jacobian (gradient of the coordinate space)
 * @param[out] normal   Inverse of mapping Jacobian at quadrature point i
 * @param[out] detJ_ptr Determinate of the Jacobian, may be NULL is not desired
 */
CEED_QFUNCTION_HELPER void NormalVectorFromdxdX_2D(CeedInt Q, CeedInt i, const CeedScalar (*dxdX_q)[CEED_Q_VLA], CeedScalar normal[2],
                                                   CeedScalar *detJ_ptr) {
  normal[0]              = dxdX_q[1][i];
  normal[1]              = -dxdX_q[0][i];
  const CeedScalar detJb = Norm2(normal);
  ScaleN(normal, 1 / detJb, 2);
  if (detJ_ptr) *detJ_ptr = detJb;
}

/**
 * @brief Calculate inverse of mapping Jacobian, (dxdX)^-1
 *
 * Reference (parent) 2D coordinates: X
 * Physical (current) 3D coordinates: x
 * Change of coordinate matrix:
 *   dxdX_{i,j} = dx_i/dX_j (indicial notation) [3 * 2]
 * Inverse change of coordinate matrix:
 *   dXdx_{i,j} = dX_i/dx_j (indicial notation) [2 * 3]
 *
 * dXdx is calculated via Moore–Penrose inverse:
 *
 *   dX_i/dx_j = (dxdX^T dxdX)^(-1) dxdX
 *             = (dx_l/dX_i * dx_l/dX_k)^(-1) dx_j/dX_k
 *
 * @param[in]  Q      Number of quadrature points
 * @param[in]  i      Current quadrature point
 * @param[in]  dxdX_q Mapping Jacobian (gradient of the coordinate space)
 * @param[out] dXdx   Inverse of mapping Jacobian at quadrature point i
 */
CEED_QFUNCTION_HELPER void InvertBoundaryMappingJacobian_3D(CeedInt Q, CeedInt i, const CeedScalar (*dxdX_q)[3][CEED_Q_VLA], CeedScalar dXdx[2][3]) {
  CeedScalar dxdX[3][2];
  GradUnpack2D(Q, i, 3, (CeedScalar *)dxdX_q, dxdX);

  // dxdX_k,j * dxdX_j,k
  CeedScalar dxdXTdxdX[2][2] = {{0.}};
  for (CeedInt j = 0; j < 2; j++) {
    for (CeedInt k = 0; k < 2; k++) {
      for (CeedInt l = 0; l < 3; l++) dxdXTdxdX[j][k] += dxdX[l][j] * dxdX[l][k];
    }
  }

  const CeedScalar detdxdXTdxdX = dxdXTdxdX[0][0] * dxdXTdxdX[1][1] - dxdXTdxdX[1][0] * dxdXTdxdX[0][1];

  // Compute inverse of dxdXTdxdX
  CeedScalar dxdXTdxdX_inv[2][2];
  dxdXTdxdX_inv[0][0] = dxdXTdxdX[1][1] / detdxdXTdxdX;
  dxdXTdxdX_inv[0][1] = -dxdXTdxdX[0][1] / detdxdXTdxdX;
  dxdXTdxdX_inv[1][0] = -dxdXTdxdX[1][0] / detdxdXTdxdX;
  dxdXTdxdX_inv[1][1] = dxdXTdxdX[0][0] / detdxdXTdxdX;

  // Compute dXdx from dxdXTdxdX^-1 and dxdX
  for (CeedInt j = 0; j < 2; j++) {
    for (CeedInt k = 0; k < 3; k++) {
      dXdx[j][k] = 0;
      for (CeedInt l = 0; l < 2; l++) dXdx[j][k] += dxdXTdxdX_inv[l][j] * dxdX[k][l];
    }
  }
}
