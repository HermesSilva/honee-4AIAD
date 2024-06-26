// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include <petsc.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  TORCH_DEVICE_CPU,
  TORCH_DEVICE_CUDA,
  TORCH_DEVICE_HIP,
  TORCH_DEVICE_XPU,
} TorchDeviceType;
static const char *const TorchDeviceTypes[] = {"CPU", "CUDA", "HIP", "XPU", "TorchDeviceType", "TORCH_DEVICE_", NULL};

PetscErrorCode LoadModel_Torch(const char *model_path, TorchDeviceType device_enum);
PetscErrorCode ModelInference_Torch(Vec DD_Inputs_loc, Vec DD_Outputs_loc);

#ifdef __cplusplus
}
#endif
