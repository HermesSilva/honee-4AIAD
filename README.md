# HONEE

[![GitLab-CI](https://gitlab.com/phypid/honee/badges/main/pipeline.svg?key_text=GitLab-CI)](https://gitlab.com/phypid/honee/badges/main/pipeline.svg?key_text=GitLab-CI)
[![BSD-2-Clause](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/license/Apache-2-0)
[![Documentation](https://img.shields.io/badge/Documentation-latest-blue.svg)](https://honee.phypid.org)
[![Code Coverage](https://gitlab.com/phypid/honee/badges/main/coverage.svg)](https://gitlab.com/phypid/honee/-/commits/main)

<!-- abstract -->
HONEE (High-Order Navier-stokes Equation Evaluator, pronounced "honey") is a fluids mechanics library based on [libCEED](https://libceed.org) and [PETSc](https://petsc.org) with support for efficient high-order elements and CUDA, ROCm, and Intel GPUs.

HONEE uses continuous-Galerkin stabilized finite element methods, namely SUPG, with a focus on scale-resolving simulations.
Effort is made to maintain flexibility in state variable choice, boundary conditions, time integration scheme (both implicit and explicit), and other solver choices.
High-order finite elements are implemented in an analytic matrix-free fashion to maximize performance on GPU hardware, while matrix assembly may also be used if desired.

# Getting Started
<!-- getting-started -->

## Download and Install

A local build and installation provides greater control over build options and optimization.
HONEE is open-source and can be downloaded from [the HONEE repository on GitLab](https://gitlab.com/phypid/honee).

```console
$ git clone https://gitlab.com/phypid/honee
```

### Prerequisites

HONEE is based upon libCEED and PETSc.

#### libCEED

HONEE requires libCEED's `main` development branch, which can be [cloned from Github](https://github.com/CEED/libCEED).

```console
$ git clone https://github.com/CEED/libCEED
$ make -C -j8 libCEED
```

The above will be enough for most simple CPU installations; see the [libCEED documentation](https://libceed.org/en/latest/gettingstarted/#) for details on using GPUs, tuning, and more complicated environments.

#### PETSc

HONEE requires PETSc's `main` development branch, which can be [cloned from GitLab](https://gitlab.com/petsc/petsc).

```console
$ git clone https://gitlab.com/petsc/petsc
```

Follow the [PETSc documentation](https://petsc.org/main/install/) to configure and build PETSc.
It is recommended that you install with [CGNS library](https://cgns.github.io/) support using `--download-hdf5 --download-cgns`.

### Building

The environment variables `CEED_DIR`, `PETSC_DIR`, and `PETSC_ARCH` must be set to build HONEE.

Assuming you have cloned the HONEE repository as above, build using:

```console
$ export CEED_DIR=[path to libCEED] PETSC_DIR=[path to PETSc] PETSC_ARCH=[PETSc arch]
$ make -j8
```

To run a sample problem, run:

```console
$ ./build/navierstokes -options_file ./examples/gaussianwave.yaml
```

To test the installation, use

```console
$ make test -j8
```

<!-- citing -->

## License

HONEE is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
 * BSD-2-Clause license ([LICENSE-BSD](LICENSE-BSD) or https://opensource.org/licenses/BSD-2-Clause)

at your option.
