# HONEE: High-Order Navier-stokes Equation Evaluator

<!-- abstract -->
This page provides a description of the Navier-Stokes example for the libCEED library, based on PETSc.

The Navier-Stokes problem solves the compressible Navier-Stokes equations in three dimensions using an explicit time integration.
The state variables are mass density, momentum density, and energy density.

The main Navier-Stokes solver for libCEED is defined in [`navierstokes.c`](navierstokes.c) with different problem definitions according to the application of interest.

## Getting Started

Build by using:

`make`

and run with:

```
./navierstokes -ceed [ceed] -problem [problem type] -degree [degree]
```

If you want to do *in situ* machine-learning training, specify `SMARTREDIS_DIR` in the make command like:

```
make SMARTREDIS_DIR=~/software/smartredis/install
```

