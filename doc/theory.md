# Theory and Background

HONEE solves the time-dependent Navier-Stokes equations of compressible gas dynamics in a static Eulerian three-dimensional frame using unstructured high-order finite/spectral element spatial discretizations and explicit or implicit high-order time-stepping (available in PETSc).
Moreover, HONEE has been developed using PETSc, so that the pointwise physics (defined at quadrature points) is separated from the parallelization and meshing concerns.

## The Navier-Stokes Equations

The mathematical formulation (from {cite}`shakib1991femcfd`) is given in what follows.
The compressible Navier-Stokes equations in conservative form are

$$
\begin{aligned}
\frac{\partial \rho}{\partial t} + \nabla \cdot \bm{U} &= 0 \\
\frac{\partial \bm{U}}{\partial t} + \nabla \cdot \left( \frac{\bm{U} \otimes \bm{U}}{\rho} + P \bm{I}_3 -\bm\sigma \right) - \rho \bm{b}  &= 0 \\
\frac{\partial E}{\partial t} + \nabla \cdot \left( \frac{(E + P)\bm{U}}{\rho} -\bm{u} \cdot \bm{\sigma} - k \nabla T \right) - \rho \bm{b} \cdot \bm{u} &= 0 \, , \\
\end{aligned}
$$ (eq-ns)

where $\bm{\sigma} = \mu(\nabla \bm{u} + (\nabla \bm{u})^T + \lambda (\nabla \cdot \bm{u})\bm{I}_3)$ is the Cauchy (symmetric) stress tensor, with $\mu$ the dynamic viscosity coefficient, and $\lambda = - 2/3$ the Stokes hypothesis constant.
In equations {eq}`eq-ns`, $\rho$ represents the volume mass density, $U$ the momentum density (defined as $\bm{U}=\rho \bm{u}$, where $\bm{u}$ is the vector velocity field), $E$ the total energy density (defined as $E = \rho e$, where $e$ is the total energy including thermal and kinetic but not potential energy), $\bm{I}_3$ represents the $3 \times 3$ identity matrix, $\bm{b}$ is a body force vector (e.g., gravity vector $\bm{g}$),  $k$ the thermal conductivity constant, $T$ represents the temperature, and $P$ the pressure, given by the following equation of state

$$
P = \left( {c_p}/{c_v} -1\right) \left( E - {\bm{U}\cdot\bm{U}}/{(2 \rho)} \right) \, ,
$$ (eq-state)

where $c_p$ is the specific heat at constant pressure and $c_v$ is the specific heat at constant volume (that define $\gamma = c_p / c_v$, the specific heat ratio).

The system {eq}`eq-ns` can be rewritten in vector form

$$
\frac{\partial \bm{q}}{\partial t} + \nabla \cdot \bm{F}(\bm{q}) -S(\bm{q}) = 0 \, ,
$$ (eq-vector-ns)

for the state variables 5-dimensional vector

$$
\bm{q} =
\begin{pmatrix}
    \rho \\
    \bm{U} \equiv \rho \bm{ u }\\
    E \equiv \rho e
\end{pmatrix}
\begin{array}{l}
    \leftarrow\textrm{ volume mass density}\\
    \leftarrow\textrm{ momentum density}\\
    \leftarrow\textrm{ energy density}
\end{array}
$$

where the flux and the source terms, respectively, are given by

$$
\begin{aligned}
\bm{F}(\bm{q}) &=
\underbrace{\begin{pmatrix}
    \bm{U}\\
    {(\bm{U} \otimes \bm{U})}/{\rho} + P \bm{I}_3 \\
    {(E + P)\bm{U}}/{\rho}
\end{pmatrix}}_{\bm F_{\text{adv}}} +
\underbrace{\begin{pmatrix}
0 \\
-  \bm{\sigma} \\
 - \bm{u}  \cdot \bm{\sigma} - k \nabla T
\end{pmatrix}}_{\bm F_{\text{diff}}},\\
S(\bm{q}) &=
 \begin{pmatrix}
    0\\
    \rho \bm{b}\\
    \rho \bm{b}\cdot \bm{u}
\end{pmatrix}.
\end{aligned}
$$ (eq-ns-flux)

### Finite Element Formulation (Spatial Discretization)

Let the discrete solution be

$$
\bm{q}_N (\bm{x},t)^{(e)} = \sum_{k=1}^{P}\psi_k (\bm{x})\bm{q}_k^{(e)}
$$

with $P=p+1$ the number of nodes in the element $e$.
We use tensor-product bases $\psi_{kji} = h_i(X_0)h_j(X_1)h_k(X_2)$.

To obtain a finite element discretization, we first multiply the strong form {eq}`eq-vector-ns` by a test function $\bm v \in H^1(\Omega)$ and integrate,

$$
\int_{\Omega} \bm v \cdot \left(\frac{\partial \bm{q}_N}{\partial t} + \nabla \cdot \bm{F}(\bm{q}_N) - \bm{S}(\bm{q}_N) \right) \,dV = 0 \, , \; \forall \bm v \in \mathcal{V}_p\,,
$$

with $\mathcal{V}_p = \{ \bm v(\bm x) \in H^{1}(\Omega_e) \,|\, \bm v(\bm x_e(\bm X)) \in P_p(\bm{I}), e=1,\ldots,N_e \}$ a mapped space of polynomials containing at least polynomials of degree $p$ (with or without the higher mixed terms that appear in tensor product spaces).

Integrating by parts on the divergence term, we arrive at the weak form,

$$
\begin{aligned}
\int_{\Omega} \bm v \cdot \left( \frac{\partial \bm{q}_N}{\partial t} - \bm{S}(\bm{q}_N) \right)  \,dV
- \int_{\Omega} \nabla \bm v \!:\! \bm{F}(\bm{q}_N)\,dV & \\
+ \int_{\partial \Omega} \bm v \cdot \bm{F}(\bm q_N) \cdot \widehat{\bm{n}} \,dS
  &= 0 \, , \; \forall \bm v \in \mathcal{V}_p \,,
\end{aligned}
$$ (eq-weak-vector-ns)

where $\bm{F}(\bm q_N) \cdot \widehat{\bm{n}}$ is typically replaced with a boundary condition.

:::{note}
The notation $\nabla \bm v \!:\! \bm F$ represents contraction over both fields and spatial dimensions while a single dot represents contraction in just one, which should be clear from context, e.g., $\bm v \cdot \bm S$ contracts over fields while $\bm F \cdot \widehat{\bm n}$ contracts over spatial dimensions.
:::

### Time Discretization
<!-- TODO: This should be reframed in terms of PETSc TS's F(t, u, \dot u) = G(t, u) rather than specifically about Explicit RK -->
For the time discretization, we use two types of time stepping schemes through PETSc.


#### Explicit Time-Stepping Method
<!-- TODO:  This should talk about the mass operator and the options associated with it (i.e. lumped mass matrix )-->
The following explicit formulation is solved with the adaptive Runge-Kutta-Fehlberg (RKF4-5) method by default (any explicit time-stepping scheme available in PETSc can be chosen at runtime)

$$
\bm{q}_N^{n+1} = \bm{q}_N^n + \Delta t \sum_{i=1}^{s} b_i k_i \, ,
$$

where

$$
\begin{aligned}
    k_1 &= f(t^n, \bm{q}_N^n)\\
    k_2 &= f(t^n + c_2 \Delta t, \bm{q}_N^n + \Delta t (a_{21} k_1))\\
    k_3 &= f(t^n + c_3 \Delta t, \bm{q}_N^n + \Delta t (a_{31} k_1 + a_{32} k_2))\\
    \vdots&\\
    k_i &= f\left(t^n + c_i \Delta t, \bm{q}_N^n + \Delta t \sum_{j=1}^s a_{ij} k_j \right)\\
\end{aligned}
$$

and with

$$
f(t^n, \bm{q}_N^n) = - [\nabla \cdot \bm{F}(\bm{q}_N)]^n + [S(\bm{q}_N)]^n \, .
$$

#### Implicit Time-Stepping Method

This time stepping method which can be selected using the option `-implicit` is solved with Backward Differentiation Formula (BDF) method by default (similarly, any implicit time-stepping scheme available in PETSc can be chosen at runtime).
The implicit formulation solves nonlinear systems for $\bm q_N$:

$$
\bm f(\bm q_N) \equiv \bm g(t^{n+1}, \bm{q}_N, \bm{\dot{q}}_N) = 0 \, ,
$$ (eq-ts-implicit-ns)

where the time derivative $\bm{\dot q}_N$ is defined by

$$
\bm{\dot{q}}_N(\bm q_N) = \alpha \bm q_N + \bm z_N
$$

in terms of $\bm z_N$ from prior state and $\alpha > 0$, both of which depend on the specific time integration scheme (backward difference formulas, generalized alpha, implicit Runge-Kutta, etc.).
Each nonlinear system {eq}`eq-ts-implicit-ns` will correspond to a weak form, as explained below.
In determining how difficult a given problem is to solve, we consider the Jacobian of {eq}`eq-ts-implicit-ns`,

$$
\frac{\partial \bm f}{\partial \bm q_N} = \frac{\partial \bm g}{\partial \bm q_N} + \alpha \frac{\partial \bm g}{\partial \bm{\dot q}_N}.
$$

The scalar "shift" $\alpha$ scales inversely with the time step $\Delta t$, so small time steps result in the Jacobian being dominated by the second term, which is a sort of "mass matrix", and typically well-conditioned independent of grid resolution with a simple preconditioner (such as Jacobi).
In contrast, the first term dominates for large time steps, with a condition number that grows with the diameter of the domain and polynomial degree of the approximation space.
Both terms are significant for time-accurate simulation and the setup costs of strong preconditioners must be balanced with the convergence rate of Krylov methods using weak preconditioners.

More details of PETSc's time stepping solvers can be found in the [TS User Guide](https://petsc.org/release/docs/manual/ts/).

### Stabilization
We solve {eq}`eq-weak-vector-ns` using a Galerkin discretization (default) or a stabilized method, as is necessary for most real-world flows.

Galerkin methods produce oscillations for transport-dominated problems (any time the cell Péclet number is larger than 1), and those tend to blow up for nonlinear problems such as the Euler equations and (low-viscosity/poorly resolved) Navier-Stokes, in which case stabilization is necessary.
Our formulation follows {cite}`hughesetal2010`, which offers a comprehensive review of stabilization and shock-capturing methods for continuous finite element discretization of compressible flows.

- **SUPG** (streamline-upwind/Petrov-Galerkin)

  In this method, the weighted residual of the strong form {eq}`eq-vector-ns` is added to the Galerkin formulation {eq}`eq-weak-vector-ns`.
  The weak form for this method is given as

  $$
  \begin{aligned}
  \int_{\Omega} \bm v \cdot \left( \frac{\partial \bm{q}_N}{\partial t} - \bm{S}(\bm{q}_N) \right)  \,dV
  - \int_{\Omega} \nabla \bm v \!:\! \bm{F}(\bm{q}_N)\,dV & \\
  + \int_{\partial \Omega} \bm v \cdot \bm{F}(\bm{q}_N) \cdot \widehat{\bm{n}} \,dS & \\
  + \int_{\Omega} \nabla\bm v \tcolon\left(\frac{\partial \bm F_{\text{adv}}}{\partial \bm q}\right) \bm\tau \left( \frac{\partial \bm{q}_N}{\partial t} \, + \,
  \nabla \cdot \bm{F} \, (\bm{q}_N) - \bm{S}(\bm{q}_N) \right) \,dV &= 0
  \, , \; \forall \bm v \in \mathcal{V}_p
  \end{aligned}
  $$ (eq-weak-vector-ns-supg)

  This stabilization technique can be selected using the option `-stab supg`.

- **SU** (streamline-upwind)

  This method is a simplified version of *SUPG* {eq}`eq-weak-vector-ns-supg` which is developed for debugging/comparison purposes. The weak form for this method is

  $$
  \begin{aligned}
  \int_{\Omega} \bm v \cdot \left( \frac{\partial \bm{q}_N}{\partial t} - \bm{S}(\bm{q}_N) \right)  \,dV
  - \int_{\Omega} \nabla \bm v \!:\! \bm{F}(\bm{q}_N)\,dV & \\
  + \int_{\partial \Omega} \bm v \cdot \bm{F}(\bm{q}_N) \cdot \widehat{\bm{n}} \,dS & \\
  + \int_{\Omega} \nabla\bm v \tcolon\left(\frac{\partial \bm F_{\text{adv}}}{\partial \bm q}\right) \bm\tau \nabla \cdot \bm{F} \, (\bm{q}_N) \,dV
  & = 0 \, , \; \forall \bm v \in \mathcal{V}_p
  \end{aligned}
  $$ (eq-weak-vector-ns-su)

  This stabilization technique can be selected using the option `-stab su`.

In both {eq}`eq-weak-vector-ns-su` and {eq}`eq-weak-vector-ns-supg`, $\bm\tau \in \mathbb R^{5\times 5}$ (field indices) is an intrinsic time scale matrix.
The SUPG technique and the operator $\frac{\partial \bm F_{\text{adv}}}{\partial \bm q}$ (rather than its transpose) can be explained via an ansatz for subgrid state fluctuations $\tilde{\bm q} = -\bm\tau \bm r$ where $\bm r$ is a strong form residual.
The forward variational form can be readily expressed by differentiating $\bm F_{\text{adv}}$ of {eq}`eq-ns-flux`

$$
\begin{aligned}
\diff\bm F_{\text{adv}}(\diff\bm q; \bm q) &= \frac{\partial \bm F_{\text{adv}}}{\partial \bm q} \diff\bm q \\
&= \begin{pmatrix}
\diff\bm U \\
(\diff\bm U \otimes \bm U + \bm U \otimes \diff\bm U)/\rho - (\bm U \otimes \bm U)/\rho^2 \diff\rho + \diff P \bm I_3 \\
(E + P)\diff\bm U/\rho + (\diff E + \diff P)\bm U/\rho - (E + P) \bm U/\rho^2 \diff\rho
\end{pmatrix},
\end{aligned}
$$

where $\diff P$ is defined by differentiating {eq}`eq-state`.

:::{dropdown} Stabilization scale $\bm\tau$
A velocity vector $\bm u$ can be pulled back to the reference element as $\bm u_{\bm X} = \nabla_{\bm x}\bm X \cdot \bm u$, with units of reference length (non-dimensional) per second.
To build intuition, consider a boundary layer element of dimension $(1, \epsilon)$, for which $\nabla_{\bm x} \bm X = \bigl(\begin{smallmatrix} 2 & \\ & 2/\epsilon \end{smallmatrix}\bigr)$.
So a small normal component of velocity will be amplified (by a factor of the aspect ratio $1/\epsilon$) in this transformation.
The ratio $\lVert \bm u \rVert / \lVert \bm u_{\bm X} \rVert$ is a covariant measure of (half) the element length in the direction of the velocity.
A contravariant measure of element length in the direction of a unit vector $\hat{\bm n}$ is given by $\lVert \bigl(\nabla_{\bm X} \bm x\bigr)^T \hat{\bm n} \rVert$.
While $\nabla_{\bm X} \bm x$ is readily computable, its inverse $\nabla_{\bm x} \bm X$ is needed directly in finite element methods and thus more convenient for our use.
If we consider a parallelogram, the covariant measure is larger than the contravariant measure for vectors pointing between acute corners and the opposite holds for vectors between oblique corners.

The cell Péclet number is classically defined by $\mathrm{Pe}_h = \lVert \bm u \rVert h / (2 \kappa)$ where $\kappa$ is the diffusivity (units of $m^2/s$).
This can be generalized to arbitrary grids by defining the local Péclet number

$$
\mathrm{Pe} = \frac{\lVert \bm u \rVert^2}{\lVert \bm u_{\bm X} \rVert \kappa}.
$$ (eq-peclet)

For scalar advection-diffusion, the stabilization is a scalar

$$
\tau = \frac{\xi(\mathrm{Pe})}{\lVert \bm u_{\bm X} \rVert},
$$ (eq-tau-advdiff)

where $\xi(\mathrm{Pe}) = \coth \mathrm{Pe} - 1/\mathrm{Pe}$ approaches 1 at large local Péclet number.
Note that $\tau$ has units of time and, in the transport-dominated limit, is proportional to element transit time in the direction of the propagating wave.
For advection-diffusion, $\bm F(q) = \bm u q$, and thus the SU stabilization term is

$$
\nabla v \cdot \bm u \tau \bm u \cdot \nabla q = \nabla_{\bm X} v \cdot (\bm u_{\bm X} \tau \bm u_{\bm X}) \cdot \nabla_{\bm X} q .
$$ (eq-su-stabilize-advdiff)

where the term in parentheses is a rank-1 diffusivity tensor that has been pulled back to the reference element.
See {cite}`hughesetal2010` equations 15-17 and 34-36 for further discussion of this formulation.

(theory-ns-tau)=
#### Navier-Stokes $\tau$ definition

For the Navier-Stokes and Euler equations, {cite}`whiting2003hierarchical` defines a $5\times 5$ diagonal stabilization $\mathrm{diag}(\tau_c, \tau_m, \tau_m, \tau_m, \tau_E)$ consisting of
1. continuity stabilization $\tau_c$
2. momentum stabilization $\tau_m$
3. energy stabilization $\tau_E$

The Navier-Stokes code in this example uses the following formulation for $\tau_c$, $\tau_m$, $\tau_E$:

$$
\begin{aligned}
\tau_c &= \frac{C_c \mathcal{F}}{8\rho \trace(\bm g)} \\
\tau_m &= \frac{C_m}{\mathcal{F}} \\
\tau_E &= \frac{C_E}{\mathcal{F} c_v} \\
\end{aligned}
$$

$$
\mathcal{F} = \sqrt{ \rho^2 \left [ \left(\frac{2C_t}{\Delta t}\right)^2
+ \bm u \cdot (\bm u \cdot  \bm g)\right]
+ C_v \mu^2 \Vert \bm g \Vert_F ^2}
$$

where $\bm g = \nabla_{\bm x} \bm{X}^T \cdot \nabla_{\bm x} \bm{X}$ is the metric tensor and $\Vert \cdot \Vert_F$ is the Frobenius norm.
This formulation is currently not available in the Euler code.

#### Advection-Diffusion $\tau$ definition

For Advection-Diffusion, we first examine a 1D definition given by:

$$
    \tau = \textrm{minreg}_2 \left\{\frac{\Delta t}{2 C_t},\ \frac{h}{aC_a}, \ \frac{h^2}{\kappa C_d} \right\}
$$

for $C_t$, $C_a$, $C_d$ being some scaling coefficients, $h$ is the element length, and $\textrm{minreg}_n \{x_j\} = (\sum_j x_j^{-n})^{-1/n}$.
To make this definition compatible with higher dimensional domains, we use a similar system from the Navier-Stokes equations.
This results in the following definition:

$$
\begin{aligned}
    \tau &= \textrm{minreg}_2 \left \{ \frac{\Delta t}{2 C_t}, \frac{1}{C_a \sqrt{\bm u \cdot (\bm u \cdot  \bm g)}}, \frac{1}{C_d \kappa \Vert \bm g \Vert_F} \right\} \\
         &= \left [ \left(\frac{2 C_t}{\Delta t}\right)^2 + C_a^2 \bm u \cdot (\bm u \cdot  \bm g) + \left(C_d \kappa\right)^2 \Vert \bm g \Vert_F^2\right]^{-1/2}
\end{aligned}
$$

Note that $\bm g$ is scaled so that it is identity for a unit square, keeping this definition aligned with the traditional 1D definition, which uses the element length directly.
The scaling coefficients are set via `-Ctau_t`, `-Ctau_a`, and `-Ctau_d`, respectively.

#### Euler $\tau$ definition
In the Euler code, we follow {cite}`hughesetal2010` in defining a $3\times 3$ diagonal stabilization according to spatial criterion 2 (equation 27) as follows.

$$
\tau_{ii} = c_{\tau} \frac{2 \xi(\mathrm{Pe})}{(\lambda_{\max \text{abs}})_i \lVert \nabla_{x_i} \bm X \rVert}
$$ (eq-tau-conservative)

where $c_{\tau}$ is a multiplicative constant reported to be optimal at 0.5 for linear elements, $\hat{\bm n}_i$ is a unit vector in direction $i$, and $\nabla_{x_i} = \hat{\bm n}_i \cdot \nabla_{\bm x}$ is the derivative in direction $i$.
The flux Jacobian $\frac{\partial \bm F_{\text{adv}}}{\partial \bm q} \cdot \hat{\bm n}_i$ in each direction $i$ is a $5\times 5$ matrix with spectral radius $(\lambda_{\max \text{abs}})_i$ equal to the fastest wave speed.
The complete set of eigenvalues of the Euler flux Jacobian in direction $i$ are (e.g., {cite}`toro2009`)

$$
\Lambda_i = [u_i - a, u_i, u_i, u_i, u_i+a],
$$ (eq-eigval-advdiff)

where $u_i = \bm u \cdot \hat{\bm n}_i$ is the velocity component in direction $i$ and $a = \sqrt{\gamma P/\rho}$ is the sound speed for ideal gasses.
Note that the first and last eigenvalues represent nonlinear acoustic waves while the middle three are linearly degenerate, carrying a contact wave (temperature) and transverse components of momentum.
The fastest wave speed in direction $i$ is thus

$$
\lambda_{\max \text{abs}} \Bigl( \frac{\partial \bm F_{\text{adv}}}{\partial \bm q} \cdot \hat{\bm n}_i \Bigr) = |u_i| + a
$$ (eq-wavespeed)

Note that this wave speed is specific to ideal gases as $\gamma$ is an ideal gas parameter; other equations of state will yield a different acoustic wave speed.

:::

(theory-divdiff)=
### Divergence of Diffusive Flux Projection

The strong residual in the SUPG operator in {eq}`eq-weak-vector-ns-supg` and {eq}`eq-weak-vector-ns-su` features the term $\nabla \cdot \bm F_{\text{diff}}(\bm{q}_N)$, the divergence of the diffusive flux.
This term requires a second derivative to evaluate; first to evaluate $\bm \sigma$ and $\nabla T$ for $F_{\text{diff}}$, the second for the divergence of the flux.
For linear elements, the flux is constant within each element so the second derivative is zero, leading to accuracy issues.
Additionally, libCEED does not currently support calculating double-derivatives.
To circumvent these issues, we (optionally) perform a projection operation to get $\nabla \cdot \bm F_{\text{diff}}(\bm{q}_N)$ at quadrature points.
This was first proposed in {cite}`jansenDiffFluxProjection1999`.
There are two methods of achieving this implemented in HONEE, denoted as the direct and indirect methods.

#### Indirect Projection

Indirect projection is the method presented in {cite}`jansenDiffFluxProjection1999`.
Here, $\bm F_{\text{diff}}$ is $L^2$ projected onto the finite element space and then the divergence is taken from that FEM function.
For linear basis functions, this leads to constant values of $\nabla \cdot \bm F_{\text{diff}}$ within each element.

For compressible Navier-Stokes, this requires projecting 12 scalars-per-node: 4 conserved scalars (mass conservation does not have a diffusive flux term) in 3 dimensional directions.
These 12 scalar finite element functions' derivatives are then evaluated at quadrature points and the divergence is calculated.
This method can be selected with `-div_diff_flux_projection_method indirect`.

#### Direct Projection
In the direct projection method, we perform an $L^2$ projection of the divergence of the diffusive flux itself, $\nabla \cdot \bm F_{\text{diff}}(\bm{q}_N)$.
Then $\nabla \cdot \bm F_{\text{diff}}$ itself is a function on the finite element space and can be interpolated onto quadrature points.

To do this, look at the RHS of the $L^2$ projection of $\nabla \cdot \bm F_{\text{diff}}(\bm{q}_N)$:

$$
\int_{\Omega} \bm v \cdot \nabla \cdot \bm F_{\text{diff}}(\bm{q}_N) \,dV
$$

As noted, we can't calculate $\nabla \cdot \bm F_{\text{diff}}(\bm{q}_N)$ at quadrature points, so we apply integration-by-parts to achieve a calculable RHS:

$$
\int_{\partial \Omega} \bm v \cdot \bm{F}_{\text{diff}}(\bm q_N) \cdot \widehat{\bm{n}} \,dS
- \int_{\Omega} \nabla \bm v \!:\! \bm{F}_{\text{diff}}(\bm{q}_N)\,dV
$$

This form is what is used for calculating the RHS of the projection.
After the projection, $\nabla \cdot \bm F_{\text{diff}}(\bm{q}_N)$ is interpolated directly to quadrature points without any extra calculations necessary.
For compressible Navier-Stokes, this means only projecting only 4 scalars-per-node.

The projection can be enabled using `-div_diff_flux_projection_method direct`.

#### General Information
The $L^2$ projection in either method uses the standard mass matrix, which is rowsum lumped for performance by default.
The linear solve for the projection can be controlled via `-div_diff_flux_projection_ksp*` flags.

(theory-sgs)=
### Subgrid Stress Modeling

When a fluid simulation is under-resolved (the smallest length scale resolved by the grid is much larger than the smallest physical scale, the [Kolmogorov length scale](https://en.wikipedia.org/wiki/Kolmogorov_microscales)), this is mathematically interpreted as filtering the Navier-Stokes equations.
This is known as large-eddy simulation (LES), as only the "large" scales of turbulence are resolved.
This filtering operation results in an extra stress-like term, $\bm{\tau}^r$, representing the effect of unresolved (or "subgrid" scale) structures in the flow.
Denoting the filtering operation by $\overline \cdot$, the LES governing equations are:

$$
\frac{\partial \bm{\overline q}}{\partial t} + \nabla \cdot \bm{\overline F}(\bm{\overline q}) -S(\bm{\overline q}) = 0 \, ,
$$ (eq-vector-les)

where

$$
\bm{\overline F}(\bm{\overline q}) =
\bm{F} (\bm{\overline q}) +
\begin{pmatrix}
    0\\
     \bm{\tau}^r \\
     \bm{u}  \cdot \bm{\tau}^r
\end{pmatrix}
$$ (eq-les-flux)

More details on deriving the above expression, filtering, and large eddy simulation can be found in {cite}`popeTurbulentFlows2000`.
To close the problem, the subgrid stress must be defined.
For implicit LES, the subgrid stress is set to zero and the numerical properties of the discretized system are assumed to account for the effect of subgrid scale structures on the filtered solution field.
For explicit LES, it is defined by a subgrid stress model.

:::{list-table} SGS Model Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-sgs_model_type`
  - Type of subgrid stress model to use. Currently only `data_driven` is available
  - `none`
  - string
:::

(sgs-dd-model)=
#### Data-Driven SGS Model

The data-driven SGS model implemented here uses a small neural network to compute the SGS term.
The SGS tensor is calculated at nodes using an $L^2$ projection of the velocity gradient and grid anisotropy tensor, and then interpolated onto quadrature points.
More details regarding the theoretical background of the model can be found in {cite}`prakashDDSGS2022` and {cite}`prakashDDSGSAnisotropic2022`.

The neural network itself consists of 1 hidden layer and 20 neurons, using Leaky ReLU as its activation function.
The slope parameter for the Leaky ReLU function is set via `-sgs_model_dd_leakyrelu_alpha`.
The outputs of the network are assumed to be normalized on a min-max scale, so they must be rescaled by the original min-max bounds.
Parameters for the neural network are put into files in a directory found in `-sgs_model_dd_parameter_dir`.
These files store the network weights (`w1.dat` and `w2.dat`), biases (`b1.dat` and `b2.dat`), and scaling parameters (`OutScaling.dat`).
The first row of each files stores the number of columns and rows in each file.
Note that the weight coefficients are assumed to be in column-major order.
This is done to keep consistent with legacy file compatibility.

:::{note}
The data-driven model parameters in the examples directory are not accurate and are for regression testing only.
:::

##### Data-Driven Model Using External Libraries

There are two different modes for using the data-driven model: fused and sequential.

In fused mode, the input processing, model inference, and output handling were all done in a single CeedOperator.
Fused mode is generally faster than the sequential mode, however fused mode requires that the model architecture be manually implemented into a libCEED QFunction.
To use the fused mode, set `-sgs_model_dd_implementation fused`.

Sequential mode has separate function calls/CeedOperators for input creation, model inference, and output handling.
By separating the three steps of the model evaluation, the sequential mode allows for functions calling external libraries to be used for the model inference step.
The use of these external libraries allows us to leverage the flexibility of those external libraries in their model architectures.

PyTorch is currently the only external library implemented with the sequential mode.
This is enabled with `USE_TORCH=1` during the build process, which will use the PyTorch accessible from the build environment's Python interpreter.
To specify the path to the PyTorch model file, use `-sgs_model_dd_torch_model_path`.
The hardware used to run the model inference is determined automatically from the libCEED backend chosen, but can be overridden with `-sgs_model_dd_torch_model_device`.
Note that if you chose to run the inference on host while using a GPU libCEED backend (e.g. `/gpu/cuda`), then host-to-device transfers (and vice versa) will be done automatically.

The sequential mode is available using a libCEED based inference evaluation via `-sgs_model_dd_implementation sequential_ceed`, but it is only for verification purposes.

:::{list-table} Data-driven SGS Model Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-sgs_model_dd_leakyrelu_alpha`
  - Slope parameter for Leaky ReLU activation function. `0` corresponds to normal ReLU
  - 0
  -

* - `-sgs_model_dd_parameter_dir`
  - Path to directory with data-driven model parameters (weights, biases, etc.)
  - `./dd_sgs_parameters`
  - string

* - `-sgs_model_dd_model_implementation`
  - Which computational implementation to use for SGS DD model (`fused`, `sequential_ceed`, `sequential_torch`)
  - `fused`
  - string

* - `-sgs_model_dd_torch_model_path`
  - Path to the PyTorch `*.pt` file containing the DD inference model
  -
  - string

* - `-sgs_model_dd_torch_model_device`
  - What hardware to perform the model inference on (`cpu`, `cuda`, `hip`, `xpu`)
  - Default matches the libCEED backend
  - string
:::


## Boundary Conditions

Below we detail the natural BC types for Newtonian.
Natural BCs are applied by replacing the normal flux in the boundary integral with a specific $\bm{h}$ flux:

$$\int_\Gamma \bm{v} \cdot \bm{h}\diff \Gamma$$

Below, we specify what $\bm{h}$ is equal to.

### Consistency Integral

The most basic natural BC type is the consistency integral.
This simply evaluates the flux in {eq}`eq-ns-flux` based on the interior solution at the boundary:

$$\bm{h} = \bm{F}_i(\bm{q}^\mathrm{int}) \hat{n}_i$$
where $\bm{q}^\mathrm{int}$ is the solution state at the boundary.
The consistency natural BC type is not meant to be used on its own, but in conjunction with some kind of essential boundary condition, such as {ref}`bc-stg`.
It is specified by `-bc_{name}_natural_type consistent` and has no other options.

(riemann-solvers)=
### Riemann Solvers

In HONEE, we use Riemann solvers at to determine the flux to apply at natural boundary conditions.
We give a brief introduction to them below.

Riemann solvers are methods to approximately solve the [Riemann problem](https://en.wikipedia.org/wiki/Riemann_problem).
The Riemann problem asks what is the inviscid flux normal to a boundary between fluids at two different constant states.
We refer to the two different states at the interior and exterior state, respectively.
The solution to the Riemann problem can be expressed as:

$$\bm{F}^{\mathrm{Riemann}}(\bm{Y}^{\mathrm{int}}, \bm{Y}^{\mathrm{ext}})$$
where $\bm{F}^{\mathrm{Riemann}}$ is the flux, $\bm{Y}^{\mathrm{int}}$ is the interior state (given in primitive variables), and $\bm{Y}^{\mathrm{ext}}$ is the exterior state.

In HONEE, we use Riemann solvers at to determine the flux to apply at natural boundary conditions.
As such, $\bm{Y}^{\mathrm{int}}$ is nearly always the solution at the boundary of the domain, while $\bm{Y}^{\mathrm{ext}}$ is some other state, often specified by the user.
The advantage of using Riemann solvers is that they allow acoustic waves to exit the domain cleanly and are generally stable to recirculation.

### Freestream

The freestream natural type uses a Riemann solver at the boundary for far-field boundaries with known exterior state and is set by `-bc_{name}_natural_type freestream`:

$$\bm{h} = \bm{F}^{\mathrm{Riemann}}(\bm{Y}^{\mathrm{int}}, \bm{Y}^{\mathrm{ext}})$$
It can handle local inflow and outflow dynamically and is the least acoustically reflective natural BC type.
The Riemann solver finds the flux at the boundary based on the state on the interior of the domain (at the boundary) and a user-set exterior state.
The exterior primitive state $\bm{Y}^\mathrm{ext} = (P^\mathrm{ext}, \bm{u}^\mathrm{ext}, T^\mathrm{ext})$ is set by`-bc_{name}_freestream_pressure`, `-bc_{name}_freestream_velocity`, and `-bc_{name}_freestream_temperature`.

:::{list-table} Freestream Natural Type Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-bc_{name}_freestream_riemann`
  - Riemann solver used for freestream boundary flux (`hll`, `hllc`)
  - `hllc`
  -

* - `-bc_{name}_freestream_pressure`
  - Pressure at freestream condition
  - `-reference_pressure`
  - `Pa`

* - `-bc_{name}_freestream_velocity`
  - Velocity at freestream condition, array of size 3
  - `-reference_velocity`
  - `m/s`

* - `-bc_{name}_freestream_temperature`
  - Temperature at freestream condition
  - `-reference_temperature`
  - `K`
:::

### Outflow

The outflow natural type has two different modes for use with predominantly-outflow boundaries
It is set by `-bc_{name}_natural_type outflow`.

The first mode is `-bc_{name}_outflow_type pressure`, which applies the consistency integral, but forcing the pressure to a user-set value.

$$\bm{h} = \bm{F}_i(\tilde{\bm{Y}}^\mathrm{int}) \hat{n}_i, \quad
\tilde{\bm{Y}}^\mathrm{int} =
\begin{pmatrix}
    P^\mathrm{ext} \\
    \bm{u}^\mathrm{int} \\
    T^\mathrm{int}
\end{pmatrix}
$$
where $P^\mathrm{ext}$ is the user-set value.
This has the effect of weakly enforcing a pressure at the boundary.
The `pressure` outflow requires that the flow have no recirculation at the boundary or the problem becomes unstable.
Additionally, the `pressure` outflow is very acoustically reflective.
These limitations motivate the second mode.

The second mode is `-bc_{name}_outflow_type riemann`, which uses a Riemann solver at the outflow to define the inviscid flux and adds in viscous fluxes at the boundary:

$$\bm{h} = \bm{F}^{\mathrm{Riemann}}(\bm{Y}^{\mathrm{int}}, \tilde{\bm{Y}}^{\mathrm{ext}}) + F_i^{\mathrm{diff}}(Y^{\mathrm{int}})\hat{n}_i, \quad
\tilde{\bm{Y}}^\mathrm{ext} =
\begin{pmatrix}
    P^\mathrm{ext} \\
    \bm{u}^\mathrm{int} \\
    T^\mathrm{ext}
\end{pmatrix}
$$
where the diffusive flux is defined in {eq}`eq-ns-flux`.
This is similar to `freestream`, but differs in 2 key ways.
First, the exterior state uses the interior velocity, which allows the boundary condition to handle significant velocity changes of the interior solution (a common feature at simulation outflows).
Second, we add in the diffusive (viscous) flux.
This makes the `riemann` outflow more consistent in viscous-dominated regions such as near no-slip walls.

This combination of features allows the `riemann` flux to be applied at boundaries near viscous-dominate boundaries, handle unknown velocity profiles, allow acoustic waves to exit the domain, and be resilient to recirculation.
As such, we generally recommend `riemann` over `pressure`: `pressure` variant is retained to facilitate comparison with other codes.

For `riemann` outflow, the velocity set in the exterior state can be modified to only allow some amount of recirculation in the exterior state.
That modification is given as $\Delta u_n = (1-r)\,\mathrm{Softplus}(-u_n, u_s)$, where $r$ is set by `-bc_{name}_outflow_recirc`, and $u_s$ is set by `-bc_{name}_outflow_softplus_velocity`.
The intention is to penalize recirculation, but it has not shown much benefit and often makes the simulation more unstable.

:::{list-table} Outflow Natural Type Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-bc_{name}_outflow_type`
  - Outflow model selection (`riemann`, `pressure`)
  - `riemann`
  -

* - `-bc_{name}_outflow_pressure`
  - Pressure at outflow condition
  - `-reference_pressure`
  - `Pa`

* - `-bc_{name}_outflow_temperature`
  - Temperature at outflow condition (Riemann outflow only)
  - `-reference_temperature`
  - `K`

* - `-bc_{name}_outflow_recirc`
  - Fraction of recirculation allowed in exterior velocity state, $r \in [0,1]$ (Riemann outflow only)
  - `1.0`
  -

* - `-bc_{name}_outflow_softplus_velocity`
  - Characteristic velocity scale (Riemann outflow only)
  - `1e-2`
  - `m/s`
:::

### Slip

A slip natural type uses a Riemann function to weakly impose no-penetration, while allowing acoustic waves to exit the domain cleanly.
To do this, `slip` sets the exterior state identical to the interior state, but reflects the velocity vector normal to the boundary:

$$\bm{h} = \bm{F}^{\mathrm{Riemann}}(\bm{Y}^{\mathrm{int}}, \tilde{\bm{Y}}^{\mathrm{ext}}), \quad
\tilde{\bm{Y}}^\mathrm{ext} =
\begin{pmatrix}
    P^\mathrm{int} \\
    \bm{u}^\mathrm{ext} \\
    T^\mathrm{int}
\end{pmatrix}
$$
where $\bm{u}^\mathrm{ext} = u_i^{int} - 2 n_i u_j^\mathrm{int} n_j$.
It is set by `-bc_{name}_natural_type slip` and has no other options.

(bc-periodicity)=
### Periodicity

PETSc provides two ways to specify periodicity:

1. Isoperiodicity, in which the donor and receiver dofs are distinct in local vectors.
To use isoperiodicity with a PETSc-built box mesh, use the `zbox` shape, as in:

```yaml
dm_plex:
  shape: zbox
  box_faces: 10,12,4
  box_bd: none,none,periodic
```

Isoperiodicity enables standard boundary integrals, and is fully supported periodicity method.
The `zbox` method uses [Z-ordering](https://en.wikipedia.org/wiki/Z-order_curve) to construct the mesh in parallel and provide an adequate initial partition, which makes it higher performance and avoids needing a partitioning package.

2. Topological periodicity, in which the donor and receiver dofs are the same, obtained using:

```yaml
dm_plex:
  shape: box
  box_faces: 10,12,4
  box_bd: none,none,periodic
```

The coordinates for such cases are stored as a new field with special cell-based indexing to enable wrapping through the boundary.
This choice of coordinates prevents evaluating boundary integrals that cross the periodicity, such as for the outflow Riemann problem in the presence of spanwise periodicity.


(bc-stg)=
### Synthetic Turbulence Generation (STG)

We use the STG method described in {cite}`shurSTG2014`.
Below follows a re-description of the formulation to match the present notation, and then a description of the implementation and usage.

#### Equation Formulation

$$
\bm{u}(\bm{x}, t) = \bm{\overline{u}}(\bm{x}) + \bm{C}(\bm{x}) \cdot \bm{v}'
$$

$$
\begin{aligned}
\bm{v}' &= 2 \sqrt{3/2} \sum^N_{n=1} \sqrt{q^n(\bm{x})} \bm{\sigma}^n \cos(\kappa^n \bm{d}^n \cdot \bm{\hat{x}}^n(\bm{x}, t) + \phi^n ) \\
\bm{\hat{x}}^n &= \left[(x - U_0 t)\max(2\kappa_{\min}/\kappa^n, 0.1) , y, z  \right]^T
\end{aligned}
$$

Here, we define the number of wavemodes $N$, set of random numbers $ \{\bm{\sigma}^n, \bm{d}^n, \phi^n\}_{n=1}^N$, the Cholesky decomposition of the Reynolds stress tensor $\bm{C}$ (such that $\bm{R} = \bm{CC}^T$ ), bulk velocity $U_0$, wavemode amplitude $q^n$, wavemode frequency $\kappa^n$, and $\kappa_{\min} = 0.5 \min_{\bm{x}} (\kappa_e)$.

$$
\kappa_e = \frac{2\pi}{\min(2d_w, 3.0 l_t)}
$$

where $l_t$ is the turbulence length scale, and $d_w$ is the distance to the nearest wall.


The set of wavemode frequencies is defined by a geometric distribution:

$$
\kappa^n = \kappa_{\min} (1 + \alpha)^{n-1} \ , \quad \forall n=1, 2, ... , N
$$

The wavemode amplitudes $q^n$ are defined by a model energy spectrum $E(\kappa)$:

$$
q^n = \frac{E(\kappa^n) \Delta \kappa^n}{\sum^N_{n=1} E(\kappa^n)\Delta \kappa^n} \ ,\quad \Delta \kappa^n = \kappa^n - \kappa^{n-1}
$$

$$ E(\kappa) = \frac{(\kappa/\kappa_e)^4}{[1 + 2.4(\kappa/\kappa_e)^2]^{17/6}} f_\eta f_{\mathrm{cut}} $$

$$
f_\eta = \exp \left[-(12\kappa /\kappa_\eta)^2 \right], \quad
f_\mathrm{cut} = \exp \left( - \left [ \frac{4\max(\kappa-0.9\kappa_\mathrm{cut}, 0)}{\kappa_\mathrm{cut}} \right]^3 \right)
$$

$\kappa_\eta$ represents turbulent dissipation frequency, and is given as $2\pi (\nu^3/\varepsilon)^{-1/4}$ with $\nu$ the kinematic viscosity and $\varepsilon$ the turbulent dissipation.
$\kappa_\mathrm{cut}$ approximates the effective cutoff frequency of the mesh (viewing the mesh as a filter on solution over $\Omega$) and is given by:

$$
\kappa_\mathrm{cut} = \frac{2\pi}{ 2\min\{ [\max(h_y, h_z, 0.3h_{\max}) + 0.1 d_w], h_{\max} \} }
$$

The enforcement of the boundary condition is identical to the blasius inflow; it weakly enforces velocity, with the option of weakly enforcing either density or temperature using the `-weakT` flag.

#### Initialization Data Flow

Data flow for initializing function (which creates the context data struct) is given below:
```{mermaid}
flowchart LR
    subgraph STGInflow.dat
    y
    lt[l_t]
    eps
    Rij[R_ij]
    ubar
    end

    subgraph STGRand.dat
    rand[RN Set];
    end

    subgraph User Input
    u0[U0];
    end

    subgraph init[Create Context Function]
    ke[k_e]
    N;
    end
    lt --Calc-->ke --Calc-->kn
    y --Calc-->ke

    subgraph context[Context Data]
    yC[y]
    randC[RN Set]
    Cij[C_ij]
    u0 --Copy--> u0C[U0]
    kn[k^n];
    ubarC[ubar]
    ltC[l_t]
    epsC[eps]
    end
    ubar --Copy--> ubarC;
    y --Copy--> yC;
    lt --Copy--> ltC;
    eps --Copy--> epsC;

    rand --Copy--> randC;
    rand --> N --Calc--> kn;
    Rij --Calc--> Cij[C_ij]
```

This is done once at runtime.
The spatially-varying terms are then evaluated at each quadrature point on-the-fly, either by interpolation (for $l_t$, $\varepsilon$, $C_{ij}$, and $\overline{\bm u}$) or by calculation (for $q^n$).

The `STGInflow.dat` file is a table of values at given distances from the wall.
These values are then interpolated to a physical location (node or quadrature point). It has the following format:
```
[Total number of locations] 14
[d_w] [u_1] [u_2] [u_3] [R_11] [R_22] [R_33] [R_12] [R_13] [R_23] [sclr_1] [sclr_2] [l_t] [eps]
```
where each `[  ]` item is a number in scientific notation (ie. `3.1415E0`), and `sclr_1` and `sclr_2` are reserved for turbulence modeling variables.
They are not used in this example.

The `STGRand.dat` file is the table of the random number set, $\{\bm{\sigma}^n, \bm{d}^n, \phi^n\}_{n=1}^N$.
It has the format:
```
[Number of wavemodes] 7
[d_1] [d_2] [d_3] [phi] [sigma_1] [sigma_2] [sigma_3]
```

The following table is presented to help clarify the dimensionality of the numerous terms in the STG formulation.

| Math                                           | Label    | $f(\bm{x})$?   | $f(n)$?   |
| -----------------                              | -------- | -------------- | --------- |
| $ \{\bm{\sigma}^n, \bm{d}^n, \phi^n\}_{n=1}^N$ | RN Set   | No             | Yes       |
| $\bm{\overline{u}}$                            | ubar     | Yes            | No        |
| $U_0$                                          | U0       | No             | No        |
| $l_t$                                          | l_t      | Yes            | No        |
| $\varepsilon$                                  | eps      | Yes            | No        |
| $\bm{R}$                                       | R_ij     | Yes            | No        |
| $\bm{C}$                                       | C_ij     | Yes            | No        |
| $q^n$                                          | q^n      | Yes            | Yes       |
| $\{\kappa^n\}_{n=1}^N$                         | k^n      | No             | Yes       |
| $h_i$                                          | h_i      | Yes            | No        |
| $d_w$                                          | d_w      | Yes            | No        |

#### Runtime Options

To use the STG boundary condition, the `-bc_inflow` option should be set to the boundary faces that need the inflow (see {ref}`bc-flags` and above).
The `-stg_use` flag is then used to enable/disable applying STG to those faces.

:::{list-table} STG Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-stg_use`
  - Enable STG for `bc_inflow` faces
  - `false`
  -

* - `-stg_inflow_path`
  - Path to the STGInflow file
  - `./STGInflow.dat`
  -

* - `-stg_rand_path`
  - Path to the STGRand file
  - `./STGRand.dat`
  -

* - `-stg_alpha`
  - Growth rate of the wavemodes
  - `1.01`
  -

* - `-stg_u0`
  - Convective velocity, $U_0$
  - `0.0`
  - `m/s`

* - `-stg_mean_only`
  - Only impose the mean velocity (no fluctutations)
  - `false`
  -

* - `-stg_strong`
  - Strongly enforce the STG inflow boundary condition
  - `false`
  -

* - `-stg_fluctuating_IC`
  - "Extrude" the fluctuations through the domain as an initial condition
  - `false`
  -

* - `-stg_dx`
  - Set the element size in the x direction. Default is calculated for box meshes, assuming equispaced elements.
  -
  - `m`

* - `-stg_h_scale_factor`
  - Scale element size for cutoff frequency calculation
  - $1/p$
  -
:::

### Internal Damping Layer (IDL)
:::{note}
IDL is not a boundary condition, but it's primary application is for use with STG.
:::
The STG inflow boundary condition creates large amplitude acoustic waves.
We use an internal damping layer (IDL) to damp them out without disrupting the synthetic structures developing into natural turbulent structures.
This implementation was inspired by {cite}`shurSTG2014`, but is implemented here as a ramped volumetric forcing term, similar to a sponge layer (see 8.4.2.4 in {cite}`colonius2023turbBC` for example).
It takes the following form:

$$
S(\bm{q}) = -\sigma(\bm{x})\left.\frac{\partial \bm{q}}{\partial \bm{Y}}\right\rvert_{\bm{q}} \bm{Y}'
$$

where $\bm{Y}' = [P - P_\mathrm{ref}, \bm{0}, 0]^T$, and $\sigma(\bm{x})$ is a linear ramp starting at `-idl_start` with length `-idl_length` and an amplitude of inverse `-idl_decay_rate`.
The damping is defined in terms of a pressure-primitive anomaly $\bm Y'$ converted to conservative source using $\partial \bm{q}/\partial \bm{Y}\rvert_{\bm{q}}$, which is linearized about the current flow state.
$P_\mathrm{ref}$ has a default value equal to `-reference_pressure` flag, with an optional flag `-idl_pressure` to set it to a different value.

:::{list-table} IDL Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-idl_decay_time`
  - Characteristic timescale of the pressure deviance decay. The timestep is good starting point
  - `-1` (disabled)
  - `s`

* - `-idl_start`
  - Start of IDL in the x direction
  - `0`
  - `m`

* - `-idl_length`
  - Length of IDL in the positive x direction
  - `0`
  - `m`

* - `-idl_pressure`
  - Pressure used for IDL reference pressure
  -  `-reference_pressure`
  - `Pa`
:::
