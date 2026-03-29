# Examples

## Compressible Navier-Stokes Equations

### Gaussian Wave
This test case is taken/inspired by that presented in {cite}`mengaldoCompressibleBC2014`. It is intended to test non-reflecting/Riemann boundary conditions. It's primarily intended for Euler equations, but has been implemented for the Navier-Stokes equations here for flexibility.

The problem has a perturbed initial condition and lets it evolve in time. The initial condition contains a Gaussian perturbation in the pressure field:

$$
\begin{aligned}
\rho &= \rho_\infty\left(1+A\exp\left(\frac{-(\bar{x}^2 + \bar{y}^2)}{2\sigma^2}\right)\right) \\
\bm{U} &= \bm U_\infty \\
E &= \frac{p_\infty}{\gamma -1}\left(1+A\exp\left(\frac{-(\bar{x}^2 + \bar{y}^2)}{2\sigma^2}\right)\right) + \frac{\bm U_\infty \cdot \bm U_\infty}{2\rho_\infty},
\end{aligned}
$$

where $A$ and $\sigma$ are the amplitude and width of the perturbation, respectively, and $(\bar{x}, \bar{y}) = (x-x_e, y-y_e)$ is the distance to the epicenter of the perturbation, $(x_e, y_e)$.
The simulation produces a strong acoustic wave and leaves behind a cold thermal bubble that advects at the fluid velocity.

The boundary conditions are freestream in the x and y directions. When using an HLL (Harten, Lax, van Leer) Riemann solver {cite}`toro2009` (option `-freestream_riemann hll`), the acoustic waves exit the domain cleanly, but when the thermal bubble reaches the boundary, it produces strong thermal oscillations that become acoustic waves reflecting into the domain.
This problem can be fixed using a more sophisticated Riemann solver such as HLLC {cite}`toro2009` (option `-freestream_riemann hllc`, which is default), which is a linear constant-pressure wave that transports temperature and transverse momentum at the fluid velocity.

#### Running
:::{list-table} Gaussian Wave Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-epicenter`
  - Coordinates of center of perturbation
  - `0,0,0`
  - `m`

* - `-amplitude`
  - Amplitude of the perturbation
  - `0.1`
  -

* - `-width`
  - Width parameter of the perturbation
  - `0.002`
  - `m`

:::

This problem can be run with the `examples/gaussianwave.yaml` file via:

```
./build/navierstokes -options_file examples/gaussianwave.yaml
```

`examples/gaussianwave.yaml`:
```{literalinclude} ../examples/gaussianwave.yaml
:language: yaml
```

### Vortex Shedding - Flow past Cylinder
This test case, based on {cite}`shakib1991femcfd`, is an example of using an externally provided mesh from Gmsh.
A cylinder with diameter $D=1$ is centered at $(0,0)$ in a computational domain $-4.5 \leq x \leq 15.5$, $-4.5 \leq y \leq 4.5$.
We solve this as a 3D problem with (default) one element in the $z$ direction.
The domain is filled with an ideal gas at rest (zero velocity) with temperature 24.92 and pressure 7143.
The viscosity is 0.01 and thermal conductivity is 14.34 to maintain a Prandtl number of 0.71, which is typical for air.
At time $t=0$, this domain is subjected to freestream boundary conditions at the inflow (left) and Riemann-type outflow on the right, with exterior reference state at velocity $(1, 0, 0)$ giving Reynolds number $100$ and Mach number $0.01$.
A symmetry (adiabatic free slip) condition is imposed at the top and bottom boundaries $(y = \pm 4.5)$ (zero normal velocity component, zero heat-flux).
The cylinder wall is an adiabatic (no heat flux) no-slip boundary condition.
As we evolve in time, eddies appear past the cylinder leading to a vortex shedding known as the vortex street, with shedding period of about 6.

The Gmsh input file, `examples/meshes/cylinder.geo` is parametrized to facilitate experimenting with similar configurations.
The Strouhal number (nondimensional shedding frequency) is sensitive to the size of the computational domain and boundary conditions.

Forces on the cylinder walls are computed using the "reaction force" method, which is variationally consistent with the volume operator.
Given the force components $\bm F = (F_x, F_y, F_z)$ and surface area $S = \pi D L_z$ where $L_z$ is the spanwise extent of the domain, we define the coefficients of lift and drag as

$$
\begin{aligned}
C_L &= \frac{2 F_y}{\rho_\infty u_\infty^2 S} \\
C_D &= \frac{2 F_x}{\rho_\infty u_\infty^2 S} \\
\end{aligned}
$$

where $\rho_\infty, u_\infty$ are the freestream (inflow) density and velocity respectively.

#### Running

The initial condition is taken from `-reference_temperature` and `-reference_pressure`.
To run this problem, first generate a mesh:

```console
$ make -C examples/meshes
```

Then run by building the executable and running:

```console
$ make -j
$ mpiexec -n 6 build/navierstokes -options_file examples/vortexshedding.yaml
```

The vortex shedding period is roughly 5.6 and this problem runs until time 100 (2000 time steps).
The above run writes a file named `force.csv` (see `ts_monitor_wall_force` in `examples/vortexshedding.yaml`), which can be postprocessed by running to create a figure showing lift and drag coefficients over time.

```console
$ python postprocess/vortexshedding.py
```

`examples/vortexshedding.yaml`:
```{literalinclude} ../examples/vortexshedding.yaml
:language: yaml
```

(example-density-current)=
### Density Current

For this test problem (from {cite}`straka1993numerical`), we solve the full Navier-Stokes equations {eq}`eq-ns`, for which a cold air bubble (of radius $r_c$) drops by convection in a neutrally stratified atmosphere.
Its initial condition is defined in terms of the Exner pressure, $\pi(\bm{x},t)$, and potential temperature, $\theta(\bm{x},t)$, that relate to the state variables via

$$
\begin{aligned} \rho &= \frac{P_0}{( c_p - c_v)\theta(\bm{x},t)} \pi(\bm{x},t)^{\frac{c_v}{ c_p - c_v}} \, , \\ e &= c_v \theta(\bm{x},t) \pi(\bm{x},t) + \bm{u}\cdot \bm{u} /2 + g z \, , \end{aligned}
$$

where $P_0$ is the atmospheric pressure.
For this problem, we have used no-slip and non-penetration boundary conditions for $\bm{u}$, and no-flux for mass and energy densities.

#### Running

:::{list-table} Density Current Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-center`
  - Location of bubble center
  - `(lx,ly,lz)/2`
  - `(m,m,m)`

* - `-dc_axis`
  - Axis of density current cylindrical anomaly, or `(0,0,0)` for spherically symmetric
  - `(0,0,0)`
  -

* - `-rc`
  - Characteristic radius of thermal bubble
  - `1000`
  - `m`

* - `-theta0`
  - Reference potential temperature
  - `300`
  - `K`

* - `-thetaC`
  - Perturbation of potential temperature
  - `-15`
  - `K`

* - `-P0`
  - Atmospheric pressure
  - `1E5`
  - `Pa`

* - `-N`
  - Brunt-Vaisala frequency
  - `0.01`
  - `1/s`
:::

This example can be run with:

```
./navierstokes -problem density_current -dm_plex_box_faces 16,1,8 -degree 1 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 2000,125,1000 -dm_plex_dim 3 -rc 400. -bc_wall 1,2,5,6 -wall_comps 1,2,3 -bc_symmetry_y 3,4 -mu 75
```

### Channel

A compressible channel flow. Analytical solution given in
{cite}`whitingStabilizedFEM1999`:

$$ u_1 = u_{\max} \left [ 1 - \left ( \frac{x_2}{H}\right)^2 \right] \quad \quad u_2 = u_3 = 0$$
$$T = T_w \left [ 1 + \frac{Pr \hat{E}c}{3} \left \{1 - \left(\frac{x_2}{H}\right)^4  \right \} \right]$$
$$p = p_0 - \frac{2\rho_0 u_{\max}^2 x_1}{Re_H H}$$

where $H$ is the channel half-height, $u_{\max}$ is the center velocity, $T_w$ is the temperature at the wall, $Pr=\frac{\mu}{c_p \kappa}$ is the Prandlt number, $\hat E_c = \frac{u_{\max}^2}{c_p T_w}$ is the modified Eckert number, and $Re_h = \frac{u_{\max}H}{\nu}$ is the Reynolds number.

Boundary conditions are periodic in the streamwise direction, and no-slip and non-penetration boundary conditions at the walls.
The flow is driven by a body force determined analytically from the fluid properties and setup parameters $H$ and $u_{\max}$.

#### Running

:::{list-table} Channel Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-umax`
  - Maximum/centerline velocity of the flow
  - `10`
  - `m/s`

* - `-theta0`
  - Reference potential temperature
  - `300`
  - `K`

* - `-P0`
  - Atmospheric pressure
  - `1E5`
  - `Pa`

* - `-body_force_scale`
  - Multiplier for body force (`-1` for flow reversal)
  - 1
  -
:::

This problem can be run with the `examples/channel.yaml` file via:

```
./build/navierstokes -options_file examples/channel.yaml
```
`examples/channel.yaml:`
```{literalinclude} ../examples/channel.yaml
:language: yaml
```

### Flat Plate Boundary Layer

#### Meshing

The flat plate boundary layer example has custom meshing features to better resolve the flow when using a generated box mesh.
These meshing features modify the nodal layout of the default, equispaced box mesh and are enabled via `-mesh_transform platemesh`.
One of those is tilting the top of the domain, allowing for it to be a outflow boundary condition.
The angle of this tilt is controlled by `-platemesh_top_angle`.

The primary meshing feature is the ability to grade the mesh, providing better resolution near the wall.
There are two methods to do this; algorithmically, or specifying the node locations via a file.
Algorithmically, a base node distribution is defined at the inlet (assumed to be $\min(x)$) and then linearly stretched/squeezed to match the slanted top boundary condition.
Nodes are placed such that `-platemesh_Ndelta` elements are within `-platemesh_refine_height` of the wall.
They are placed such that the element height matches a geometric growth ratio defined by `-platemesh_growth`.
The remaining elements are then distributed from `-platemesh_refine_height` to the top of the domain linearly in logarithmic space.

Alternatively, a file may be specified containing the locations of each node.
The file should be newline delimited, with the first line specifying the number of points and the rest being the locations of the nodes.
The node locations used exactly at the inlet (assumed to be $\min(x)$) and linearly stretched/squeezed to match the slanted top boundary condition.
The file is specified via `-platemesh_y_node_locs_path`.
If this flag is given an empty string, then the algorithmic approach will be performed.

:::{list-table} Boundary Layer Meshing Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-platemesh_modify_mesh`
  - Whether to modify the mesh using the given options below.
  - `false`
  -

* - `-platemesh_refine_height`
  - Height at which `-platemesh_Ndelta` number of elements should refined into
  - `5.9E-4`
  - `m`

* - `-platemesh_Ndelta`
  - Number of elements to keep below `-platemesh_refine_height`
  - `45`
  -

* - `-platemesh_growth`
  - Growth rate of the elements in the refinement region
  - `1.08`
  -

* - `-platemesh_top_angle`
  - Downward angle of the top face of the domain. This face serves as an outlet.
  - `5`
  - `degrees`

* - `-platemesh_y_node_locs_path`
  - Path to file with y node locations. If empty, will use mesh warping instead.
  - `""`
  -
:::

(example-blasius)=
#### Laminar Boundary Layer - Blasius

Simulation of a laminar boundary layer flow, with the inflow being prescribed by a [Blasius similarity solution](https://en.wikipedia.org/wiki/Blasius_boundary_layer).
At the inflow, the velocity is prescribed by the Blasius soution profile, density is set constant, and temperature is allowed to float.
Using `weakT: true`, density is allowed to float and temperature is set constant.
At the outlet, a user-set pressure is used for pressure in the inviscid flux terms (all other inviscid flux terms use interior solution values).
The wall is a no-slip, no-penetration, no-heat flux condition.
The top of the domain is treated as an outflow and is tilted at a downward angle to ensure that flow is always exiting it.

#### Running
:::{list-table} Blasius Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-velocity_infinity`
  - Freestream velocity
  - `40`
  - `m/s`

* - `-temperature_infinity`
  - Freestream temperature
  - `288`
  - `K`

* - `-pressure_infinity`
  - Atmospheric pressure, also sets IDL reference pressure
  - `1.01E5`
  - `Pa`

* - `-temperature_wall`
  - Wall temperature
  - `288`
  - `K`

* - `-delta0`
  - Boundary layer height at the inflow
  - `4.2e-3`
  - `m`

* - `-weakT`
  - Blasius weak inflow switch (weak temperature vs weak density form)
  - `false`
  - -

* - `-n_chebyshev`
  - Number of Chebyshev terms
  - `20`
  -

* - `-chebyshev_`
  - Prefix for Chebyshev snes solve
  -
  -
:::

This problem can be run with the `examples/blasius.yaml` file via:

```
./build/navierstokes -options_file examples/blasius.yaml
```

`examples/blasius.yaml`:
```{literalinclude} ../examples/blasius.yaml
:language: yaml
```

#### Turbulent Boundary Layer

Simulating a turbulent boundary layer without modeling the turbulence requires resolving the turbulent flow structures.
These structures may be introduced into the simulations either by allowing a laminar boundary layer naturally transition to turbulence, or imposing turbulent structures at the inflow.
The latter approach has been taken here, specifically using a *synthetic turbulence generation* (STG) method.
See {ref}`bc-stg` for details on STG.

#### Running
This problem can be run with the `examples/blasius.yaml` file via:

```
./build/navierstokes -options_file examples/blasius.yaml -stg_use true
```

Note the added `-stg_use true` flag
This overrides the `stg: use: false` setting in the `examples/blasius.yaml` file, enabling the use of the STG inflow.

### Taylor-Green Vortex

This problem is really just an initial condition, the [Taylor-Green Vortex](https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex):

$$
\begin{aligned}
u &= V'_x + V \sin(\hat x) \cos(\hat y) \sin(\hat z) \\
v &= V'_y - V \cos(\hat x) \sin(\hat y) \sin(\hat z) \\
w &= V'_z\\
p &= p_0 + \frac{\rho_0 V_0^2}{16} \left ( \cos(2 \hat x) + \cos(2 \hat y)\right) \left( \cos(2 \hat z) + 2 \right) \\
\rho &= \frac{p}{R T_0} \\
\end{aligned}
$$

where $\hat x = 2 \pi x / L$ for $L$ the length of the domain in that specific direction, $V$ is the magnitude of the vortex velocity, and $V'_i$ is the background velocity that (optionally) advects the initial condition.
The coordinate modification is done to transform a given grid onto a domain of $x,y,z \in [0, 2\pi)$.

This initial condition is traditionally given for the incompressible Navier-Stokes equations.
The reference state is selected using the `-reference_{velocity,pressure,temperature}` flags (Euclidean norm of `-reference_velocity` is used for $V$).

:::{list-table} Taylor-Green Vortex Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-taylorgreen_background_velocity`
  - Background velocity added to Taylor-Green initial condition
  - `0,0,0`
  - `m/s`
:::

## Compressible Euler Equations

(problem-euler-vortex)=

### Isentropic Vortex

Three-dimensional Euler equations, which are simplified and nondimensionalized version of system {eq}`eq-ns` and account only for the convective fluxes, are given by

$$
\begin{aligned}
\frac{\partial \rho}{\partial t} + \nabla \cdot \bm{U} &= 0 \\
\frac{\partial \bm{U}}{\partial t} + \nabla \cdot \left( \frac{\bm{U} \otimes \bm{U}}{\rho} + P \bm{I}_3 \right) &= 0 \\
\frac{\partial E}{\partial t} + \nabla \cdot \left( \frac{(E + P)\bm{U}}{\rho} \right) &= 0 \, , \\
\end{aligned}
$$ (eq-euler)

Following the setup given in {cite}`zhang2011verification`, the mean flow for this problem is $\rho=1$, $P=1$, $T=P/\rho= 1$ (Specific Gas Constant, $R$, is 1), and $\bm{u}=(u_1,u_2,0)$ while the perturbation $\delta \bm{u}$, and $\delta T$ are defined as

$$
\begin{aligned} (\delta u_1, \, \delta u_2) &= \frac{\epsilon}{2 \pi} \, e^{0.5(1-r^2)} \, (-\bar{y}, \, \bar{x}) \, , \\ \delta T &= - \frac{(\gamma-1) \, \epsilon^2}{8 \, \gamma \, \pi^2} \, e^{1-r^2} \, , \\ \end{aligned}
$$

where $(\bar{x}, \, \bar{y}) = (x-x_c, \, y-y_c)$, $(x_c, \, y_c)$ represents the center of the domain, $r^2=\bar{x}^2 + \bar{y}^2$, and $\epsilon$ is the vortex strength ($\epsilon$ < 10).
There is no perturbation in the entropy $S=P/\rho^\gamma$ ($\delta S=0)$.

#### Running
:::{list-table} Isentropic Vortex Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-center`
  - Location of vortex center
  - `(lx,ly,lz)/2`
  - `(m,m,m)`

* - `-mean_velocity`
  - Background velocity vector
  - `(1,1,0)`
  -

* - `-vortex_strength`
  - Strength of vortex < 10
  - `5`
  -

* - `-c_tau`
  - Stabilization constant
  - `0.5`
  -
:::

This problem can be run with:

```
./navierstokes -problem euler_vortex -dm_plex_box_faces 20,20,1 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1000,1000,50 -dm_plex_dim 3 -bc_inflow 4,6 -bc_outflow 3,5 -bc_symmetry_z 1,2 -mean_velocity .5,-.8,0.
```

(problem-shock-tube)=
### Shock Tube

This test problem is based on Sod's Shock Tube (from{cite}`sodshocktubewiki`), a canonical test case for discontinuity capturing in one dimension. For this problem, the three-dimensional Euler equations are formulated exactly as in the Isentropic Vortex problem. The default initial conditions are $P=1$, $\rho=1$ for the driver section and $P=0.1$, $\rho=0.125$ for the driven section. The initial velocity is zero in both sections. Symmetry boundary conditions are applied to the side walls and wall boundary conditions are applied at the end walls.

SU upwinding and discontinuity capturing have been implemented into the explicit timestepping operator for this problem. Discontinuity capturing is accomplished using a modified version of the $YZ\beta$ operator described in {cite}`tezduyar2007yzb`. This discontinuity capturing scheme involves the introduction of a dissipation term of the form

$$
\int_{\Omega} \nu_{SHOCK} \nabla \bm v \!:\! \nabla \bm q dV
$$

The shock capturing viscosity is implemented following the first formulation described in {cite}`tezduyar2007yzb`. The characteristic velocity $u_{cha}$ is taken to be the acoustic speed while the reference density $\rho_{ref}$ is just the local density. Shock capturing viscosity is defined by the following

$$
\nu_{SHOCK} = \tau_{SHOCK} u_{cha}^2
$$

where,

$$
\tau_{SHOCK} = \frac{h_{SHOCK}}{2u_{cha}} \left( \frac{ \,|\, \nabla \rho \,|\, h_{SHOCK}}{\rho_{ref}} \right)^{\beta}
$$

$\beta$ is a tuning parameter set between 1 (smoother shocks) and 2 (sharper shocks). The parameter $h_{SHOCK}$ is a length scale that is proportional to the element length in the direction of the density gradient unit vector. This density gradient unit vector is defined as $\hat{\bm j} = \frac{\nabla \rho}{|\nabla \rho|}$. The original formulation of Tezduyar and Senga relies on the shape function gradient to define the element length scale, but this gradient is not available to qFunctions in libCEED. To avoid this problem, $h_{SHOCK}$ is defined in the current implementation as

$$
h_{SHOCK} = 2 \left( C_{YZB} \,|\, \bm p \,|\, \right)^{-1}
$$

where

$$
p_k = \hat{j}_i \frac{\partial \xi_i}{x_k}
$$

The constant $C_{YZB}$ is set to 0.1 for piecewise linear elements in the current implementation. Larger values approaching unity are expected with more robust stabilization and implicit timestepping.

#### Running
:::{list-table} Shock Tube Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-yzb`
  - Use YZB discontinuity capturing
  - `none`
  -

* - `-stab`
  - Stabilization method (`none`, `su`, or `supg`)
  - `none`
  -

* - `-c_tau`
  - Stabilization constant
  - `0.5`
  -
:::

This problem can be run with:

```
./navierstokes -problem shocktube -yzb -stab su -bc_symmetry_z 3,4 -bc_symmetry_y 1,2 -bc_wall 5,6 -dm_plex_dim 3 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1000,100,100 -dm_plex_box_faces 200,1,1 -units_second 0.1
```

(problem-advection)=
## Advection-Diffusion

There is a reduced mode in HONEE for pure advection-diffusion, which holds density $\rho$ and momentum density $\rho \bm u$ constant while advecting "total energy density" $E$.
This reduced mode is given by

$$
\frac{\partial E}{\partial t} + \nabla \cdot (\bm{u} E ) - \kappa \nabla E = 0 \, ,
$$ (eq-advection)

with $\bm{u}$ the vector velocity field and $\kappa$ the diffusion coefficient.

### Advection Field Options
There are three different definitions for $\bm{u}$:

- **Rotation**

  A uniform circular velocity field transports the blob of total energy.
  We have solved {eq}`eq-advection` applying zero energy density $E$, and no-flux for $\bm{u}$ on the boundaries.

- **Translation**

  In this case, a background wind with a constant rectilinear velocity field, enters the domain and transports the blob of total energy out of the domain.

- **Boundary Layer**

  This case has a linear velocity profile with only the y component set: $u_y = y / L_y$.
  It starts at 0 for $y=0$ and then increases to 1 at the top of the domain.

### Initial Condition options
There are also several different definitions for initial conditions.
Some require specific advection profiles, other's can be used with multiple.
- **Bubble (Sphere and Cylinder)**

  These are simple initial conditions with a controllable radius and center point.
  They use an magnitude of one within the bubble and zero outside and have different smoothing options for the boundary of the bubble with the rest of the domain.
  The difference between sphere and cylinder is whether the radius is applied in all 3 dimensions (sphere) or just in the x and y directions.

- **Cosine Hill**

  This is similar to the bubble ICs, but uses a cosine wave to define the bubble and it's radius is set to half the width of the domain (so the bubble fills the entire domain).

- **Skew**

  This IC is meant for for the translation advection profile only.
  This IC features a line discontinuity intersecting the midpoint of the lower edge of the box and in the same direction as the advection velocity
  The solution is either 0 or 1 on either side of the discontinuity.

- **Wave**

  This IC is meant for for the translation advection profile only.
  This either a sine or square wave that oscillates in the direction of advection velocity.
  The frequency and phase of the wave is controllable.

- **Boundary Layer**

  This IC is meant to be paired with the boundary layer advection profile.
  This initial condition features a linear profile in the y direction up to a height set by the user.

For the inflow boundary conditions, a prescribed $E_{wind}$ is applied weakly on the inflow boundaries such that the weak form boundary integral in {eq}`eq-weak-vector-ns` is defined as

$$
\int_{\partial \Omega_{inflow}} \bm v \cdot \bm{F}(\bm q_N) \cdot \widehat{\bm{n}} \,dS = \int_{\partial \Omega_{inflow}} \bm v \, E_{wind} \, \bm u \cdot \widehat{\bm{n}} \,dS  \, ,
$$

For the outflow boundary conditions, we have used the current values of $E$, following {cite}`papanastasiou1992outflow` which extends the validity of the weak form of the governing equations to the outflow instead of replacing them with unknown essential or natural boundary conditions.
The weak form boundary integral in {eq}`eq-weak-vector-ns` for outflow boundary conditions is defined as

$$
\int_{\partial \Omega_{outflow}} \bm v \cdot \bm{F}(\bm q_N) \cdot \widehat{\bm{n}} \,dS = \int_{\partial \Omega_{outflow}} \bm v \, E \, \bm u \cdot \widehat{\bm{n}} \,dS  \, ,
$$

The advection problems can be run in both 2D and 3D, based on the DM defined for the problem.
The following additional command-line options are available:

### Running

:::{list-table} Advection Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-strong_form`
  - Strong (1) or weak/integrated by parts (0) advection term of the residual
  - `0`
  -

* - `-stab`
  - Stabilization method (`none`, `su`, or `supg`)
  - `none`
  -

* - `-stab_tau`
  - Formulation for $\tau$ in stabilization (`ctau`, `advdiff_shakib`)
  - `ctau`
  -

* - `-Ctau_t`
  - Scaling factor on the temporal portion of the $\tau$ formulation
  - 0.
  -

* - `-Ctau_a`
  - Scaling factor on the advection portion of the $\tau$ formulation
  - $P^2$
  -

* - `-Ctau_d`
  - Scaling factor on the diffusion portion of the $\tau$ formulation
  - $P^4$
  -

* - `-CtauS`
  - Scale coefficient for stabilization tau (nondimensional)
  - `0`
  -

* - `-diffusion_coeff`
  - Diffusion coefficient
  - `0`
  -

* - `-wind_type`
  - Wind type in Advection (`rotation`, `translation`, `boundary_layer`)
  - `rotation`
  -

* - `-wind_translation`
  - Constant wind vector when `-wind_type translation`
  - `1,0,0`
  -

* - `-E_wind`
  - Total energy of inflow wind when `-wind_type translation`
  - `1E6`
  - `J`

* - `-advection_ic_type`
  - Initial condition type, (`sphere`, `cylinder`, `cosine_hill`, `skew`, `wave`, `boundary_layer`)
  - `sphere`
  -

* - `-advection_ic_bubble_rc`
  - For `sphere` or `cylinder` IC, characteristic radius of thermal bubble
  - `1000`
  - `m`

* - `-advection_ic_bubble_continuity`
  - For `sphere` or `cylinder` IC, different shapes of bubble, (`smooth`, `back_sharp`, `thick`, `cosine`)
  - `smooth`
  -

* - `-advection_ic_wave_type`
  - For `wave` IC, the wave form used for `-advection_ic_type wave` (`sine`, `square`)
  - `sine`
  -

* - `-advection_ic_wave_frequency`
  - For `wave` IC, frequency of the wave
  - $2\pi$
  - `1/s`

* - `-advection_ic_wave_phase`
  - For `wave` IC, phase angle of the wave
  - 0
  -

* - `-advection_ic_bl_height_factor`
  - For `boundary_layer` IC, sets the height of the linear boundary layer initial condition in proportion to the domain height
  - 1
  -
:::

For 3D advection, an example of the `rotation` mode can be run with:

```
./navierstokes -problem advection -dm_plex_box_faces 10,10,10 -dm_plex_dim 3 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 8000,8000,8000 -bc_wall 1,2,3,4,5,6 -wall_comps 4 -wind_type rotation -implicit -stab su
```

and the `translation` mode with:

```
./navierstokes -problem advection -dm_plex_box_faces 10,10,10 -dm_plex_dim 3 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 8000,8000,8000 -wind_type translation -wind_translation .5,-1,0 -bc_inflow 1,2,3,4,5,6
```

For 2D advection, an example of the `rotation` mode can be run with:

```
./navierstokes -problem advection -dm_plex_box_faces 20,20 -dm_plex_box_lower 0,0 -dm_plex_box_upper 1000,1000 -bc_wall 1,2,3,4 -wall_comps 4 -wind_type rotation -implicit -stab supg
```

and the `translation` mode with:

```
./navierstokes -problem advection -dm_plex_box_faces 20,20 -dm_plex_box_lower 0,0 -dm_plex_box_upper 1000,1000 -units_meter 1e-4 -wind_type translation -wind_translation 1,-.5 -bc_inflow 1,2,3,4
```
Note the lengths in `-dm_plex_box_upper` are given in meters, and will be nondimensionalized according to `-units_meter`.


The boundary layer problem can be run with:

```
./build/navierstokes -options_file examples/advection_bl.yaml
```

`examples/advection_bl.yaml`:
```{literalinclude} ../examples/advection_bl.yaml
:language: yaml
```

The wave advection problem can be run with:

```
./build/navierstokes -options_file examples/advection_wave.yaml
```

`examples/advection_wave.yaml`:
```{literalinclude} ../examples/advection_wave.yaml
:language: yaml
```
Note that the wave frequency, velocity direction, and domain size are set specifically to allow bi-periodic boundary conditions.
