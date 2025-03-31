# Auxiliary Functionality
This section documents functionality that is not apart of the core PDE solver, but is used for other miscellaneous tasks, such as statistics collection or in-situ machine learning.

(aux-statistics)=
## Statistics Collection
For scale-resolving simulations (such as LES and DNS), statistics for a simulation are more often useful than time-instantaneous snapshots of the simulation itself.
To make this process more computationally efficient, averaging in the spanwise direction, if physically correct, can help reduce the amount of simulation time needed to get converged statistics.

First, let's more precisely define what we mean by spanwise average.
Denote $\langle \phi \rangle$ as the Reynolds average of $\phi$, which in this case would be a average over the spanwise direction and time:

$$
\langle \phi \rangle(x,y) = \frac{1}{L_z + (T_f - T_0)}\int_0^{L_z} \int_{T_0}^{T_f} \phi(x, y, z, t) \mathrm{d}t \mathrm{d}z
$$

where $z$ is the spanwise direction, the domain has size $[0, L_z]$ in the spanwise direction, and $[T_0, T_f]$ is the range of time being averaged over.
Note that here and in the code, **we assume the spanwise direction to be in the $z$ direction**.

To discuss the details of the implementation we'll first discuss the spanwise integral, then the temporal integral, and lastly the statistics themselves.

### Spanwise Integral
The function $\langle \phi \rangle (x,y)$ is represented on a 2-D finite element grid, taken from the full domain mesh itself.
If isoperiodicity is set, the periodic face is extracted as the spanwise statistics mesh.
Otherwise the negative z face is used.
We'll refer to this mesh as the *parent grid*, as for every "parent" point in the parent grid, there are many "child" points in the full domain.
Define a function space on the parent grid as $\mathcal{V}_p^\mathrm{parent} = \{ \bm v(\bm x) \in H^{1}(\Omega_e^\mathrm{parent}) \,|\, \bm v(\bm x_e(\bm X)) \in P_p(\bm{I}), e=1,\ldots,N_e \}$.
We enforce that the order of the parent FEM space is equal to the full domain's order.

Many statistics are the product of 2 or more solution functions, which results in functions of degree higher than the parent FEM space, $\mathcal{V}_p^\mathrm{parent}$.
To represent these higher-order functions on the parent FEM space, we perform an $L^2$ projection.
Define the spanwise averaged function as:

$$
\langle \phi \rangle_z(x,y,t) = \frac{1}{L_z} \int_0^{L_z} \phi(x, y, z, t) \mathrm{d}z
$$

where the function $\phi$ may be the product of multiple solution functions and $\langle \phi \rangle_z$ denotes the spanwise average.
The projection of a function $u$ onto the parent FEM space would look like:

$$
\bm M u_N = \int_0^{L_x} \int_0^{L_y} u \psi^\mathrm{parent}_N \mathrm{d}y \mathrm{d}x
$$
where $\bm M$ is the mass matrix for $\mathcal{V}_p^\mathrm{parent}$, $u_N$ the coefficients of the projected function, and $\psi^\mathrm{parent}_N$ the basis functions of the parent FEM space.
Substituting the spanwise average of $\phi$ for $u$, we get:

$$
\bm M [\langle \phi \rangle_z]_N = \int_0^{L_x} \int_0^{L_y} \left [\frac{1}{L_z} \int_0^{L_z} \phi(x,y,z,t) \mathrm{d}z \right ] \psi^\mathrm{parent}_N(x,y) \mathrm{d}y \mathrm{d}x
$$

The triple integral in the right hand side is just an integral over the full domain

$$
\bm M [\langle \phi \rangle_z]_N = \frac{1}{L_z} \int_\Omega \phi(x,y,z,t) \psi^\mathrm{parent}_N(x,y) \mathrm{d}\Omega
$$

We need to evaluate $\psi^\mathrm{parent}_N$ at quadrature points in the full domain.
To do this efficiently, **we assume and exploit the full domain grid to be a tensor product in the spanwise direction**.
This assumption means quadrature points in the full domain have the same $(x,y)$ coordinate location as quadrature points in the parent domain.
This also allows the use of the full domain quadrature weights for the triple integral.

### Temporal Integral/Averaging
To calculate the temporal integral, we do a running average using left-rectangle rule.
At the beginning of each simulation, the time integral of a statistic is set to 0, $\overline{\phi} = 0$.
Periodically, the integral is updated using left-rectangle rule:

$$\overline{\phi}_\mathrm{new} = \overline{\phi}_{\mathrm{old}} + \phi(t_\mathrm{new}) \Delta T$$
where $\phi(t_\mathrm{new})$ is the statistic at the current time and $\Delta T$ is the time since the last update.
When stats are written out to file, this running sum is then divided by $T_f - T_0$ to get the time average.

With this method of calculating the running time average, we can plug this into the $L^2$ projection of the spanwise integral:

$$
\bm M [\langle \phi \rangle]_N = \frac{1}{L_z + (T_f - T_0)} \int_\Omega \int_{T_0}^{T_f} \phi(x,y,z,t) \psi^\mathrm{parent}_N \mathrm{d}t \mathrm{d}\Omega
$$
where the integral $\int_{T_0}^{T_f} \phi(x,y,z,t) \mathrm{d}t$ is calculated on a running basis.


### Running
As the simulation runs, it takes a running time average of the statistics at the full domain quadrature points.
This running average is only updated at the interval specified by `-ts_monitor_spanstats_turbulence_collect_interval` as number of timesteps.
The $L^2$ projection problem is only solved when statistics are written to file, which is controlled by `-ts_monitor_spanstats_turbulence_interval`.
Note that the averaging is not reset after each file write.
The average is always over the bounds $[T_0, T_f]$, where $T_f$ in this case would be the time the file was written at and $T_0$ is the solution time at the beginning of the run.

:::{list-table} Spanwise Turbulent Statistics Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value

* - `-ts_monitor_spanstats_turbulence_collect_interval`
  - Number of timesteps between statistics collection
  - `1`

* - `-ts_monitor_spanstats_turbulence`
  - Sets the `PetscViewer` for the statistics file writing, such as `cgns:output-%d.cgns` (requires PETSc `--download-cgns`). Also turns the statistics collection on.
  -

* - `-ts_monitor_spanstats_turbulence_interval`
  - Number of timesteps between statistics file writing (`-1` means only at end of run)
  - `-1`

* - `-ts_monitor_spanstats_turbulence_cgns_batch_size`
  - Number of frames written per CGNS file if the CGNS file name includes a format specifier (`%d`).
  - `20`
:::

### Turbulent Statistics

The focus here are those statistics that are relevant to turbulent flow.
The terms collected are listed below, with the mathematical definition on the left and the label (present in CGNS output files) is on the right.

| Math                           | Label                           |
| -----------------              | --------                        |
| $\langle \rho \rangle$         | MeanDensity                     |
| $\langle p \rangle$            | MeanPressure                    |
| $\langle p^2 \rangle$          | MeanPressureSquared             |
| $\langle p u_i \rangle$        | MeanPressureVelocity[$i$]       |
| $\langle \rho T \rangle$       | MeanDensityTemperature          |
| $\langle \rho T u_i \rangle$   | MeanDensityTemperatureFlux[$i$] |
| $\langle \rho u_i \rangle$     | MeanMomentum[$i$]               |
| $\langle \rho u_i u_j \rangle$ | MeanMomentumFlux[$ij$]          |
| $\langle u_i \rangle$          | MeanVelocity[$i$]               |

where [$i$] are suffixes to the labels. So $\langle \rho u_x u_y \rangle$ would correspond to MeanMomentumFluxXY.
This naming convention is chosen to align with the CGNS standard naming style.

To get second-order statistics from these terms, simply use the identity:

$$
\langle \phi' \theta' \rangle = \langle \phi \theta \rangle - \langle \phi \rangle \langle \theta \rangle
$$

(aux-differential-filtering)=
## Differential Filtering

There is the option to filter the solution field using differential filtering.
This was first proposed in {cite}`germanoDiffFilterLES1986`, using an inverse Hemholtz operator.
The strong form of the differential equation is

$$
\overline{\phi} - \nabla \cdot (\beta (\bm{D}\bm{\Delta})^2 \nabla \overline{\phi} ) = \phi
$$

for $\phi$ the scalar solution field we want to filter, $\overline \phi$ the filtered scalar solution field, $\bm{\Delta} \in \mathbb{R}^{3 \times 3}$ a symmetric positive-definite rank 2 tensor defining the width of the filter, $\bm{D}$ is the filter width scaling tensor (also a rank 2 SPD tensor), and $\beta$ is a kernel scaling factor on the filter tensor.
This admits the weak form:

$$
\int_\Omega \left( v \overline \phi + \beta \nabla v \cdot (\bm{D}\bm{\Delta})^2 \nabla \overline \phi \right) \,d\Omega
- \cancel{\int_{\partial \Omega} \beta v \nabla \overline \phi \cdot (\bm{D}\bm{\Delta})^2 \bm{\hat{n}} \,d\partial\Omega} =
\int_\Omega v \phi \, , \; \forall v \in \mathcal{V}_p
$$

The boundary integral resulting from integration-by-parts is crossed out, as we assume that $(\bm{D}\bm{\Delta})^2 = \bm{0} \Leftrightarrow \overline \phi = \phi$ at boundaries (this is reasonable at walls, but for convenience elsewhere).

### Filter Width Tensor, Δ
For homogenous filtering, $\bm{\Delta}$ is defined as the identity matrix.

:::{note}
It is common to denote a filter width dimensioned relative to the radial distance of the filter kernel.
Note here we use the filter *diameter* instead, as that feels more natural (albeit mathematically less convenient).
For example, under this definition a box filter would be defined as:

$$
B(\Delta; \bm{r}) =
\begin{cases}
1 & \Vert \bm{r} \Vert \leq \Delta/2 \\
0 & \Vert \bm{r} \Vert > \Delta/2
\end{cases}
$$
:::

For inhomogeneous anisotropic filtering, we use the finite element grid itself to define $\bm{\Delta}$.
This is set via `-diff_filter_grid_based_width`.
Specifically, we use the filter width tensor defined in {cite}`prakashDDSGSAnisotropic2022`.
For finite element grids, the filter width tensor is most conveniently defined by $\bm{\Delta} = \bm{g}^{-1/2}$ where $\bm g = \nabla_{\bm x} \bm{X} \cdot \nabla_{\bm x} \bm{X}$ is the metric tensor.

### Filter Width Scaling Tensor, $\bm{D}$
The filter width tensor $\bm{\Delta}$, be it defined from grid based sources or just the homogenous filtering, can be scaled anisotropically.
The coefficients for that anisotropic scaling are given by `-diff_filter_width_scaling`, denoted here by $c_1, c_2, c_3$.
The definition for $\bm{D}$ then becomes

$$
\bm{D} =
\begin{bmatrix}
    c_1 & 0        & 0        \\
    0        & c_2 & 0        \\
    0        & 0        & c_3 \\
\end{bmatrix}
$$

In the case of $\bm{\Delta}$ being defined as homogenous, $\bm{D}\bm{\Delta}$ means that $\bm{D}$ effectively sets the filter width.

The filtering at the wall may also be damped, to smoothly meet the $\overline \phi = \phi$ boundary condition at the wall.
The selected damping function for this is the van Driest function {cite}`vandriestWallDamping1956`:

$$
\zeta = 1 - \exp\left(-\frac{y^+}{A^+}\right)
$$

where $y^+$ is the wall-friction scaled wall-distance ($y^+ = y u_\tau / \nu = y/\delta_\nu$), $A^+$ is some wall-friction scaled scale factor, and $\zeta$ is the damping coefficient.
For this implementation, we assume that $\delta_\nu$ is constant across the wall and is defined by `-diff_filter_friction_length`.
$A^+$ is defined by `-diff_filter_damping_constant`.

To apply this scalar damping coefficient to the filter width tensor, we construct the wall-damping tensor from it.
The construction implemented currently limits damping in the wall parallel directions to be no less than the original filter width defined by $\bm{\Delta}$.
The wall-normal filter width is allowed to be damped to a zero filter width.
It is currently assumed that the second component of the filter width tensor is in the wall-normal direction.
Under these assumptions, $\bm{D}$ then becomes:

$$
\bm{D} =
\begin{bmatrix}
    \max(1, \zeta c_1) & 0         & 0                  \\
    0                  & \zeta c_2 & 0                  \\
    0                  & 0         & \max(1, \zeta c_3) \\
\end{bmatrix}
$$

### Filter Kernel Scaling, β
While we define $\bm{D}\bm{\Delta}$ to be of a certain physical filter width, the actual width of the implied filter kernel is quite larger than "normal" kernels.
To account for this, we use $\beta$ to scale the filter tensor to the appropriate size, as is done in {cite}`bullExplicitFilteringExact2016`.
To match the "size" of a normal kernel to our differential kernel, we attempt to have them match second order moments with respect to the prescribed filter width.
To match the box and Gaussian filters "sizes", we use $\beta = 1/10$ and $\beta = 1/6$, respectively.
$\beta$ can be set via `-diff_filter_kernel_scaling`.

### Runtime Options

:::{list-table} Differential Filtering Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-diff_filter_monitor`
  - Enable differential filter TSMonitor
  - `false`
  - boolean

* - `-diff_filter_grid_based_width`
  - Use filter width based on the grid size
  - `false`
  - boolean

* - `-diff_filter_width_scaling`
  - Anisotropic scaling for filter width in wall-aligned coordinates (snz)
  - `1,1,1`
  - `m`

* - `-diff_filter_kernel_scaling`
  - Scaling to make differential kernel size equivalent to other filter kernels
  - `0.1`
  - `m^2`

* - `-diff_filter_wall_damping_function`
  - Damping function to use at the wall for anisotropic filtering (`none`, `van_driest`)
  - `none`
  - string

* - `-diff_filter_wall_damping_constant`
  - Constant for the wall-damping function. $A^+$ for `van_driest` damping function.
  - 25
  -

* - `-diff_filter_friction_length`
  - Friction length associated with the flow, $\delta_\nu$. Used in wall-damping functions
  - 0
  - `m`
:::

(aux-in-situ-ml)=
## *In Situ* Machine-Learning Model Training
Training machine-learning models normally uses *a priori* (already gathered) data stored on disk.
This is computationally inefficient, particularly as the scale of the problem grows and the data that is saved to disk reduces to a small percentage of the total data generated by a simulation.
One way of working around this to to train a model on data coming from an ongoing simulation, known as *in situ* (in place) learning.

This is implemented in the code using [SmartSim](https://www.craylabs.org/docs/overview.html).
Briefly, the fluid simulation will periodically place data for training purposes into a database that a separate process uses to train a model.
The database used by SmartSim is [Redis](https://redis.com/modules/redis-ai/) and the library to connect to the database is called [SmartRedis](https://www.craylabs.org/docs/smartredis.html).
More information about how to utilize this code in a SmartSim configuration can be found on [SmartSim's website](https://www.craylabs.org/docs/overview.html).

To use this code in a SmartSim *in situ* setup, first the code must be built with SmartRedis enabled.
This is done by specifying the installation directory of SmartRedis using the `SMARTREDIS_DIR` environment variable when building:

```
make SMARTREDIS_DIR=~/software/smartredis/install
```

### SGS Data-Driven Model *In Situ* Training
Currently the code is only setup to do *in situ* training for the SGS data-driven model.
Training data is split into the model inputs and outputs.
The model inputs are calculated as the same model inputs in the SGS Data-Driven model described {ref}`earlier<sgs-dd-model>`.
The model outputs (or targets in the case of training) are the subgrid stresses.
Both the inputs and outputs are computed from a filtered velocity field, which is calculated via {ref}`aux-differential-filtering`.
The settings for the differential filtering used during training are described in {ref}`aux-differential-filtering`.
The training will create multiple sets of data per each filter width defined in `-sgs_train_filter_widths`.
Those scalar filter widths correspond to the scaling correspond to $\bm{D} = c \bm{I}$, where $c$ is the scalar filter width.

The SGS *in situ* training can be enabled using the `-sgs_train_enable` flag.
Data can be processed and placed into the database periodically.
The interval between is controlled by `-sgs_train_write_data_interval`.
There's also the choice of whether to add new training data on each database write or to overwrite the old data with new data.
This is controlled by `-sgs_train_overwrite_data`.

The database may also be located on the same node as a MPI rank (collocated) or located on a separate node (distributed).
It's necessary to know how many ranks are associated with each collocated database, which is set by `-smartsim_collocated_database_num_ranks`.

### Runtime Options
:::{list-table} *In Situ* Machine-Learning Training Runtime Options
:header-rows: 1

* - Option
  - Description
  - Default value
  - Unit

* - `-sgs_train_enable`
  - Whether to enable *in situ* training of data-driven SGS model. Require building with SmartRedis.
  - `false`
  - boolean

* - `-sgs_train_write_data_interval`
  - Number of timesteps between writing training data into SmartRedis database
  - `1`
  -

* - `-sgs_train_overwrite_data`
  - Whether new training data should overwrite old data on database
  - `true`
  - boolean

* - `-sgs_train_filter_widths`
  - List of scalar values for different filter widths to calculate for training data
  -
  - `m`

* - `-smartsim_collocated_num_ranks`
  - Number of MPI ranks associated with each collocated database (i.e. ranks per node)
  - `1`
  -
:::
