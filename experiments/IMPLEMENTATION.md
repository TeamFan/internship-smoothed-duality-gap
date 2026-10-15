# Implementation details for the reported experiments

This note records the numerical conventions behind Section 6. It supplements
the mathematical models and acceptance tests in the report; it does not define
additional experiments. Indices in formulas below start at 1, while NumPy
indices start at 0.

## Fixed instances and initial points

The master seed is `20260709`. Each independent generator is created with
`numpy.random.default_rng(numpy.random.SeedSequence(...))`.

For the affine rank-one instance, the substream `[20260709, 1401]` generates
an 8-by-8 standard Gaussian matrix. Its NumPy QR factorization gives `Q, R`;
each column of `Q` is multiplied by the sign of the corresponding diagonal
entry of `R` (zero signs are replaced by 1). This is the matrix `V`.
The spectrum is `(4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5)`.

The affine starts use substream `[20260709, 1441, 8]`. The code first draws
all 160 Gaussian directions and normalizes them, then draws 160 log-uniform
radii between 0.08 and 3.2, then 160 Gaussian multipliers of standard deviation
1.8. The optimization variables are unbounded. The three methods use identical
arrays of initial points, not successive draws from the same generator.

For the box instance, substream `[20260709, 1414, 7]` perturbs the vector
`(1.1, -2, -2.12, -2.24, 1.25, 1.4, 1.55)` by independent uniform values
in `[-2e-4, 2e-4]`. Set `r[0] = -0.22`. The remaining linear coefficients
are recomputed as `r[i] = -q[i] * a[i]`, where the six prescribed stationary
coordinates are `(0.30, 0.335, 0.370, 0.36, 0.44, 0.52)`.
The report describes the curvature and stationary-state structure; this
full-precision construction defines the numerical instance.

The box starts use `scipy.stats.qmc.Sobol(d=8, scramble=True, seed=3184245526)`
and `random_base2(m=8)`. That seed is obtained from
`SeedSequence([20260709, 991, 7]).generate_state(1)[0]`.
The first seven unit coordinates are unchanged; the last is mapped to
`y = -4 + 8*u[7]`. Only the seven primal variables are box-bounded during
optimization; the multiplier is unbounded after initialization.

The arrays in `results/inputs/reported_inputs.npz` preserve `V`, `M`, the
spectrum, `box_q`, `box_r`, and both common initial-point arrays. Their
initial points have been checked elementwise against the retained candidate
tables, including repeated starts across all three methods.

## Local optimizer

All calls use `scipy.optimize.minimize(method="L-BFGS-B", jac=True)`.
The explicitly supplied options are:

| Option | Affine rank one | Box QP |
| --- | --- | --- |
| `maxiter` | 320 | 1000 |
| `maxfun` | 320 | SciPy default (not overridden) |
| `ftol` | `1e-15` | `1e-15` |
| `gtol` | `2e-10` | `2e-10` |
| `maxls` | 40 | 60 |

Other options retain the defaults of the pinned SciPy version. The affine
`maxfun` option is a solver stopping setting, not a measured bound on actual
callback counts. Endpoint acceptance is independent of solver success flags.

## Proximal evaluations and derivatives

For the affine model, let `a = v_1`, `h = a^T x - c`,
`r_p = beta_x*(x-p)`, `ell = grad f(x) + a*y`, and
`H_p = Hessian f(p) + beta_x*I`. The callbacks return

\[
\begin{aligned}
\nabla M_{\rm SDG}
 &= (\ell-r_p+(h/\beta_y)a,\ a^\top(x-p)),\\
\nabla M_{\rm prox}
 &= (\beta_x(r_p-\beta_xH_p^{-1}r_p)+ha,
     \ \beta_xa^\top H_p^{-1}r_p),\\
\nabla M_{\rm KKT}
 &= (\nabla^2f(x)\ell+ha,\ a^\top\ell).
\end{aligned}
\]

The inverse action is evaluated by `numpy.linalg.solve`, not by explicitly
forming an inverse. The proximal residual callback consequently has a
different primitive cost from the SDG or KKT callback. The report compares
branch coverage, not equal-cost or wall-time performance.

The affine proximal point is computed through the scalar secular equation
in Section 6. Starting from `[0, max(1, ||z||^2)]`, the upper endpoint is
doubled until the secular function is nonpositive. The routine raises an
error if the endpoint overflows. For nonzero centers, `brentq` uses
`xtol=1e-13, rtol=2e-14`; the zero center returns zero directly.

For the box proximal and natural-KKT merits, the scalar clipping derivative
selection is `chi(s)=1` if `0<s<1`, and `chi(s)=0` otherwise, including
`s=0` and `s=1`. This is a piecewise derivative convention, not classical
differentiability at a kink. In `box_qp.py`, the arrays `raw` and `shifted`
are the arguments of these projections; the Boolean mask `free` implements
`chi`. Box SC-SDG uses its smooth finite-extension gradient, which does not
differentiate the proximal map.

The branch-search callbacks evaluate the SDG by direct objective differences;
no clipping of negative roundoff values is applied. This revision retains
those evaluations and the recorded search trajectories. Near-zero gap
values may suffer cancellation, so endpoint acceptance uses the independently
recomputed residuals below, not the merit value or optimizer success flag.

## Endpoint acceptance and matching

The maximum of the following three affine residuals must be at most `1e-7`:

\[
\frac{\|Ax-b\|}{1+\|b\|},\qquad
\frac{\|\nabla f(x)+A^\top y\|}
     {1+\|\nabla f(x)\|+\|A^\top y\|},\qquad
\frac{\beta_x\|x-p(x,y)\|}
     {1+\beta_x\|x\|+\|A^\top y\|}.
\]

For the box problem, write \(g_L=\nabla F(x)+e_1y\) and
\(N_B=x-\Pi_B(x-g_L)\). The same threshold is applied to

\[
\begin{aligned}
R_{\rm feas}&=\frac{|x_1-0.37|}{1+0.37},&
R_{\rm stat}&=\frac{\|N_B\|}{1+\|x\|},\\
R_{\rm prox}&=\frac{\beta_x\|x-p_B(x,y)\|}
 {1+\beta_x\|x\|},&
R_{\rm Cprox}&=\frac{\beta_x\|x-P_{1/\beta_x}^{C}(x)\|}
 {1+\beta_x\|x\|}.
\end{aligned}
\]

The box projection residuals are scaled only by the primal point: an
unbounded multiplier must not make a saturated projection residual small.
The box `R_prox` evaluates
the box-only Lagrangian map; `R_Cprox` evaluates the distinct hard-constrained
map. The latter fixes the first coordinate to 0.37. These are not
interchangeable checks.

For each accepted endpoint, the nearest catalog pair is selected using
Euclidean distance in the concatenated primal-multiplier vector. The affine
matching tolerance is `8e-5 * (1 + ||(x,y)||)` at the **computed endpoint**.
The box matching tolerance is the absolute value `2e-5`.
Distinct-root counts are the number of different accepted catalog identifiers;
there is no separate clustering procedure. Each repeated endpoint can
increase the accepted count but cannot increase distinct-root coverage.
Each candidate row stores the initial point, final point, recomputed merit,
residuals, catalog match, and the optimizer's status, success flag and message.
The solver's stopping reason is kept separate from the independent acceptance
decision: budget or line-search termination need not mean a failed residual
test, and nominal solver success does not imply an accepted KKT endpoint.

## Escape diagnostics

The box diagnostic evaluates the analytic formulas in Proposition 6.1 with
`lambda=1/3`; its representative curves are exact full-ball profiles up to
floating-point evaluation. No local outer optimization is used for them. For the IUU, LUU and UUU
representatives, put \(h=-q_2\), \(a=a_2\), \(k=1-\lambda h\). The curves are

\[
\begin{aligned}
m_I(\delta)&=\min\{\delta/k,1-a\},&
m_L(\delta)&=\Pi_{[0,1]}((\delta-\lambda ha)/k),\\
\mathcal R_I(\delta)&=\tfrac h2 m_I(\delta)^2,&
\mathcal R_L(\delta)&=[\tfrac h2m_L(\delta)(m_L(\delta)-2a)]_+,\\
\mathcal R_U(\delta)&=0.
\end{aligned}
\]

The other coordinates are at scalar global minima, so spending any part of
the perturbation radius on them cannot improve the value. These expressions
are full-ball optima, not sampled directional estimates.

For the Hessian-inconclusive radial illustration, `lambda=1/8` and the ten
radii are `delta_j = 10**(-4+j/3)`, `j=0,...,9`, as recorded in
`config/report.json`. The unique scalar proximal root is
computed by `brentq` on `[0, 1-delta]`, with `xtol=1e-14, rtol=2e-14`.
Its derivative is at least `16/5` at the reported step size, and the endpoint
signs bracket the unique root. The CSV also records the stationarity residual.
Stable differences are used: direct `(2*delta-delta**2)**3`, polished
`((1-t)*(1+t))**3`, and envelope `polished-4*(t-1+delta)**2`.
The direction is prescribed analytically. The direct, envelope and polished
values are pathwise witnesses, not computed maxima over a ball.
The plot omits values below `100*eps*(1+abs(q3(e1)))`; this is a reporting
floor, not an interval-arithmetic error certificate. Positive descent follows
from the analytic direct-decrease identity and proximal optimality.

## Environment and scope

The recorded environment is CPython 3.14.0, NumPy 2.3.5, SciPy 1.16.3 and
Matplotlib 3.10.7 on Windows 11. `requirements.txt` pins package versions;
`results/run_metadata.json` records the environment of the retained run.
Results are for one master seed and one Sobol scramble. No multiple-seed
stability or nonsmooth convergence claim follows from these runs.
