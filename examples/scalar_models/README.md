# Scalar model examples

Scalar φ⁴ lattice examples built with `normflow`. Most follow the model in
[arXiv:2301.01504], combining a PSD (power-spectral-density) flow with a
stack of coupling layers; they differ mainly in which coupling transformation
is used.

- **`scalar_1dof.py`** -- Minimal single-degree-of-freedom example: inverse
  transform sampling with a rational quadratic spline (RQS) for a quartic
  action.

- **`scalar_affine_coupling.py`** -- Affine coupling layers only, no PSD
  flow or spline activations.

- **`scalar_psd.py`** -- PSD flow only (mean-field + FFT-based), no
  coupling layers.

- **`scalar_psd_affine_coupling.py`** -- PSD flow + affine coupling layers.
  The reference implementation for the naming/structure the other coupling
  variants below follow.

- **`scalar_psd_pade32a_coupling.py`** -- Same as
  `scalar_psd_affine_coupling.py`, but with a Pade [3/2]-based coupling
  (`Pade32aCoupling_`) instead of affine coupling.

- **`scalar_psd_spline_coupling.py`** -- Same as
  `scalar_psd_affine_coupling.py`, but with a rational-quadratic-spline
  coupling (`RQSplineCoupling_`) instead of affine coupling.

- **`scalar_fibo_autoreg.py`** -- Autoregressive model for large lattices,
  using a Fibonacci tiling splitter to keep the number of layers growing
  only logarithmically with lattice size.

Each file can be run directly (`python3 <file>.py`) to train with default
options, and supports `torchrun --nproc_per_node=N <file>.py` for
data-parallel training across `N` processes on one node.
