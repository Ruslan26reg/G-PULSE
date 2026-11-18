# Experimental conditions

Everything listed here is what the code in this repository does. The defining source files are named in brackets.

## 1. Hardware and software

| | |
|---|---|
| Hardware | Linux virtual machine (KVM): 2 vCPU Intel Xeon @ 2.10 GHz (x86-64 with AVX-512; family 6, model 207), 7.8 GB RAM |
| GPU | none: all computations run on the CPU |
| Parallelism | one BLAS thread per run (`OMP_NUM_THREADS=1`), two runs in parallel |
| Operating system | Ubuntu 24.04.5 LTS, Linux kernel 6.18 |
| Python | 3.13.16 |
| Libraries | NumPy 2.5.3 with its bundled OpenBLAS 0.3.34 (scipy-openblas64, kernel SkylakeX); SciPy 1.18.1 (reference solvers, statistics); Matplotlib 3.11.2 (figures); threadpoolctl 3.7.0 (environment report only) |
| Deep-learning framework | none: own reverse-mode automatic differentiation (`autodiff.py`) and forward Taylor-mode input derivatives (`networks.py`) |
| Arithmetic | float64 throughout |
| Compute | 300 logged runs, 52.6 CPU-hours of training; learning-rate tuning about 25 CPU-hours more |

The runs were made over several sessions on virtual machines of this type. The CPU of each session was not recorded
individually. The logs were re-verified on such a machine, on which retraining reproduces them bit for bit
(`results/paper/environment_verification.json`, `scripts/check_paper.py`).

## 2. Problems and data (`problems.py`, `reference.py`)

| Problem | Equation (nondimensional) | Domain | Delay / nonlocal term |
|---|---|---|---|
| Pantograph–catenary (Volterra) | y(t) − ∫₀ᵗ [−y(s) + 0.8 y(s/3)] ds = f(t) | t ∈ [0, 1] | y(s/3) |
| Cooling fin (Fredholm) | θ(x) − ∫₀¹ e^{−\|x−s\|} [0.5 θ(s) + 0.5 θ(s/2)] ds = f(x) | x ∈ [0, 1] | θ(s/2) |
| Graded bar (parabolic) | θ_t = (κθ_x)_x − θ − 10 θ(x, t−τ), κ = 1 + 3.1(1−x)², θ = 0 at x = 0, 1, history θ = f₀(x) | [0,1] × [0, 0.04] | τ = 0.01 |
| Tapered cable (hyperbolic) | μ w_tt = w_xx − 20 μ w(x, t−τ), μ = 1 + x/2, w = 0 at x = 0, 1, history w = f₀ + t g₁ | [0,1] × [0, 0.3] | τ = 0.1 |
| Cold plate (elliptic) | (1+x/2) u_xx + u_yy + u_x/2 − (1+x) u + 8 u(x/2, y) = −Φ, u = 0 on the boundary | (0,1)² | u(x/2, y) |

* **Input functions.** These are function 1000 of a random stream (`numpy.random.default_rng(2026)`, 1200 functions per
  problem); their coefficients are stored in `data/<problem>.npz`.
  * Integral equations: f = Σ_{k≤8} a_k cos kπx + b_k sin kπx, with a_k, b_k ~ N(0, (1+k)^{−3}).
  * Bar: f₀ = Σ_{k≤4} b_k sin kπx, with b_k ~ N(0, (1+k)^{−4}); the history is constant.
  * Cable: f₀ and g₁ = 4 · 1024 [x(1−x)]⁵ Σ_{k≤6} b_k sin kπx, with b_k ~ N(0, (1+k)^{−2}).
  * Plate: Φ = Σ_{i,j≤4} (A_ij cos iπx + B_ij sin iπx) cos jπy, with A_ij, B_ij ~ N(0, (1+i+j)^{−3}).
* **Residual scaling.** The residual is multiplied by T for the bar and by T² for the cable (time written as t/T).
* **Integrals.** They use 24-point Gauss–Legendre quadrature. For the fin the integral is split at s = x, where the
  kernel has a kink.
* **Reference solutions.** They are used only to measure errors. All use Chebyshev–Lobatto collocation:
  * Volterra and Fredholm equations: N = 160;
  * bar and cable: all discrete Sturm–Liouville modes (N = 256 and 384), with the exact delay response of each mode;
  * plate: N = 96 × 96.

  `scripts/make_data.py` recomputes them from scratch and agrees with `data/` to about 10⁻¹⁵.

## 3. Networks and trial functions (`networks.py`, `config.py`)

| | PINN | Chebyshev-PIKAN |
|---|---|---|
| Hidden layers | 4 × 128, tanh | 3 KAN layers of width 32: y_o = Σ_{i,k} C_oik T_k(tanh x_i), degree 5 |
| Inputs | z ∈ [−1,1]^d with Fourier features sin(πfz), cos(πfz). f ∈ {1,2,4,8} for pantograph, fin and plate; {1,2,4,8,16} for the bar; {1,2,4} for the cable. Bar only: time features exp(−a t/T), a ∈ {4,16,64} | z ∈ [−1,1]^d |
| Initialisation | Xavier normal, std √(2/(n_in+n_out)); zero biases | C ~ N(0, 1/(n_in(K+1))) |
| Least-squares features | 128 last-layer activations + constant | T₁..T₅(tanh h) of the 32 last-layer outputs (160) + constant |
| Trainable parameters (incl. head) | 50,945 (integral), 52,993 (bar), 51,585 (cable), 52,097 (plate) | 12,641 (integral), 12,833 (PDEs) |

Hard constraints (trial functions):
* integral equations: u = N;
* bar: u = f₀ + (t/T) x(1−x) N;
* cable: u = f₀ + t g₁ + (t/T)² x(1−x) N;
* plate: u = 16 x(1−x) y(1−y) N.

## 4. Training (`varpro.py`, `config.py`)

| Setting | Integral equations | PDEs |
|---|---|---|
| Iterations | 1500 | 4000 |
| Collocation points per step (uniform, resampled each step) | 256 | 1024 |
| Least-squares head | ridge μ_rel = 10⁻¹² | μ_rel = 10⁻¹² (bar); PINN 10⁻¹⁰ and PIKAN 10⁻⁸ (cable, plate) |
| Split head (head and loss on different batches) | no | plate (both networks), cable with the PIKAN |
| Validation points (fixed, `default_rng(12345)`) | 256 | 512 |
| Points for the final head (`default_rng(777)`) | 1024 | 4096 |

* **Each step.** The features are computed once. The head is solved exactly: c* = argmin ‖Jᵀc + r₀‖² + μ‖c‖², with column
  scaling, by QR least squares (`numpy.linalg.lstsq`). The loss mean(r(c*)²) is differentiated with respect to the hidden
  parameters with c* fixed (envelope theorem). The gradient is clipped to a global norm of 5.
* **Schedule.** Cosine: lr_t = lr · [0.02 + 0.49 (1 + cos(π t / t_max))].
* **Checkpoint.** The validation residual is computed at iteration 1, every 100 iterations and at the end. The
  checkpoint with the smallest validation residual is restored, and its head is recomputed on the final batch.
* **Seeds.** Seed s sets the collocation sampling (`default_rng(s)`) and the initialisation (`default_rng(100+s)`),
  identically for every optimizer. Seeds 0–4 are used.

## 5. Optimizers (`optimizers.py`)

All hyperparameters except the learning rate are fixed. t is the step counter, and g is the gradient after clipping.

| Optimizer | Update and fixed hyperparameters |
|---|---|
| AdamW | β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸ (added to √v̂), bias correction, decoupled weight decay 10⁻⁶ |
| Yogi | β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸; v ← v − (1−β₂) sign(v − g²) g², with v₀ = 0; step lr · m̂ / (√max(v̂, ε) + ε); bias correction; no weight decay |
| AdaBelief | β₁ = 0.9, β₂ = 0.999, ε = 10⁻¹² (added to √ŝ), s from (g − m)², bias correction, no weight decay |
| Muon | Nesterov momentum 0.95. Five Newton–Schulz iterations with coefficients (3.4445, −4.7750, 2.0315) on the Frobenius-normalised matrix (ε = 10⁻⁸). Update scaled by √max(1, rows/cols) and clipped to norm 1. Decoupled weight decay 10⁻⁶. KAN coefficient tensors [n_out, n_in, K+1] are matricised to [n_out, n_in(K+1)]. Vectors (biases) are updated by Adam (β = 0.9/0.999, ε = 10⁻⁸, weight decay 10⁻⁶) |
| G-PULSE (`strider`) | β₀ = 0.7 (positive–negative momentum), β₁ = 0.9, β₂ = 0.999, β₃ = 0.95, γ = 0.05, ε = 10⁻¹², δ = 10⁻⁸. Each step: m ← β₁m + (1−β₁)g; q = (1+β₀)m − β₀m_prev + γ(g − g_prev); u ← β₂u + (1−β₂)g²; u_max = max(u_max, u); d = q / (u_max + δ/t + ε); r = √(mean u_max + ε) · d; s ← β₃s + (1−β₃) r²; update = lr / (√mean s + ε) · r, clipped to norm 1 |
| G-PULSE-O (`strider_o`) | as G-PULSE, but for every matrix and every matricised KAN tensor r is replaced by 5 Newton–Schulz iterations of it (same coefficients, ε = 10⁻⁷) before s and the update are formed |

**Learning-rate protocol** (`experiments.py`), identical for every optimizer:
1. Initial grid with a quarter of the budget (375 or 1000 iterations, validation every 25 iterations):
   {3, 10, 30, 100}·10⁻⁴, and {1, 3, 10, 30}·10⁻³ for Muon. The rate with the smallest validation residual of the final
   model is selected. No reference solution is used.
2. If the selected rate is on the edge of the grid, the grid is extended in that direction (×3; ×10 for Yogi), by up to
   six points, while the residual improves.
3. A full-budget run of the selected rate (seed 0). If its final residual is more than 3× the tuning residual, the
   second-best rate is also run, and the one with the lower residual is kept.

The selected rates and the residual of every rate tried are in `results/paper/selected_lr.json`.

## 6. Evaluation and statistics (`varpro.py`, `metrics.py`, `stats.py`)

* **Error measures.**
  * Relative L² error, RMSE and MSE on 129 Chebyshev–Lobatto nodes (plate: 49 × 49) with Clenshaw–Curtis weights, at
    t = T for the bar and the cable.
  * Relative H¹ error by spectral differentiation.
  * Space–time error over 11 times in [0, T].
  * Engineering quantities of interest.
* **Statistics.** All on log₁₀ L² over five seeds. A failed run (L² ≥ 1) enters with L² = 1.
  * Geometric means with 95% t-intervals.
  * Seed-paired t-tests with the Holm correction: 40 comparisons for G-PULSE and 50 for G-PULSE-O.
  * The Friedman test with the Nemenyi critical difference (α = 0.05).
* **Timing.** `scripts/time_benchmark.py` measures all optimizers in one process under identical conditions: 50
  iterations after the first, three repeats with rotated order.
