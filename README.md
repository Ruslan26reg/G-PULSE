# G-PULSE on engineering problems with delay

Code and logs for the experiments of the paper *G-PULSE: A Gradient-Change Adaptive Optimizer for Physics-Informed
Engineering Problems with Delayed Arguments*: physics-informed networks (PINN and Chebyshev PIKAN) with an exact
least-squares output layer, trained by **G-PULSE**, its orthogonalised variant **G-PULSE-O**, AdamW, Yogi, AdaBelief and
Muon on five engineering problems with delay.

G-PULSE was developed under the working name Strider. The code keeps the identifiers `strider` (G-PULSE) and
`strider_o` (G-PULSE-O) in file names, logs and command-line options, so that the archived runs remain readable.

Everything is plain NumPy/SciPy (float64, CPU); there is no deep-learning framework. Input derivatives are propagated
in forward Taylor mode and parameter gradients come from a small reverse-mode autodiff (about 200 lines), so every step
of the computation can be read in this repository.

| Problem | Type | Equation (nondimensional) |
|---|---|---|
| `epanto` pantograph–catenary | Volterra, proportional delay | y(t) − ∫₀ᵗ [−y(s) + 0.8 y(s/3)] ds = f(t) |
| `efin` cooling fin | Fredholm, proportional delay | θ(x) − ∫₀¹ e^{−\|x−s\|} [0.5 θ(s) + 0.5 θ(s/2)] ds = f(x) |
| `eheat` graded bar | parabolic, constant delay | θ_t = (κθ_x)_x − θ − 10 θ(x, t−0.01), κ = 1 + 3.1(1−x)² |
| `ecable` tapered cable | hyperbolic, constant delay | μ w_tt = w_xx − 20 μ w(x, t−0.1), μ = 1 + x/2 |
| `eplate` cold plate | elliptic, proportional argument | (1+x/2) u_xx + u_yy + u_x/2 − (1+x) u + 8 u(x/2, y) = −Φ |

The full experimental conditions (hardware, libraries, problems, networks, training settings, every optimizer
hyperparameter, tuning protocol, statistics) are listed in [EXPERIMENTS.md](EXPERIMENTS.md). All computations run on
the CPU; no GPU is used.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.13 (3.10 or newer works)
pip install -r requirements-exact.txt                     # the exact versions of the paper's runs
python scripts/env_report.py                              # hardware/software report: is exact reproduction possible?
```

`requirements.txt` lists minimum versions instead. The code then runs, but only `requirements-exact.txt` gives the
paper's numbers bit for bit (see below).

## Exact reproduction

Training amplifies differences in the last bit of floating-point results, so the logged numbers are reproduced exactly
only with the same machine instructions. The paper's environment is:
* Python 3.13 with NumPy 2.5.3 and SciPy 1.18.1, from the pip wheels (`requirements-exact.txt`);
* an x86-64 CPU with AVX-512, for example Intel Xeon Skylake-SP or newer, or AMD Zen 4 or newer;
* Linux; the paper's runs used Ubuntu 24.04.

On such a machine `repro_env.py` fixes the remaining choices automatically: one BLAS thread, the OpenBLAS kernel
SkylakeX (`OPENBLAS_CORETYPE=SkylakeX`) and NumPy SIMD paths up to the AVX-512 level
(`NPY_ENABLE_CPU_FEATURES="X86_V3 X86_V4"`). The scripts and tests import it before NumPy. `scripts/env_report.py` tells
whether your machine qualifies, and `scripts/check_paper.py` confirms it: the maximum difference to the logs is then 0.

On other machines the results agree with the paper statistically, within the spread over seeds, but not bit for bit.
This covers AVX2-only x86 CPUs, Apple Silicon and other operating systems. For example, replacing only the OpenBLAS
kernel (AVX2 instead of AVX-512) changes the validation residual after 100 iterations by up to a factor of 4 on the
integral equations, although every operation is still accurate to about 10⁻¹⁶. All statistical conclusions are drawn
over five seeds for this reason.

On Windows or macOS with an AVX-512 CPU, set the two variables yourself before running the scripts.

## Checking reproducibility (about 10 minutes)

```bash
bash reproduce.sh quick                         # all of the following
# or step by step:
python -m unittest discover -s tests -v        # unit tests + comparison with the paper's logs (~1 min)
python scripts/check_paper.py                   # shipped weights + first 100 iterations of the G-PULSE runs (~5 min)
python scripts/analyze.py                       # all tables of the paper from the logs  -> results/paper/tables/
python scripts/latex_tables.py                  # the same tables in LaTeX (tab_delay_*.tex)
python scripts/time_benchmark.py                # cost per iteration of all optimizers under identical conditions (~1 h)
python scripts/make_figures.py                  # all figures of the paper               -> results/paper/figures/
python scripts/make_data.py                     # recompute the reference solutions from scratch and compare (~20 s)
```

`check_paper.py` does two things. It evaluates the shipped trained models with this code and compares the errors with
the logged ones. It then retrains the logged runs for 100 iterations and compares the training loss and validation
residual with the logs. Use `--optimizers all --seeds 0,1,2,3,4` to cover all 300 runs, `--iters 4000` for complete
runs, and `--workers N` to run in parallel.

In the paper's floating-point environment (see *Exact reproduction*) the reruns are bit-identical to the logs. Elsewhere
early iterations agree to several digits and complete runs agree statistically.

## Training one model

```bash
python scripts/train.py --problem eheat --arch pinn --optimizer strider_o --lr 3e-3 --seed 0
```

This writes the result (errors, quantities of interest, training history) to `results/single/*.json` and the weights to
`*.npz`. The learning rates selected in the paper are listed in `results/paper/selected_lr.json` and in the table
produced by `analyze.py`.

## Reproducing the whole study

```bash
bash reproduce.sh full                       # everything below, in order (resumable)
# or step by step:
python scripts/tune_lr.py   --workers 8      # lr protocol + seed-0 runs, all optimizers and configurations
python scripts/run_seeds.py --workers 8      # seeds 1-4 at the selected learning rates
python scripts/run_curves.py --workers 8     # convergence curves of G-PULSE
python scripts/time_benchmark.py             # controlled timing (run alone on an idle machine)
python scripts/analyze.py --results results
python scripts/make_figures.py --results results
```

All runs are resumable: finished runs are read back from `results/`. One run takes 5–20 minutes on one core. The study
needs about 70 CPU-hours: about 25 for the tuning (including the seed-0 runs) and 40 for seeds 1–4. To skip the tuning and use the paper's
learning rates, run `python scripts/run_seeds.py --lr-from results/paper/selected_lr.json --seeds 0,1,2,3,4`.

## Method in brief

* **Trial functions** satisfy the initial and boundary conditions exactly:
  * integral equations: u = N;
  * bar: u = f₀ + (t/T) x(1−x) N;
  * cable: u = f₀ + t g₁ + (t/T)² x(1−x) N;
  * plate: u = 16 x(1−x) y(1−y) N.
* **Least-squares head (variable projection).** The residual is affine in the coefficients c of the last linear layer.
  At every step c is computed exactly by ridge-regularised least squares on the current collocation batch. The optimizer
  updates only the hidden layers, using the gradient of the reduced loss (`gpulse/varpro.py`). On the plate (both
  networks) and on the cable with the PIKAN the head is solved on one batch and the loss is taken on a fresh batch
  (split head).
* **Networks.**
  * PINN: Fourier features followed by 4 × 128 tanh layers.
  * PIKAN: three Chebyshev-KAN layers (width 32, degree 5); the last layer's basis T_k(tanh h) is the least-squares head.
* **Optimizers** (`gpulse/optimizers.py`).
  * G-PULSE-O replaces G-PULSE's preconditioned direction r_t, for every weight matrix and every matricised KAN
    coefficient tensor, by five Newton–Schulz iterations of it. G-PULSE's step normalisation and update clipping are
    kept.
  * Muon matricises the KAN coefficient tensors [out, in, K+1] into [out, in(K+1)].
* **Learning-rate protocol**, the same for every optimizer (`gpulse/experiments.py`):
  * four rates at a quarter of the budget, selected by the validation residual of the final model;
  * if the best rate lies on the edge of the grid, extension in that direction (×3; ×10 for Yogi), up to six points;
  * a full-budget run of the selected rate; if its residual is more than 3× worse than in tuning, the second-best rate
    is also run and the lower residual is kept.
* **Statistics** (`gpulse/stats.py`), on log₁₀ of the relative L2 error over five seeds:
  * geometric means with 95% t-intervals;
  * seed-paired t-tests with the Holm correction;
  * the Friedman test with the Nemenyi critical difference.

## Repository layout

```
gpulse/          the package
  autodiff.py           reverse-mode automatic differentiation on NumPy arrays
  networks.py           PINN and Chebyshev-KAN features with Taylor-mode input derivatives
  problems.py           the five problems: inputs, hard constraints, physics-informed residuals
  varpro.py             training with the least-squares head; error evaluation; saving/loading models
  optimizers.py         G-PULSE, G-PULSE-O, AdamW, Yogi, AdaBelief, Muon
  config.py             experimental settings and the tuning protocol constants
  experiments.py        resumable runs, the tuning protocol, parallel execution
  reference.py          spectral reference solvers (used only for the errors) and the data files
  metrics.py            MSE, relative L2/H1 errors, quantities of interest
  stats.py              statistical tests
  environment.py        hardware/software description stored in every run log
repro_env.py            floating-point environment for exact reproduction (imported before NumPy)
EXPERIMENTS.md          full experimental conditions and optimizer hyperparameters
scripts/                command-line entry points (see above)
data/                   input coefficients and reference solutions of the test input, one file per problem
results/paper/           logs of the paper: 300 runs (5 seeds x 6 optimizers x 10 configurations), weights of the
                        G-PULSE and G-PULSE-O runs, convergence curves, selected learning rates, timing.json
                        (time_benchmark.py); tables/ (analyze.py, latex_tables.py) and figures/ (make_figures.py)
tests/                  unit and reproduction tests (standard library unittest)
```

Every run in `results/paper/runs/<problem>_<arch>_<optimizer>_s<seed>.json` contains the settings, the selected
checkpoint, the errors (relative L2, RMSE, quantities of interest, space–time error for the bar and the cable) and the
training history.

## Citation

If you use this code, please cite the G-PULSE paper (reference to be added on publication).

## License

MIT, see `LICENSE`.
