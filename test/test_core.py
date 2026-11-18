"""Unit tests (standard library unittest, ~1 min):  python -m unittest discover -s tests -v
Automatic differentiation, Taylor-mode input derivatives, optimizers, physics residuals against the reference."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import repro_env  # noqa: E402,F401  (before numpy)
import numpy as np  # noqa: E402
from gpulse import autodiff as ad  # noqa: E402
from gpulse.networks import F, FeatureNet  # noqa: E402
from gpulse.optimizers import OPTIMIZERS, make_optimizer, newton_schulz  # noqa: E402
from gpulse.problems import Problem, residual_elliptic  # noqa: E402
from gpulse.reference import bary_matrix, bary_weights, cheb_lobatto  # noqa: E402


class _Reference:
    """A 'network' returning the reference solution of an integral equation (barycentric interpolation)."""

    def __init__(self, prob):
        self.xe, self.wb, self.Ye = prob.xe, bary_weights(prob.xe), prob.Ye

    def __call__(self, Z, d1, d2, scale):
        x = (Z[..., 0] + 1) / 2
        return F({"v": ad.T((bary_matrix(self.xe, self.wb, x.ravel()) @ self.Ye).reshape(x.shape))})


class TestAutodiff(unittest.TestCase):
    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        W = ad.param(rng.standard_normal((3, 4)))
        b = ad.param(rng.standard_normal(4))
        X = rng.standard_normal((5, 3))

        def loss():
            h = ad.tanh(ad.add(ad.einsum("nf,fo->no", X, W), b))
            s = ad.stack([h, ad.square(h)], -1)
            c = ad.concat([ad.reshape(s, (5, 8)), ad.getitem(h, (Ellipsis, slice(0, 2)))], -1)
            return ad.mean(ad.square(ad.div(c, 2.0 + ad.square(h).sum(-1, keepdims=True))))

        loss().backward()
        for p in (W, b):
            num = np.zeros_like(p.v)
            for i in np.ndindex(p.v.shape):
                old = p.v[i]
                p.v[i] = old + 1e-6
                lp = loss().v
                p.v[i] = old - 1e-6
                lm = loss().v
                p.v[i] = old
                num[i] = (lp - lm) / 2e-6
            np.testing.assert_allclose(p.g, num, rtol=1e-6, atol=1e-9)


class TestTaylorMode(unittest.TestCase):
    def test_input_derivatives_match_finite_differences(self):
        for arch, size in (("pinn", dict(width=8, depth=2, freqs=[1, 2], tfeat=[4.0])),
                           ("pikan", dict(kan_hidden=[4, 4, 3]))):
            net = FeatureNet(arch, 2, size, np.random.default_rng(1))
            scale = {"x": (0, 2.0), "t": (1, 2.0 / 0.3)}
            pts = np.random.default_rng(2).uniform(0.1, 0.9, (1, 6, 2))
            pts[..., 1] *= 0.3
            Z = lambda p: np.stack([2 * p[..., 0] - 1, 2 * p[..., 1] / 0.3 - 1], -1)      # noqa: E731
            with ad.no_grad():
                Fe = net.features(Z(pts), ("x", "t"), ("xx", "tt"), scale)
                for c, k in ((0, "x"), (1, "t")):
                    e = np.zeros(2)
                    e[c] = 1e-5 * (1.0 if k == "x" else 0.3)          # central differences, error O(h^2)
                    fp = net.features(Z(pts + e), (), (), scale)["v"].v
                    fm = net.features(Z(pts - e), (), (), scale)["v"].v
                    d1 = (fp - fm) / (2 * e[c])
                    self.assertLess(np.max(np.abs(Fe[k].v - d1)) / np.max(np.abs(d1)), 1e-6, arch + k)
                    e *= 30
                    fp = net.features(Z(pts + e), (), (), scale)["v"].v
                    fm = net.features(Z(pts - e), (), (), scale)["v"].v
                    d2 = (fp - 2 * Fe["v"].v + fm) / e[c] ** 2
                    self.assertLess(np.max(np.abs(Fe[k + k].v - d2)) / np.max(np.abs(d2)), 1e-3, arch + k + k)   # FD error ~1e-4


class TestOptimizers(unittest.TestCase):
    def test_newton_schulz_is_nearly_orthogonal(self):
        sv = np.linalg.svd(newton_schulz(np.random.default_rng(3).standard_normal((8, 20))), compute_uv=False)
        self.assertTrue(sv.min() > 0.6 and sv.max() < 1.25)    # Muon's quintic iteration: singular values ~ 1

    def test_all_optimizers_decrease_a_quadratic(self):
        for name in OPTIMIZERS:
            rng = np.random.default_rng(4)
            A = rng.standard_normal((6, 6))
            A = A @ A.T / 6 + np.eye(6)
            p, q = ad.param(rng.standard_normal((6, 5))), ad.param(rng.standard_normal(5))
            opt = make_optimizer(name, [p, q], 1e-2)
            f = lambda: float(np.sum(p.v * (A @ p.v)) + np.sum(q.v ** 2))      # noqa: E731
            f0 = f()
            for _ in range(200):
                p.g, q.g = 2 * A @ p.v, 2 * q.v
                opt.step()
            self.assertLess(f(), 0.5 * f0, name)

    def test_strider_o_equals_strider_on_vectors(self):
        rng = np.random.default_rng(5)
        v1 = ad.param(rng.standard_normal(7))
        v2 = ad.param(v1.v.copy())
        o1, o2 = make_optimizer("strider", [v1], 1e-3), make_optimizer("strider_o", [v2], 1e-3)
        for _ in range(10):
            g = rng.standard_normal(7)
            v1.g, v2.g = g, g.copy()
            o1.step()
            o2.step()
        self.assertTrue(np.array_equal(v1.v, v2.v))


class TestPhysics(unittest.TestCase):
    def test_reference_satisfies_integral_equations(self):
        """The training residual vanishes on the independent spectral reference solution."""
        for name in ("epanto", "efin"):
            prob = Problem(name)
            r = prob.residual(_Reference(prob), prob.sample(1, 200, np.random.default_rng(6))).v
            self.assertLess(np.max(np.abs(r)), 1e-10 * np.max(np.abs(prob.Ye)), name)

    def test_reference_satisfies_plate_equation(self):
        """Plate: residual of the reference (49 x 49 Chebyshev grid, spectral derivatives) at nodes with
        0.1 < x, y < 0.9 (the corners carry weak singularities that spoil spectral derivatives next to the boundary)."""
        prob = Problem("eplate")
        n = 49
        x, _, Dm = cheb_lobatto(n - 1)
        U = prob.Ye.reshape(n, n)                              # U[i, j] = u(x_i, y_j)
        X, Y = np.meshgrid(x, x, indexing="ij")
        ushift = bary_matrix(x, bary_weights(x), 0.5 * x) @ U  # u(x/2, y)
        r = residual_elliptic("eplate", X, Y, U, Dm @ U, Dm @ Dm @ U, U @ (Dm @ Dm).T, ushift, prob.inp)
        phi = prob.inp.phi(X.reshape(1, -1), Y.reshape(1, -1))
        inner = (X > 0.1) & (X < 0.9) & (Y > 0.1) & (Y < 0.9)
        self.assertLess(np.max(np.abs(r[inner])), 1e-8 * np.max(np.abs(phi)))


if __name__ == "__main__":
    unittest.main()
