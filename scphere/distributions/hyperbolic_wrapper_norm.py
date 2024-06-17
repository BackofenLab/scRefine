# Pytorch Distribution wrapper for the hyperbolic normal distribution
#

import torch
import torch.distributions as dist
import torch.nn.functional as F


class HyperbolicWrappedNorm(dist.Distribution):
    def __init__(self, loc, scale, validate_args=None):
        self.loc = loc
        self.scale = scale
        self.dim = loc.shape[1] - 1
        self.base_dist = dist.Normal(torch.zeros_like(scale), scale)
        super(HyperbolicWrappedNorm, self).__init__(validate_args=validate_args)

    @property
    def loc(self):
        return self.loc

    @property
    def scale(self):
        return self.scale

    def _sample_n(self, n, seed=0):
        shape = torch.concat([[n], torch.shape(self.scale)], 0)

        zn = torch.randn(shape)
        zn *= self.scale

        shape1 = [n, self.batch_shape_tensor()[0], 1]
        z0 = torch.concat([torch.zeros(shape1), zn], axis=-1)

        loc0 = self._lorentzian_orig(shape1, shape)
        tmp = torch.unsqueeze(self.loc, 0)

        shape2 = [n, 1, 1]
        zt = self._parallel_transport(z0, loc0, torch.tile(tmp, shape2))
        z = self._exp_map(zt, torch.tile(tmp, shape2))

    @staticmethod
    def _lorentzian_orig(s1, s0):
        x1 = torch.ones(s1)
        x0 = torch.zeros(s0)

        x_orig = torch.cat([x1, x0], dim=-1)
        return x_orig

    @staticmethod
    def _clip_min_value(x, eps=1e-6):
        return F.relu(x - eps) + eps

    def _exp_map(self, x, mu):
        res = self._lorentzian_product(x, x)
        res = torch.math.sqrt(self._clip_min_value(res))

        res = torch.clip_by_value(res, 0, 32)

        return torch.math.cosh(res) * mu + torch.math.sinh(res) * x / res

    @staticmethod
    def _lorentzian_product(x, y):
        y0, y1 = torch.split(y, [1, y.shape[-1] - 1], dim=-1)
        y_neg_first = torch.cat([-y0, y1], dim=-1)
        return torch.sum(x * y_neg_first, dim=-1, keepdim=True)

    def _parallel_transport(self, x, m1, m2):
        alpha = -self._lorentzian_product(m1, m2)
        coef = self._lorentzian_product(m2, x) / (alpha + 1.0)

        return x + coef * (m1 + m2)

    def _lorentz_distance(self, x, y):
        res = -self._lorentzian_product(x, y)
        res = self._clip_min_value(res, 1 + 1e-6)

        z = torch.sqrt(res + 1) * torch.sqrt(res - 1)

        return torch.math.log(res + z)

    def _inv_exp_map(self, x, mu):
        alpha = -self._lorentzian_product(x, mu)
        alpha = self._clip_min_value(alpha, 1 + 1e-6)

        tmp = (
            self._lorentz_distance(x, mu)
            / torch.sqrt(alpha + 1)
            / torch.sqrt(alpha - 1)
        )

        return tmp * (x - alpha * mu)

    def _log_prob(self, x):
        v = self._inv_exp_map(x, self._loc)
        tmp = self._lorentzian_product(v, v)
        x_norm = torch.math.sqrt(self._clip_min_value(tmp))

        x_norm = torch.clip_by_value(x_norm, 0, 32)
        res = (torch.tensor(self._dim).to(torch.float32) - 1.0) * torch.math.log(
            torch.math.sinh(x_norm) / x_norm
        )

        shape = torch.shape(self.scale)
        shape1 = [self.batch_shape_tensor()[0], 1]

        loc0 = self._lorentzian_orig(shape1, shape)
        v1 = self._parallel_transport(v, self._loc, loc0)
        xx = v1[..., 1:]
        log_base_prob = torch.sum(self._base_dist.log_prob(xx), -1)
        log_base_prob = torch.reshape(log_base_prob, shape1)

        return log_base_prob - res

    def _prob(self, x):
        return torch.exp(self._log_prob(x))
