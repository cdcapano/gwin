# Copyright (C) 2018  Collin Capano
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
This modules provides models that have analytic solutions for the
log likelihood.
"""

import numpy
from scipy import stats

from .base import BaseModel


class TestNormal(BaseModel):
    r"""The test distribution is an multi-variate normal distribution.

    The number of dimensions is set by the number of ``model_params`` that are
    passed. For details on the distribution used, see
    ``scipy.stats.multivariate_normal``.

    Parameters
    ----------
    model_params : (tuple of) string(s)
        A tuple of parameter names that will be varied.
    mean : array-like, optional
        The mean values of the parameters. If None provide, will use 0 for all
        parameters.
    cov : array-like, optional
        The covariance matrix of the parameters. If None provided, will use
        unit variance for all parameters, with cross-terms set to 0.
    **kwargs :
        All other keyword arguments are passed to ``BaseModel``.

    """
    name = "test_normal"

    def __init__(self, model_params, mean=None, cov=None, **kwargs):
        # set up base likelihood parameters
        super(TestNormal, self).__init__(model_params, **kwargs)
        # set the lognl to 0 since there is no data
        self.set_lognl(0.)
        # store the pdf
        if mean is None:
            mean = [0.]*len(model_params)
        if cov is None:
            cov = [1.]*len(model_params)
        self._dist = stats.multivariate_normal(mean=mean, cov=cov)
        # check that the dimension is correct
        if self._dist.dim != len(model_params):
            raise ValueError("dimension mis-match between model_params and "
                             "mean and/or cov")

    def loglikelihood(self, **params):
        """Returns the log pdf of the multivariate normal.
        """
        return self._dist.logpdf([params[p] for p in self.model_params])


class TestEggbox(BaseModel):
    r"""The test distribution is an 'eggbox' function:

    .. math::

        \log \mathcal{L}(\Theta) = \left[
            2+\prod_{i=1}^{n}\cos\left(\frac{\theta_{i}}{2}\right)\right]^{5}

    The number of dimensions is set by the number of ``model_params`` that are
    passed.

    Parameters
    ----------
    model_params : (tuple of) string(s)
        A tuple of parameter names that will be varied.
    **kwargs :
        All other keyword arguments are passed to ``BaseModel``.

    """
    name = "test_eggbox"

    def __init__(self, model_params, **kwargs):
        # set up base likelihood parameters
        super(TestEggbox, self).__init__(model_params, **kwargs)

        # set the lognl to 0 since there is no data
        self.set_lognl(0.)

    def loglikelihood(self, **params):
        """Returns the log pdf of the eggbox function.
        """
        return (2 + numpy.prod(numpy.cos([
            params[p]/2. for p in self.model_params]))) ** 5


class TestRosenbrock(BaseModel):
    r"""The test distribution is the Rosenbrock function:

    .. math::

        \log \mathcal{L}(\Theta) = -\sum_{i=1}^{n-1}[
            (1-\theta_{i})^{2}+100(\theta_{i+1} - \theta_{i}^{2})^{2}]

    The number of dimensions is set by the number of ``model_params`` that are
    passed.

    Parameters
    ----------
    model_params : (tuple of) string(s)
        A tuple of parameter names that will be varied.
    **kwargs :
        All other keyword arguments are passed to ``BaseModel``.

    """
    name = "test_rosenbrock"

    def __init__(self, model_params, **kwargs):
        # set up base likelihood parameters
        super(TestRosenbrock, self).__init__(model_params, **kwargs)

        # set the lognl to 0 since there is no data
        self.set_lognl(0.)

    def loglikelihood(self, **params):
        """Returns the log pdf of the Rosenbrock function.
        """
        logl = 0
        p = [params[p] for p in self.model_params]
        for i in range(len(p) - 1):
            logl -= ((1 - p[i])**2 + 100 * (p[i+1] - p[i]**2)**2)
        return logl


class TestVolcano(BaseModel):
    r"""The test distribution is a two-dimensional 'volcano' function:

    .. math::
        \Theta =
            \sqrt{\theta_{1}^{2} + \theta_{2}^{2}} \log \mathcal{L}(\Theta) =
            25\left(e^{\frac{-\Theta}{35}} +
                    \frac{1}{2\sqrt{2\pi}} e^{-\frac{(\Theta-5)^{2}}{8}}\right)

    Parameters
    ----------
    model_params : (tuple of) string(s)
        A tuple of parameter names that will be varied. Must have length 2.
    **kwargs :
        All other keyword arguments are passed to ``BaseModel``.

    """
    name = "test_volcano"

    def __init__(self, model_params, **kwargs):
        # set up base likelihood parameters
        super(TestVolcano, self).__init__(model_params, **kwargs)

        # make sure there are exactly two variable args
        if len(self.model_params) != 2:
            raise ValueError("TestVolcano distribution requires exactly "
                             "two variable args")

        # set the lognl to 0 since there is no data
        self.set_lognl(0.)

    def loglikelihood(self, **params):
        """Returns the log pdf of the 2D volcano function.
        """
        p = [params[p] for p in self.model_params]
        r = numpy.sqrt(p[0]**2 + p[1]**2)
        mu, sigma = 5.0, 2.0
        return 25 * (
            numpy.exp(-r/35) + 1 / (sigma * numpy.sqrt(2 * numpy.pi)) *
            numpy.exp(-0.5 * ((r - mu) / sigma) ** 2))
