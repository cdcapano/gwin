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
This package provides classes and functions for evaluating Bayesian statistics
assuming various noise models.
"""


from .analytic import (TestEggbox, TestNormal, TestRosenbrock, TestVolcano)
from .gaussian_noise import (GaussianNoise,
                             MarginalizedPhaseGaussianNoise)


# Used to manage a model instance across multiple cores or MPI
_global_instance = None


def _call_global_model(*args, **kwds):
    """Private function for global model (needed for parallelization)."""
    return _global_instance(*args, **kwds)  # pylint:disable=not-callable


class CallModel(object):
    """Wrapper class for calling models from a sampler.

    This class can be called like a function.

    This class must be initalized prior to the creation of a ``Pool`` object.

    Parameters
    ----------
    model : Model instance
        The model to call.
    callfunction : str
        The statistic to call.
    return_stats : bool, optional
        Whether or not to return all of the other statistics along with the
        ``callfunction`` value.
    """

    def __init__(self, model, callfunction, return_all_stats=True):
        self.model = model
        self.callfunction = callfunction
        self.return_all_stats = return_all_stats

    def __call__(self, param_values):
        """Updates the model with the given parameter values, then calls the
        call function.

        Parameters
        ----------
        param_values : list of float
            The parameter values to test. Assumed to be in the same order as
            ``model.sampling_params``.

        Returns
        -------
        stat : float
            The statistic returned by the ``callfunction``.
        all_stats : tuple, optional
            The values of all of the model's ``default_stats`` at the given
            param values. Any stat that has not be calculated is set to
            ``numpy.nan``. This is only returned if ``return_all_stats`` is
            set to ``True``.
        """
        params = dict(zip(self.model.sampling_params, param_values))
        self.model.update(**params)
        val = getattr(self, model, self.callfunction)
        if self.return_stats:
            return val, self.model.current_stats
        else:
            return val


def read_from_config(cp, **kwargs):
    """Initializes a model from the given config file.

    The section must have a ``name`` argument. The name argument corresponds to
    the name of the class to initialize.

    Parameters
    ----------
    cp : WorkflowConfigParser
        Config file parser to read.
    \**kwargs :
        All other keyword arguments are passed to the ``from_config`` method
        of the class specified by the name argument.

    Returns
    -------
    cls
        The initialized model.
    """
    # use the name to get the distribution
    name = cp.get(section, "name")
    return models[name].from_config(cp, **kwargs)


models = {_cls.name: _cls for _cls in (
    TestEggbox,
    TestNormal,
    TestRosenbrock,
    TestVolcano,
    GaussianNoise,
    MarginalizedPhaseGaussianNoise,
)}
