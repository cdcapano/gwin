# Copyright (C) 2016  Collin Capano
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

"""
Base class structures.
"""

import numpy
import logging

from abc import ABCMeta, abstractmethod, abstractproperty

from pycbc import (conversions, transforms, distributions)
from pycbc.waveform import generator
from pycbc.io import FieldArray
from pycbc.workflow import ConfigParser


class _NoPrior(object):
    """Dummy class to just return 0 if no prior is given to a model.
    """
    @staticmethod
    def apply_boundary_conditions(**params):
        return params

    def __call__(self, **params):
        return 0.


class ModelStats(object):
    """Class to hold model's current stat values."""

    @property
    def statnames(self):
        """Returns the names of the stats that have been stored."""
        return self.__dict__.keys()

    def getstats(self, names, default=numpy.nan):
        """Get the requested stats.
        
        If a requested stat is not an attribute (implying it hasn't been
        stored), then the default value is returned for that stat.

        Parameters
        ----------
        names : list of str
            The names of the stats to get.
        default : float, optional
            What to return if a requested stat is not an attribute of self.
            Default is ``numpy.nan``.

        Returns
        -------
        tuple
            A tuple of the requested stats.
        """
        return tuple(getattr(self, n, default) for n in names)


def modelstats_to_arrays(model_stats):
    """Given a list of model stats, converts to a dictionary of numpy arrays.

    Parameters
    ----------
    model_stats : list of ModelStats instances
        The list to convert.

    Returns
    -------
    dict :
        Dictionary mapping stat names -> numpy arrays.
    """
    # use the first one to get the names of the stats
    statnames = model_stats[0].statnames
    # check that all of the model stats have the same stats in the same order
    assert(all(x.statnames == statnames for x in model_stats),
           "all model stats instances must have the same stats stored")
    # store in memory as a 2D array
    arr = numpy.array([x.stats for x in model_stats])
    # return as a dict
    return {p: arr[:, jj] for jj,p in enumerate(statnames)}


class BaseModel(object):
    r"""Base class for all models.

    The nomenclature used by this class and those that inherit from it is as
    follows: Given some model parameters :math:`\Theta` and some data
    :math:`d` with noise model :math:`n`, we define:

     * the **likelihood function**: :math:`p(d|\Theta)`

     * the **noise likelihood**: :math:`p(d|n)`

     * the **likelihood ratio**:
       :math:`\mathcal{L}(\Theta) = \frac{p(d|\Theta)}{p(d|n)}`

     * the **prior**: :math:`p(\Theta)`

     * the **posterior**: :math:`p(\Theta|d) \propto p(d|\Theta)p(\Theta)`

     * the **prior-weighted likelihood ratio**:
       :math:`\hat{\mathcal{L}}(\Theta) = \frac{p(d|\Theta)p(\Theta)}{p(d|n)}`

     * the **SNR**: :math:`\rho(\Theta) = \sqrt{2\log\mathcal{L}(\Theta)}`;
       for two detectors, this is approximately the same quantity as the
       coincident SNR used in the CBC search.

    .. note::

        Although the posterior probability is only proportional to
        :math:`p(d|\Theta)p(\Theta)`, here we refer to this quantity as the
        posterior. Also note that for a given noise model, the prior-weighted
        likelihood ratio is proportional to the posterior, and so the two can
        usually be swapped for each other.

    When performing parameter estimation we work with the log of these values
    since we are mostly concerned with their values around the maxima. If
    we have multiple detectors, each with data :math:`d_i`, then these values
    simply sum over the detectors. For example, the log likelihood ratio is:

    .. math::

        \log \mathcal{L}(\Theta) =
            \sum_i \left[\log p(\Theta|d_i) - \log p(n|d_i)\right]

    This class provides boiler-plate methods and attributes for evaluating the
    log likelihood ratio, log prior, and log likelihood. This class makes no
    assumption about the detectors' noise model :math:`n`. As such, the methods
    for computing these values raise ``NotImplementedErrors``. These functions
    need to be monkey patched, or other classes that inherit from this class
    need to define their own functions.

    Instances of this class can be called like a function. The default is for
    this class to call its ``logposterior`` function, but this can be changed
    with the ``set_callfunc`` method.

    Parameters
    ----------
    variable_params : (tuple of) string(s)
        A tuple of parameter names that will be varied.
    static_params : dict, optional
        A dictionary of parameter names -> values to keep fixed.
    prior : callable, optional
        A callable class or function that computes the log of the prior. If
        None provided, will use ``_noprior``, which returns 0 for all parameter
        values.
    sampling_params : list, optional
        Replace one or more of the ``variable_params`` with the given
        parameters for sampling.
    replace_parameters : list, optional
        The ``variable_params`` to replace with sampling parameters. Must be
        the same length as ``sampling_params``.
    sampling_transforms : list, optional
        List of transforms to use to go between the ``variable_params`` and the
        sampling parameters. Required if ``sampling_params`` is not None.

    Attributes
    ----------
    lognl : {None, float}
        The log of the noise likelihood summed over the number of detectors.
    return_meta : {True, bool}
        If True, ``prior``, ``logposterior``, and ``logplr`` will return the
        value of the prior, the loglikelihood ratio, and the log jacobian,
        along with the posterior/plr.

    Methods
    -------
    logjacobian :
        Returns the log of the jacobian needed to go from the parameter space
        of the ``variable_params`` to the sampling args.
    prior :
        A function that returns the log of the prior.
    loglikelihood :
        A function that returns the log of the likelihood function.
    logposterior :
        A function that returns the log of the posterior.
    loglr :
        A function that returns the log of the likelihood ratio.
    logplr :
        A function that returns the log of the prior-weighted likelihood ratio.
    snr :
        A function that returns the square root of twice the log likelihood
        ratio. If the log likelihood ratio is < 0, will return 0.
    evaluate :
        Maps a list of values to their parameter names and calls whatever the
        call function is set to.
    set_callfunc :
        Set the function to use when the class is called as a function.
    """
    __metaclass__ = ABCMeta
    name = None

    def __init__(self, variable_params, static_params=None, prior=None,
                 sampling_transforms=None):
        # store variable and static args
        if isinstance(variable_params, basestring):
            variable_params = (variable_params,)
        if not isinstance(variable_params, tuple):
            variable_params = tuple(variable_params)
        self._variable_params = variable_params
        if static_params is None:
            static_params = {}
        self._static_params = static_params
        # store prior
        if prior is None:
            self.prior_distribution = _NoPrior()
        else:
            # check that the variable args of the prior evaluator is the same
            # as the waveform generator
            if prior.variable_args != variable_params:
                raise ValueError("variable args of prior and waveform "
                                 "generator do not match")
            self.prior_distribution = prior
        # store sampling transforms
        self.sampling_transforms = sampling_transforms
        # initialize current params to None
        self._current_params = None
        # initialize a model stats
        self._current_stats = ModelStats()

    @property
    def variable_params(self):
        """Returns the model parameters."""
        return self._variable_params

    @property
    def static_params(self):
        """Returns the model's static arguments."""
        return self._static_params

    @property
    def sampling_params(self):
        """Returns the sampling parameters.

        If ``sampling_transforms`` is None, this is the same as the
        ``variable_params``.
        """
        if self.sampling_transforms is None:
            sampling_params = self.variable_params
        else:
            sampling_params = self.sampling_transforms.sampling_params
        return sampling_params

    def update(self, **params):
        """Updates the current parameter positions and resets stats.
        
        If any sampling transforms are specified, they are applied to the
        params before being stored.
        """
        self._current_params = self._transform_params(**params)
        self._current_stats = ModelStats()

    @property
    def current_params(self):
        assert(self._current_params is not None,
               "no parameters values currently stored; run update to add some")
        return self._current_params

    def current_stats(self, names=None):
        """Return one or more of the current stats as a tuple.

        This function does no computation. It only returns what has already
        been calculated. If a stat hasn't been calculated, it will be returned
        as ``numpy.nan``.

        Parameters
        ----------
        names : list of str, optional
            Specify the names of the stats to retrieve. If ``None`` (the
            default), will return ``default_stats``.

        Returns
        -------
        tuple :
            The current values of the requested stats, as a tuple. The order
            of the stats is the same as the names.
        """
        if names is None:
            names = self.default_stats
        return self._current_stats.getstats(names)

    @property
    def default_stats(self):
        """The stats that ``get_current_stats`` returns by default."""
        return ['logjacobian', 'logprior', 'loglikelihood']

    @abstractproperty
    def loglikelihood(self):
        """The log likelihood at the current parameters.

        This will initially try to return the ``current_stats.loglikelihood``.
        If that raises an ``AttributeError``, will call `_loglikelihood`` to
        calculate it and store it to ``current_stats``.
        """
        try:
            return self._current_stats.loglikelihood
        except AttributeError:
            logl = self._loglikelihood()
            self._current_stats.loglikelihood = logl
            return logl

    @abstractmethod
    def _loglikelihood(self):
        """Low-level function that calculates the log likelihood of the current
        params."""
        pass

    @property
    def logjacobian(self):
        """The log jacobian of the sampling transforms at the current postion.

        If no sampling transforms were provided, will just return 0.

        Parameters
        ----------
        \**params :
            The keyword arguments should specify values for all of the variable
            args and all of the sampling args.

        Returns
        -------
        float :
            The value of the jacobian.
        """
        try:
            logj = self._current_stats.logjacobian
        except AttributeError:
            # hasn't been calculated on these params yet
            if self.sampling_transforms is None:
                logj = 0.
            else:
                logj = self.sampling_transforms.logjacobian(
                    **self.current_params)
            # add to current stats
            self._current_stats.logjacobian = logj
        return logj

    @property
    def logprior(self):
        """Returns the prior at the current parameter points.
        """
        try:
            logp = self._current_stats.logprior
        except AttributeError:
            # hasn't been calculated on these params yet
            logj = self.logjacobian
            logp = self.prior_distribution(**self.current_params) + logj
            if numpy.isnan(logp):
                logp = -numpy.inf
            # add to current stats
            self._current_stats.logprior = logp
        return logp 

    @property
    def logposterior(self):
        """Returns the log of the posterior of the current parameter values.

        The logprior is calculated first. If the logprior returns ``-inf``
        (possibly indicating a non-physical point), then the ``loglikelihood``
        is not called.
        """
        logp = self.logprior
        if logp == -numpy.inf:
            return logp
        else:
            return logp + self.loglikelihood

    def prior_rvs(self, size=1, prior=None):
        """Returns random variates drawn from the prior.

        If the ``sampling_params`` are different from the ``variable_params``,
        the variates are transformed to the `sampling_params` parameter space
        before being returned.

        Parameters
        ----------
        size : int, optional
            Number of random values to return for each parameter. Default is 1.
        prior : JointDistribution, optional
            Use the given prior to draw values rather than the saved prior.

        Returns
        -------
        FieldArray
            A field array of the random values.
        """
        # draw values from the prior
        if prior is None:
            prior = self.prior_distribution
        p0 = prior.rvs(size=size)
        # transform if necessary
        if self.sampling_transforms is not None:
            ptrans = self.sampling_transforms.apply(p0)
            # pull out the sampling args
            p0 = FieldArray.from_arrays([ptrans[arg]
                                         for arg in self.sampling_params],
                                        names=self.sampling_params)
        return p0

    def _transform_params(self, **params):
        """Applies all transforms to the given params.

        Parameters
        ----------
        \**params :
            Key, value pairs of parameters to apply the transforms to.

        Returns
        -------
        dict
            A dictionary of the transformed parameters.
        """
        # apply inverse transforms to go from sampling parameters to
        # variable args
        if self.sampling_transforms is not None:
            params = self.sampling_transforms.apply(params, inverse=True)
        # apply boundary conditions
        params = self.prior_distribution.apply_boundary_conditions(**params)
        return params

    #
    # Methods for initiating from a config file.
    #
    @staticmethod
    def extra_args_from_config(cp, section, skip_args=None, dtypes=None):
        """Gets any additional keyword in the given config file.

        Parameters
        ----------
        cp : WorkflowConfigParser
            Config file parser to read.
        section : str
            The name of the section to read.
        skip_args : list of str, optional
            Names of arguments to skip.
        dtypes : dict, optional
            A dictionary of arguments -> data types. If an argument is found
            in the dict, it will be cast to the given datatype. Otherwise, the
            argument's value will just be read from the config file (and thus
            be a string).

        Returns
        -------
        dict
            Dictionary of keyword arguments read from the config file.
        """
        kwargs = {}
        if dtypes is None:
            dtypes = {}
        if skip_args is None:
            skip_args = []
        read_args = [opt for opt in cp.options(section)
                     if opt not in skip_args]
        for opt in read_args:
            val = cp.get(section, opt)
            # try to cast the value if a datatype was specified for this opt
            try:
                val = dtypes[opt](val)
            except KeyError:
                pass
            kwargs[opt] = val
        return kwargs

    @staticmethod
    def prior_from_config(cp, variable_params, prior_section,
                          constraint_section):
        """Gets arguments and keyword arguments from a config file.

        Parameters
        ----------
        cp : WorkflowConfigParser
            Config file parser to read.
        variable_params : list
            List of of model parameter names.
        prior_section : str
            Section to read prior(s) from.
        constraint_section : str
            Section to read constraint(s) from.

        Returns
        -------
        pycbc.distributions.JointDistribution
            The prior.
        """
        # get prior distribution for each variable parameter
        logging.info("Setting up priors for each parameter")
        dists = distributions.read_distributions_from_config(cp, prior_section)
        constraints = distributions.read_constraints_from_config(
            cp, constraint_section)
        return distributions.JointDistribution(variable_params, *dists,
                                               constraints=constraints)

    @classmethod
    def _init_args_from_config(cls, cp):
        """Helper function for loading parameters."""
        section = "model"
        prior_section = "prior",
        vparams_section = 'variable_params'
        sparams_section = 'static_params'
        constraint_section = 'constraint'
        # check that the name exists and matches
        name = cp.get(section, 'name')
        if name != cls.name:
            raise ValueError("section's {} name does not match mine {}".format(
                             name, cls.name))
        # get model parameters
        # Requires PyCBC 1.11.2
        variable_params, static_params = distributions.read_params_from_config(
            cp, prior_section=prior_section, vargs_section=vparams_section,
            sparams_section=sparams_section)
        # get prior
        prior = cls.prior_from_config(cp, variable_params, prior_section,
                                      constraint_section)
        args = {'variable_params': variable_params,
                'static_params': static_params,
                'prior': prior}
        # get any other keyword arguments provided
        args.update(cls.extra_args_from_config(cp, section,
                                               skip_args=['name']))
        return args

    def from_config(cls, cp, **kwargs):
        """Initializes an instance of this class from the given config file.

        Parameters
        ----------
        cp : WorkflowConfigParser
            Config file parser to read.
        \**kwargs :
            All additional keyword arguments are passed to the class. Any
            provided keyword will over ride what is in the config file.
        """
        args = cls._init_args_from_config(cp)
        # try to load sampling transforms
        try:
            sampling_transforms = SamplingTransforms.from_config(
                cp, args['variable_params'])
        except AssertionError:
            sampling_transforms = None
        args['sampling_transforms'] = sampling_transforms
        args.update(kwargs)
        return cls(**args)



class BaseDataModel(BaseModel):
    r"""A model that requires data and a waveform generator.

    Like ``BaseModel``, this class only provides boiler-plate
    attributes and methods for evaluating models. Classes that make use
    of data and a waveform generator should inherit from this.

    Parameters
    ----------
    variable_params : (tuple of) string(s)
        A tuple of parameter names that will be varied.
    data : dict
        A dictionary of data, in which the keys are the detector names and the
        values are the data.
    waveform_generator : generator class
        A generator class that creates waveforms.
    waveform_transforms : list, optional
        List of transforms to use to go from the variable args to parameters
        understood by the waveform generator.

    \**kwargs :
        All other keyword arguments are passed to ``BaseModel``.

    Attributes
    ----------
    waveform_generator : dict
        The waveform generator that the class was initialized with.
    data : dict
        The data that the class was initialized with.

    For additional attributes and methods, see ``BaseModel``.
    """
    __metaclass__ = ABCMeta

    def __init__(self, variable_params, data, waveform_generator,
                 waveform_transforms=None, **kwargs):
        # we'll store a copy of the data
        self._data = {ifo: d.copy() for (ifo, d) in data.items()}
        self._waveform_generator = waveform_generator
        self._waveform_transforms = waveform_transforms
        super(BaseDataModel, self).__init__(
            variable_params, **kwargs)

    @property
    def default_stats(self):
        """The stats that ``get_current_stats`` returns by default."""
        return ['logjacobian', 'logprior', 'loglr', 'lognl']

    @property
    def lognl(self):
        """The log likelihood of the model assuming the data is noise.

        This will initially try to return the ``current_stats.lognl``.
        If that raises an ``AttributeError``, will call `_lognl`` to
        calculate it and store it to ``current_stats``.
        """
        try:
            return self._current_stats.lognl
        except AttributeError:
            lognl = self._lognl()
            self._current_stats.lognl = lognl
            return lognl

    @abstractmethod
    def _lognl(self):
        """Low-level function that calculates the lognl."""
        pass

    @property
    def loglr(self):
        """The log likelihood ratio at the current parameters.

        This will initially try to return the ``current_stats.loglr``.
        If that raises an ``AttributeError``, will call `_loglr`` to
        calculate it and store it to ``current_stats``.
        """
        pass

    @abstractmethod
    def _loglr(self):
        """Low-level function that calculates the loglr."""
        pass

    @property
    def logplr(self):
        """Returns the log of the prior-weighted likelihood ratio at the
        current parameter values.

        The logprior is calculated first. If the logprior returns ``-inf``
        (possibly indicating a non-physical point), then ``loglr`` is not
        called.
        """
        logp = self.logprior
        if logp == -numpy.inf:
            return logp
        else:
            return logp + self.loglr

    @property
    def waveform_generator(self):
        """Returns the waveform generator that was set."""
        return self._waveform_generator

    @property
    def data(self):
        """Returns the data that was set."""
        return self._data

    def _transform_params(self, params):
        """Adds waveform transforms to parent's ``_transform_params``."""
        params = super(BaseDataModel, self)._transform_params(
            params)
        # apply waveform transforms
        if self._waveform_transforms is not None:
            params = transforms.apply_transforms(params,
                                                 self._waveform_transforms,
                                                 inverse=False)
        return params

    @classmethod
    def _init_args_from_config(cls, cp):
        """Adds loading waveform_transforms to parent function.

        For details on parameters, see ``from_config``.
        """
        args = super(DataModel, cls)._init_args_from_config(cp)
        # add waveform transforms to the arguments
        if any(cp.get_subsections('waveform_transforms')):
            logging.info("Loading waveform transforms")
            args['waveform_transforms'] = \
                transforms.read_transforms_from_config(
                    cp, 'waveform_transforms')
        return args

    @classmethod
    def from_config(cls, cp, data, delta_f=None, delta_t=None,
                    gates=None, recalibration=None,
                    **kwargs):
        """Initializes an instance of this class from the given config file.

        Parameters
        ----------
        cp : WorkflowConfigParser
            Config file parser to read.
        data : dict
            A dictionary of data, in which the keys are the detector names and
            the values are the data. This is not retrieved from the config
            file, and so must be provided.
        delta_f : float
            The frequency spacing of the data; needed for waveform generation.
        delta_t : float
            The time spacing of the data; needed for time-domain waveform
            generators.
        recalibration : dict of pycbc.calibration.Recalibrate, optional
            Dictionary of detectors -> recalibration class instances for
            recalibrating data.
        gates : dict of tuples, optional
            Dictionary of detectors -> tuples of specifying gate times. The
            sort of thing returned by `pycbc.gate.gates_from_cli`.
        \**kwargs :
            All additional keyword arguments are passed to the class. Any
            provided keyword will over ride what is in the config file.
        """
        if data is None:
            raise ValueError("must provide data")
        args = cls._init_args_from_config(cp)
        args['data'] = data
        args.update(kwargs)

        variable_params = args['variable_params']
        try:
            static_params = args['static_params']
        except KeyError:
            static_params = {}

        # set up waveform generator
        try:
            approximant = static_params['approximant']
        except KeyError:
            raise ValueError("no approximant provided in the static args")
        generator_function = generator.select_waveform_generator(approximant)
        waveform_generator = generator.FDomainDetFrameGenerator(
            generator_function, epoch=data.values()[0].start_time,
            variable_args=variable_params, detectors=data.keys(),
            delta_f=delta_f, delta_t=delta_t,
            recalib=recalibration, gates=gates,
            **static_params)
        args['waveform_generator'] = waveform_generator

        return cls(**args)


class SamplingTransforms(object):
    """Provides methods for transforming between sampling parameter space and
    model parameter space.
    """

    def __init__(self, variable_params, sampling_params,
                 replace_parameters, sampling_transforms):
        assert(len(replace_parameters) == len(sampling_params),
               "number of sampling parameters must be the "
               "same as the number of replace parameters")
        # pull out the replaced parameters
        self.sampling_params = [arg for arg in variable_params
                                if arg not in replace_parameters]
        # add the sampling parameters
        self.sampling_params += sampling_params
        self.sampling_transforms = sampling_transforms

    def logjacobian(self, **params):
        r"""Returns the log of the jacobian needed to transform pdfs in the
        ``variable_params`` parameter space to the ``sampling_params``
        parameter space.

        Let :math:`\mathbf{x}` be the set of variable parameters,
        :math:`\mathbf{y} = f(\mathbf{x})` the set of sampling parameters, and
        :math:`p_x(\mathbf{x})` a probability density function defined over
        :math:`\mathbf{x}`.
        The corresponding pdf in :math:`\mathbf{y}` is then:

        .. math::

            p_y(\mathbf{y}) =
                p_x(\mathbf{x})\left|\mathrm{det}\,\mathbf{J}_{ij}\right|,

        where :math:`\mathbf{J}_{ij}` is the Jacobian of the inverse transform
        :math:`\mathbf{x} = g(\mathbf{y})`. This has elements:

        .. math::

            \mathbf{J}_{ij} = \frac{\partial g_i}{\partial{y_j}}

        This function returns
        :math:`\log \left|\mathrm{det}\,\mathbf{J}_{ij}\right|`.


        Parameters
        ----------
        \**params :
            The keyword arguments should specify values for all of the variable
            args and all of the sampling args.

        Returns
        -------
        float :
            The value of the jacobian.
        """
        return numpy.log(abs(transforms.compute_jacobian(
            params, self.sampling_transforms, inverse=True)))

    def apply(self, samples, inverse=False):
        """Applies the sampling transforms to the given samples.

        Parameters
        ----------
        samples : dict or FieldArray
            The samples to apply the transforms to.
        inverse : bool, optional
            Whether to apply the inverse transforms (i.e., go from the sampling
            args to the ``variable_params``). Default is False.

        Returns
        -------
        dict or FieldArray
            The transformed samples, along with the original samples.
        """
        return transforms.apply_transforms(samples, self.sampling_transforms,
                                           inverse=inverse)

    @classmethod
    def from_config(cls, cp, variable_params):
        """Gets sampling transforms specified in a config file.

        Sampling parameters and the parameters they replace are read from the
        ``sampling_params`` section, if it exists. Sampling transforms are
        read from the ``sampling_transforms`` section(s), using
        ``transforms.read_transforms_from_config``.

        An ``AssertionError`` is raised if no ``sampling_params`` section
        exists in the config file.

        Parameters
        ----------
        cp : WorkflowConfigParser
            Config file parser to read.
        variable_params : list
            List of parameter names of the original variable params.

        Returns
        -------
        SamplingTransforms
            A sampling transforms class.
        """
        assert(cp.has_section('sampling_params'),
               "no sampling_params section found in config file")
        # get sampling transformations
        sampling_params, replace_parameters = \
            read_sampling_params_from_config(cp)
        sampling_transforms = transforms.read_transforms_from_config(
            cp, 'sampling_transforms')
        logging.info("Sampling in {} in place of {}".format(
            ', '.join(sampling_params), ', '.join(replace_parameters)))
        return cls(variable_params, sampling_params,
                   replace_parameters, sampling_transforms)


def read_sampling_params_from_config(cp, section_group=None,
                                     section='sampling_params'):
    """Reads sampling parameters from the given config file.

    Parameters are read from the `[({section_group}_){section}]` section.
    The options should list the variable args to transform; the parameters they
    point to should list the parameters they are to be transformed to for
    sampling. If a multiple parameters are transformed together, they should
    be comma separated. Example:

    .. code-block:: ini

        [sampling_params]
        mass1, mass2 = mchirp, logitq
        spin1_a = logitspin1_a

    Note that only the final sampling parameters should be listed, even if
    multiple intermediate transforms are needed. (In the above example, a
    transform is needed to go from mass1, mass2 to mchirp, q, then another one
    needed to go from q to logitq.) These transforms should be specified
    in separate sections; see ``transforms.read_transforms_from_config`` for
    details.

    Parameters
    ----------
    cp : WorkflowConfigParser
        An open config parser to read from.
    section_group : str, optional
        Append `{section_group}_` to the section name. Default is None.
    section : str, optional
        The name of the section. Default is 'sampling_params'.

    Returns
    -------
    sampling_params : list
        The list of sampling parameters to use instead.
    replaced_params : list
        The list of variable args to replace in the sampler.
    """
    if section_group is not None:
        section_prefix = '{}_'.format(section_group)
    else:
        section_prefix = ''
    section = section_prefix + section
    replaced_params = set()
    sampling_params = set()
    for args in cp.options(section):
        map_args = cp.get(section, args)
        sampling_params.update(set(map(str.strip, map_args.split(','))))
        replaced_params.update(set(map(str.strip, args.split(','))))
    return list(sampling_params), list(replaced_params)
