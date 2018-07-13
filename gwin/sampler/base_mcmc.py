# Copyright (C) 2016  Christopher M. Biwer, Collin Capano
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
"""Provides constructor classes for MCMC samplers."""

from abc import ABCMeta, abstractmethod, abstractproperty

class BaseMCMC(object):
    """This class provides methods common to MCMCs.

    It is not a sampler class itself. Sampler classes can inherit from this
    along with ``BaseSampler``.

    Attributes
    ----------
    p0 : dict
        A dictionary of the initial position of the walkers. Set by using
        ``set_p0``. If not set yet, a ``ValueError`` is raised when the
        attribute is accessed.
    pos : dict
        A dictionary of the current walker positions. If the sampler hasn't
        been run yet, returns p0.
    """
    __metaclass__ = ABCMeta

    lastclear = None
    _itercounter = None
    _pos = None
    _p0 = None
    _nwalkers = None

    @abstractproperty(self):
    def samples_shape(self):
        """Should define what shape to expect samples to be in."""
        pass

    @property
    def nwalkers(self):
        """Get the number of walkers."""
        if self._nwalkers is None:
            raise ValueError("number of walkers not set")
        return self._nwalkers

    @property
    def niterations(self):
        """Get the current number of iterations."""
        itercounter = self._itercounter
        if _itercounter is None:
            itercounter = 0
        lastclear = self.lastclear
        if lastclear is None:
            lastclear = 0
        return itercounter + lastclear

    @property
    def pos(self):
        pos = self._pos
        if pos is None:
            return self.p0
        # convert to dict
        pos = {param: self._pos[..., k]
               for (k, param) in enumerate(self.sampling_params)}
        return pos

    @property
    def p0(self):
        """The starting position of the walkers in the sampling param space.
        
        The returned object is a dict mapping the sampling parameters to the
        values.
        """
        if self._p0 is None:
            raise ValueError("initial positions not set; run set_p0")
        # convert to dict
        p0 = {param: self._p0[..., k]
              for (k, param) in enumerate(self.sampling_params)}
        return p0

    def set_p0(self, samples_file=None, prior=None):
        """Sets the initial position of the walkers.

        Parameters
        ----------
        samples_file : InferenceFile, optional
            If provided, use the last iteration in the given file for the
            starting positions.
        prior : JointDistribution, optional
            Use the given prior to set the initial positions rather than
            ``model``'s prior.

        Returns
        -------
        p0 : dict
            A dictionary maping sampling params to the starting positions.
        """
        # if samples are given then use those as initial positions
        if samples_file is not None:
            with self.io(samples_file, 'r') as fp:
                samples = fp.read_samples(self.variable_params,
                                          iteration=-1)
                # make sure we have the same shape
                assert(samples.shape == self.samples_shape,
                       "samples in file {} have shape {}, but I have shape {}".
                       format(samples_file, samples.shape, self.samples_shape))
            # transform to sampling parameter space
            samples = self.model.apply_sampling_transforms(samples)
        # draw random samples if samples are not provided
        else:
            nsamples = numpy.prod(self.samples_shape)
            samples = self.model.prior_rvs(size=nsamples, prior=prior).reshape(
                self.samples_shape)
        # store as ND array with shape [samples_shape] x nparams
        ndim = len(self.variable_params)
        p0 = numpy.ones(list(self.samples_shape)+[ndim])
        for i, param in enumerate(self.sampling_params):
            p0[..., i] = samples[param]
        self._p0 = p0
        return self.p0

    @classmethod
    def n_independent_samples(cls, fp):
        """Returns the number of independent samples stored in a file.

        The number of independent samples are counted starting from after
        burn-in. If the sampler hasn't burned in yet, then 0 is returned.

        Parameters
        -----------
        fp : InferenceFile
            An open file handler to read.

        Returns
        -------
        int
            The number of independent samples.
        """
        # check if burned in
        if not fp.is_burned_in:
            return 0
        # we'll just read a single parameter from the file
        samples = cls.read_samples(fp, fp.variable_params[0])
        return samples.size

    @classmethod
    def compute_acfs(cls, fp, start_index=None, end_index=None,
                     per_walker=False, walkers=None, parameters=None):
        """Computes the autocorrleation function of the model params in the
        given file.

        By default, parameter values are averaged over all walkers at each
        iteration. The ACF is then calculated over the averaged chain. An
        ACF per-walker will be returned instead if ``per_walker=True``.

        Parameters
        -----------
        fp : InferenceFile
            An open file handler to read the samples from.
        start_index : {None, int}
            The start index to compute the acl from. If None, will try to use
            the number of burn-in iterations in the file; otherwise, will start
            at the first sample.
        end_index : {None, int}
            The end index to compute the acl to. If None, will go to the end
            of the current iteration.
        per_walker : optional, bool
            Return the ACF for each walker separately. Default is False.
        walkers : optional, int or array
            Calculate the ACF using only the given walkers. If None (the
            default) all walkers will be used.
        parameters : optional, str or array
            Calculate the ACF for only the given parameters. If None (the
            default) will calculate the ACF for all of the model params.

        Returns
        -------
        FieldArray
            A ``FieldArray`` of the ACF vs iteration for each parameter. If
            `per-walker` is True, the FieldArray will have shape
            ``nwalkers x niterations``.
        """
        acfs = {}
        if parameters is None:
            parameters = fp.variable_params
        if isinstance(parameters, str) or isinstance(parameters, unicode):
            parameters = [parameters]
        for param in parameters:
            if per_walker:
                # just call myself with a single walker
                if walkers is None:
                    walkers = numpy.arange(fp.nwalkers)
                arrays = [cls.compute_acfs(fp, start_index=start_index,
                                           end_index=end_index,
                                           per_walker=False, walkers=ii,
                                           parameters=param)[param]
                          for ii in walkers]
                acfs[param] = numpy.vstack(arrays)
            else:
                samples = cls.read_samples(fp, param,
                                           thin_start=start_index,
                                           thin_interval=1, thin_end=end_index,
                                           walkers=walkers,
                                           flatten=False)[param]
                samples = samples.mean(axis=0)
                acfs[param] = autocorrelation.calculate_acf(samples).numpy()
        return FieldArray.from_kwargs(**acfs)

    @classmethod
    def compute_acls(cls, fp, start_index=None, end_index=None):
        """Computes the autocorrleation length for all model params in the
        given file.

        Parameter values are averaged over all walkers at each iteration.
        The ACL is then calculated over the averaged chain. If the returned ACL
        is `inf`,  will default to the number of current iterations.

        Parameters
        -----------
        fp : InferenceFile
            An open file handler to read the samples from.
        start_index : {None, int}
            The start index to compute the acl from. If None, will try to use
            the number of burn-in iterations in the file; otherwise, will start
            at the first sample.
        end_index : {None, int}
            The end index to compute the acl to. If None, will go to the end
            of the current iteration.

        Returns
        -------
        dict
            A dictionary giving the ACL for each parameter.
        """
        acls = {}
        for param in fp.variable_params:
            samples = cls.read_samples(fp, param,
                                       thin_start=start_index,
                                       thin_interval=1, thin_end=end_index,
                                       flatten=False)[param]
            samples = samples.mean(axis=0)
            acl = autocorrelation.calculate_acl(samples)
            if numpy.isinf(acl):
                acl = samples.size
            acls[param] = acl
        return acls

    @staticmethod
    def write_acls(fp, acls):
        """Writes the given autocorrelation lengths to the given file.

        The ACL of each parameter is saved to ``fp['acls/{param}']``.
        The maximum over all the parameters is saved to the file's 'acl'
        attribute.

        Parameters
        ----------
        fp : InferenceFile
            An open file handler to write the samples to.
        acls : dict
            A dictionary of ACLs keyed by the parameter.

        Returns
        -------
        ACL
            The maximum of the acls that was written to the file.
        """
        group = 'acls/{}'
        # write the individual acls
        for param in acls:
            try:
                # we need to use the write_direct function because it's
                # apparently the only way to update scalars in h5py
                fp[group.format(param)].write_direct(numpy.array(acls[param]))
            except KeyError:
                # dataset doesn't exist yet
                fp[group.format(param)] = acls[param]
        # write the maximum over all params
        fp.attrs['acl'] = numpy.array(acls.values()).max()
        return fp.attrs['acl']

    @staticmethod
    def read_acls(fp):
        """Reads the acls of all the parameters in the given file.

        Parameters
        ----------
        fp : InferenceFile
            An open file handler to read the acls from.

        Returns
        -------
        dict
            A dictionary of the ACLs, keyed by the parameter name.
        """
        group = fp['acls']
        return {param: group[param].value for param in group.keys()}


class MCMCBurnInSupport(object):
    """Provides methods for estimating burn-in."""

    def write_burn_in_iterations(fp, burn_in_iterations, is_burned_in=None):
        """Writes the burn in iterations to the given file.

        Parameters
        ----------
        fp : InferenceFile
            A file handler to an open inference file.
        burn_in_iterations : array
            Array of values giving the iteration of the burn in of each walker.
        is_burned_in : array
            Array of booleans indicating which chains are burned in.
        """
        try:
            fp['burn_in_iterations'][:] = burn_in_iterations
        except KeyError:
            fp['burn_in_iterations'] = burn_in_iterations
        fp.attrs['burn_in_iterations'] = burn_in_iterations.max()
        if is_burned_in is not None:
            try:
                fp['is_burned_in'][:] = is_burned_in
            except KeyError:
                fp['is_burned_in'] = is_burned_in
            fp.attrs['is_burned_in'] = is_burned_in.all()

