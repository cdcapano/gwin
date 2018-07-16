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
"""Provides constructor classes and convenience functions for MCMC samplers."""

from abc import ABCMeta, abstractmethod, abstractproperty
import logging
import numpy

#
# =============================================================================
#
#                              Convenience functions
#
# =============================================================================
#
def raw_samples_to_dict(sampler, raw_samples):
    """Convenience function for converting ND array to a dict of samples.

    The samples are assumed to have dimension
    ``[sampler.base_shape x] niterations x len(sampler.sampling_params)``.

    Parameters
    ----------
    sampler : sampler instance
        An instance of an MCMC sampler.
    raw_samples : array
        The array of samples to convert.

    Returns
    -------
    dict :
        A dictionary mapping the raw samples to the variable params. If the
        sampling params are not the same as the variable params, they will
        also be included. Each array will have shape
        ``[sampler.base_shape x] niterations``.
    """
    sampling_params = sampler.sampling_params
    # convert to dictionary
    samples = {param: raw_samples[..., ii] for
               ii, param in enumerate(sampling_params)}
    # apply boundary conditions
    samples = sampler.model.prior_distribution.apply_boundary_conditions(
        **samples)
    # apply transforms to go to model's variable params space
    return sampler.model.sampling_transforms.apply(samples, inverse=True)


def raw_stats_to_dict(sampler, raw_stats):
    """Converts an ND array of model stats to a dict.

    The ``raw_stats`` may either be a numpy array or a list. If the
    former, the stats are assumed to have shape
    ``[sampler.base_shape x] niterations x nstats, where nstats are the number
    of stats returned by ``sampler.model.default_stats``. If the latter, the
    list is cast to an array that is assumed to be the same shape as if an
    array was given.

    Parameters
    ----------
    sampler : sampler instance
        An instance of an MCMC sampler.
    raw_stats : array or list
        The stats to convert.

    Returns
    -------
    dict :
        A dictionary mapping the model's ``default_stats`` to arrays of values.
        Each array will have shape ``[sampler.base_shape x] niterations``.
    """
    if not isinstance(raw_stats, numpy.ndarray):
        # Assume list. Since the model returns a tuple of values, this should
        # be a [sampler.base_shape x] x niterations list of tuples. We can
        # therefore immediately convert this to a ND array.
        raw_stats = numpy.array(raw_stats)
    return {stat: raw_stats[..., ii]
            for (ii, stat) in enumerate(self.model.default_stats)}

#
# =============================================================================
#
#                              BaseMCMC definition
#
# =============================================================================
#


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

    _lastclear = None
    _itercounter = None
    _pos = None
    _p0 = None
    _nwalkers = None

    @abstractproperty(self):
    def base_shape(self):
        """What shape the sampler's samples arrays are in, excluding 
        the iterations dimension.
        
        For example, if a sampler uses 20 walkers and 3 temperatures, this
        would be ``(3, 20)``. If a sampler only uses a single walker and no
        temperatures this would be ``()``.
        """
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
        lastclear = self._lastclear
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
                assert(samples.shape[:-1] == self.samples_shape,
                       "samples in file {} have shape {}, but I have shape {}".
                       format(samples_file, samples.shape, self.samples_shape))
            # transform to sampling parameter space
            samples = self.model.sampling_transforms.apply(samples)
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

    def set_initial_conditions(self, initial_distribution=None,
                               samples_file=None):
        """Sets the initial starting point for the MCMC.

        If a starting samples file is provided, will also load the random
        state from it.
        """
        self.set_p0(samples_file=samples_file, prior=initial_distribution)
        # if a samples file was provided, use it to set the state of the
        # sampler
        if samples_file is not None:
            self.set_state_from_file(samples_file)

    @abstractmethod
    def set_state_from_file(self, filename):
        """Sets the state of the sampler to the instance saved in a file.
        """
        pass

    @abstractmethod
    def write_state(self, filename):
        """Saves the state of the sampler to the given file.
        """
        pass

    def run(self):
        """Runs the sampler."""

        if self.require_indep_samples and self.checkpoint_interval is None:
            raise ValueError("A checkpoint interval must be set if "
                             "independent samples are required")
        # get the starting number of samples:
        # "nsamples" keeps track of the number of samples we've obtained (if
        # require_indep_samples is used, this is the number of independent
        # samples; otherwise, this is the total number of samples).
        # "startiter" is the number of iterations that the file already contains
        # (either due to sampler burn-in, or a previous checkpoint)
        try:
            with self.io(self.checkpoint_file, "r") as fp:
                start = fp.niterations
        except KeyError:
            startiter = 0
        if self.require_indep_samples:
            with self.io(self.checkpoint_file, "r") as fp:
                nsamples = fp.n_indep_samples
        else:
            # the number of samples is the number of iterations times the
            # number of walkers
            nsamples = startiter * self.nwalkers
        # to ensure iterations are counted properly, the sampler's lastclear
        # should be the same as start
        self._lastclear = startiter
        # keep track of the number of iterations we've done
        self._itercounter = startiter
        # figure out the interval to use
        iterinterval = self.checkpoint_interval
        if iterinterval is None:
            iterinterval = int(numpy.ceil(
                float(self.target_nsamples) / self.nwalkers))
        # run sampler until we have the desired number of samples
        while nsamples < self.target_nsamples:
            enditer = startiter + iterinterval
            # adjust the interval if we would go past the number of iterations
            endnsamp = enditer * self.nwalkers
            if endnsamp > self.target_nsamples \
                    and not self.require_indep_samples:
                iterinterval = int(numpy.ceil(
                    (endnsamp - self.target_nsamples) / self.nwalkers))
            # run sampler and set initial values to None so that sampler
            # picks up from where it left off next call
            logging.info("Running sampler for {} to {} iterations".format(
                startiter, enditer))
            # run the underlying sampler for the desired interval
            self.run_mcmc(iterinterval)
            # dump the current results
            self.checkpoint()
            # update nsamples for next loop
            if self.require_indep_samples:
                nsamples = self.n_indep_samples
                logging.info("Have {} independent samples".format(nsamples))
            else:
                nsamples += iterinterval * self.nwalkers
            self._itercounter = startiter = enditer

    @abstractproperty
    def n_indep_samples(self):
        """Should return the number of independent samples the sampler has
        acquired so far."""
        pass

    @abstractmethod
    def run_mcmc(self, niterations):
        """Run the MCMC for the given number of iterations."""
        pass

    def checkpoint(self):
        """Dumps current samples to the checkpoint file."""
        # write new samples
        logging.info("Writing samples to file")
        self.write_results(self.checkpoint_file)
        # write other stuff
        with self.io(checkpoint_file, "a") as fp:
            # write the current number of iterations
            fp.attrs['niterations'] = self.niterations
            # FIXME
            # logging.info("Updating burn in")
            # burnidx, is_burned_in = burn_in_eval.update(self, fp)

            # compute the acls and write
            acls = None
            if self.require_indep_samples:
                logging.info("Computing acls")
                acls = self.compute_acls(fp)
                sampler.write_acls(fp, acls)

        # write to backup
        with InferenceFile(backup_file, "a") as fp:

            logging.info("Writing to backup file")
            sampler.write_results(fp,
                                  ifos=opts.instruments)
            sampler.write_burn_in_iterations(fp, burnidx, is_burned_in)
            if acls is not None:
                sampler.write_acls(fp, acls)

        # check validity
        checkpoint_valid = validate_checkpoint_files(checkpoint_file,
                                                     backup_file)
        if not checkpoint_valid:
            raise IOError("error writing to checkpoint file")

        # clear the in-memory chain to save memory
        logging.info("Clearing chain")
        self.clear_chain()


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

