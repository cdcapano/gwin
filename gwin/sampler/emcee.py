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
This modules provides classes and functions for using the emcee sampler
packages for parameter estimation.
"""

from __future__ import absolute_import

import numpy
from pycbc.io import FieldArray
from pycbc.filter import autocorrelation

from .base import BaseMCMCSampler


#
# =============================================================================
#
#                                   Samplers
#
# =============================================================================
#

class EmceeEnsembleSampler(BaseMCMC, BaseSampler):
    """This class is used to construct an MCMC sampler from the emcee
    package's EnsembleSampler.

    Parameters
    ----------
    model : model
        A model from ``gwin.models``.
    nwalkers : int
        Number of walkers to use in sampler.
    pool : function with map, Optional
        A provider of a map function that allows a function call to be run
        over multiple sets of arguments and possibly maps them to
        cores/nodes/etc.
    """
    name = "emcee"

    def __init__(self, model, nwalkers, pool=None,
                 model_call=None):
        try:
            import emcee
        except ImportError:
            raise ImportError("emcee is not installed.")

        if model_call is None:
            model_call = model

        ndim = len(model.variable_params)
        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                        model_call,
                                        pool=pool)
        # emcee uses it's own internal random number generator; we'll set it
        # to have the same state as the numpy generator
        rstate = numpy.random.get_state()
        sampler.random_state = rstate
        self._sampler = sampler
        self._nwalkers = nwalkers

    @classmethod
    def from_cli(cls, opts, model, pool=None,
                 model_call=None):
        """Create an instance of this sampler from the given command-line
        options.

        Parameters
        ----------
        opts : ArgumentParser options
            The options to parse.
        model : LikelihoodEvaluator
            The model to use with the sampler.

        Returns
        -------
        EmceeEnsembleSampler
            An emcee sampler initialized based on the given arguments.
        """
        return cls(model, opts.nwalkers,
                   pool=pool, model_call=model_call)

    @property
    def raw_samples(self):
        """A dict mapping sampling_params to arrays of samples currently
        in memory.
        
        The arrays have shape ``nwalkers`` x ``niterations``.
        """
        # chain is a [additional dimensions x] niterations x ndim array
        samples = self.chain
        sampling_params = self.sampling_params
        # convert to dictionary to apply boundary conditions
        samples = {param: samples[..., ii] for
                   ii, param in enumerate(sampling_params)}
        samples = self.model._prior.apply_boundary_conditions(
            **samples)
        # now convert to field array
        samples = FieldArray.from_arrays([samples[param]
                                          for param in sampling_params],
                                         names=sampling_params)
        # apply transforms to go to model params space
        return self.model.apply_sampling_transforms(
            samples, inverse=True)

    @property
    def model_stats(self):
        """Returns the model stats as a FieldArray, with field names
        corresponding to the type of data returned by the model.
        The returned array has shape nwalkers x niterations. If no additional
        stats were returned to the sampler by the model, returns
        None.
        """
        stats = numpy.array(self._sampler.blobs)
        if stats.size == 0:
            return None
        # we'll force arrays to float; this way, if there are `None`s in the
        # blobs, they will be changed to `nan`s
        arrays = {field: stats[..., fi].astype(float)
                  for fi, field in
                  enumerate(self.model.metadata_fields)}
        return FieldArray.from_kwargs(**arrays).transpose()

    @property
    def lnpost(self):
        """Get the natural logarithm of the likelihood as an
        nwalkers x niterations array.
        """
        # emcee returns nwalkers x niterations
        return self._sampler.lnprobability

    @property
    def chain(self):
        """Get all past samples as an nwalker x niterations x ndim array."""
        # emcee returns the chain as nwalker x niterations x ndim
        return self._sampler.chain

    def clear_chain(self):
        """Clears the chain and blobs from memory.
        """
        # store the iteration that the clear is occuring on
        self.lastclear = self.niterations
        # now clear the chain
        self._sampler.reset()
        self._sampler.clear_blobs()

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
        p0 : array
            An nwalkers x ndim array of the initial positions that were set.
        """
        # we define set_p0 here to ensure that emcee's internal random number
        # generator is set to numpy's after the distributions' rvs functions
        # are called
        super(EmceeEnsembleSampler, self).set_p0(samples_file=samples_file,
                                                 prior=prior)
        # update the random state
        self._sampler.random_state = numpy.random.get_state()

    def write_state(self, fp):
        """Saves the state of the sampler in a file.
        """
        fp.write_random_state(state=self._sampler.random_state)

    def set_state_from_file(self, fp):
        """Sets the state of the sampler back to the instance saved in a file.
        """
        rstate = fp.read_random_state()
        # set the numpy random state
        numpy.random.set_state(rstate)
        # set emcee's generator to the same state
        self._sampler.random_state = rstate

    def run(self, niterations, **kwargs):
        """Advance the ensemble for a number of samples.

        Parameters
        ----------
        niterations : int
            Number of samples to get from sampler.

        Returns
        -------
        p : numpy.array
            An array of current walker positions with shape (nwalkers, ndim).
        lnpost : numpy.array
            The list of log posterior probabilities for the walkers at
            positions p, with shape (nwalkers, ndim).
        rstate :
            The current state of the random number generator.
        """
        pos = self._pos
        if pos is None:
            pos = self.p0
        res = self._sampler.run_mcmc(pos, niterations, **kwargs)
        p, lnpost, rstate = res[0], res[1], res[2]
        # update the positions
        self._pos = p
        return p, lnpost, rstate

    def write_results(self, fp, start_iteration=None,
                      max_iterations=None, **metadata):
        """Writes metadata, samples, model stats, and acceptance fraction
        to the given file. See the write function for each of those for
        details.

        Parameters
        -----------
        fp : InferenceFile
            A file handler to an open inference file.
        start_iteration : int, optional
            Write results to the file's datasets starting at the given
            iteration. Default is to append after the last iteration in the
            file.
        max_iterations : int, optional
            Set the maximum size that the arrays in the hdf file may be resized
            to. Only applies if the samples have not previously been written
            to file. The default (None) is to use the maximum size allowed by
            h5py.
        \**metadata :
            All other keyword arguments are passed to ``write_metadata``.
        """
        self.write_metadata(fp, **metadata)
        self.write_chain(fp, start_iteration=start_iteration,
                         max_iterations=max_iterations)
        self.write_model_stats(fp, start_iteration=start_iteration,
                               max_iterations=max_iterations)
        self.write_acceptance_fraction(fp)
        self.write_state(fp)
