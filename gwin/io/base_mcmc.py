# Copyright (C) 2016 Christopher M. Biwer, Collin Capano
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# self.option) any later version.
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
"""Provides I/O that is specific to MCMC samplers.
"""

from __future__ import absolute_import

from abc import (ABCMeta, abstractmethod)

import numpy
from .base_hdf import write_kwargs_to_hdf_attrs

class MCMCIO(object):
    """Abstract base class that provides some IO functions for ensemble MCMCs.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def read_acls(self):
        """Should return all of the individual chains' acls.
        """
        pass

    def write_samples(self, samples, parameters=None,
                      start_iteration=None, max_iterations=None):
        """Writes samples to the given file.

        Results are written to:

            ``fp[samples_group/{vararg}]``,

        where ``{vararg}`` is the name of a model params. The samples are
        written as an ``nwalkers x niterations`` array.

        Parameters
        -----------
        samples : dict
            The samples to write. Each array in the dictionary should have
            shape nwalkers x niterations.
        parameters : list, optional
            Only write the specified parameters to the file. If None, will
            write all of the keys in the ``samples`` dict.
        start_iteration : int, optional
            Write results to the file's datasets starting at the given
            iteration. Default is to append after the last iteration in the
            file.
        max_iterations : int, optional
            Set the maximum size that the arrays in the hdf file may be resized
            to. Only applies if the samples have not previously been written
            to file. The default (None) is to use the maximum size allowed by
            h5py.
        """
        nwalkers, niterations = samples.values()[0].shape
        assert all(p.shape == (nwalkers, niterations)
                   for p in samples.values()), (
               "all samples must have the same shape")
        if max_iterations is not None and max_iterations < niterations:
            raise IndexError("The provided max size is less than the "
                             "number of iterations")
        group = self.samples_group + '/{name}'
        if parameters is None:
            parameters = samples.keys()
        # loop over number of dimensions
        for param in parameters:
            dataset_name = group.format(name=param)
            istart = start_iteration
            try:
                fp_niterations = self[dataset_name].shape[-1]
                if istart is None:
                    istart = fp_niterations
                istop = istart + niterations
                if istop > fp_niterations:
                    # resize the dataset
                    self[dataset_name].resize(istop, axis=1)
            except KeyError:
                # dataset doesn't exist yet
                if istart is not None and istart != 0:
                    raise ValueError("non-zero start_iteration provided, "
                                     "but dataset doesn't exist yet")
                istart = 0
                istop = istart + niterations
                self.create_dataset(dataset_name, (nwalkers, istop),
                                    maxshape=(nwalkers, max_iterations),
                                    dtype=float, fletcher32=True)
            self[dataset_name][:, istart:istop] = samples[param]

    def read_raw_samples(self, fields,
                         thin_start=None, thin_interval=None, thin_end=None,
                         iteration=None, walkers=None, flatten=True):
        """Base function for reading samples.

        Parameters
        -----------
        fields : list
            The list of field names to retrieve. Must be names of datasets in
            the ``samples_group``.

        Returns
        -------
        dict
            A dictionary of field name -> numpy array pairs.
        """
        # walkers to load
        if walkers is not None:
            widx = numpy.zeros(fp.nwalkers, dtype=bool)
            widx[walkers] = True
        else:
            widx = slice(0, None)
        # get the slice to use
        if iteration is not None:
            get_index = iteration
        else:
            get_index = self.get_slice(thin_start=thin_start,
                                       thin_end=thin_end,
                                       thin_interval=thin_interval)
        # load
        group = self.samples_group + '/{name}'
        arrays = {}
        for name in fields:
            arr = self[group.format(name=name)][widx, get_index]
            if flatten:
                arr = arr.flatten()
            arrays[name] = arr
        return arrays

    def write_resume_point(self):
        """Keeps a list of the number of iterations that were in a file when a
        run was resumed from a checkpoint."""
        try:
            resume_pts = self.attrs["resume_points"].tolist()
        except KeyError:
            resume_pts = []
        try:
            niterations = self.niterations
        except KeyError:
            niterations = 0
        resume_pts.append(niterations)
        self.attrs["resume_points"] = resume_pts

    def write_niterations(self, niterations):
        """Writes the given number of iterations to the sampler group."""
        self[self.sampler_group].attrs['niterations'] = niterations

    @property
    def niterations(self):
        """Returns the number of iterations the sampler was run for."""
        return self[self.sampler_group].attrs['niterations']

    def write_sampler_metadata(self, sampler):
        """Writes the sampler's metadata."""
        self.attrs['sampler'] = sampler.name
        if self.sampler_group not in self.keys():
            # create the sampler group
            self.create_group(self.sampler_group)
        self[self.sampler_group].attrs['nwalkers'] = sampler.nwalkers
        # write the model's metadata
        sampler.model.write_metadata(self)
        

    def write_acls(self, acls):
        """Writes the given autocorrelation lengths.

        The ACL of each parameter is saved to
        ``[sampler_group]/acls/{param}']``.  The maximum over all the
        parameters is saved to the file's 'acl' attribute.

        Parameters
        ----------
        acls : dict
            A dictionary of ACLs keyed by the parameter.

        Returns
        -------
        ACL
            The maximum of the acls that was written to the file.
        """
        group = self.sampler_group + '/acls/{}'
        # write the individual acls
        for param in acls:
            try:
                # we need to use the write_direct function because it's
                # apparently the only way to update scalars in h5py
                self[group.format(param)].write_direct(
                    numpy.array(acls[param]))
            except KeyError:
                # dataset doesn't exist yet
                self[group.format(param)] = acls[param]
        # write the maximum over all params
        acl = numpy.array(acls.values()).max()
        self[self.sampler_group].attrs['acl'] = acl
        # set the default thin interval to be the acl (if it is finite)
        if numpy.isfinite(acl):
            self.attrs['thin_interval'] = acl

    def read_acls(self):
        """Reads the acls of all the parameters.

        Parameters
        ----------
        fp : InferenceFile
            An open file handler to read the acls from.

        Returns
        -------
        dict
            A dictionary of the ACLs, keyed by the parameter name.
        """
        group = self[self.sampler_group]['acls']
        return {param: group[param].value for param in group.keys()}

    def write_burn_in(self, burn_in):
        """Write the given burn-in data to the given filename."""
        group = self[self.sampler_group]
        group.attrs['burn_in_test'] = burn_in.burn_in_test
        group.attrs['is_burned_in'] = burn_in.is_burned_in
        group.attrs['burn_in_iteration'] = burn_in.burn_in_iteration
        # set the defaut thin_start to be the burn_in_iteration
        self.attrs['thin_start'] = burn_in.burn_in_iteration
        # write individual test data
        for tst in burn_in.burn_in_data:
            key = 'burn_in_tests/{}'.format(tst)
            try:
                attrs = group[key].attrs
            except KeyError:
                group.create_group(key)
                attrs = group[key].attrs
            write_kwargs_to_hdf_attrs(attrs, **burn_in.burn_in_data[tst])
