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
"""This modules defines functions for reading and writing samples that the
inference samplers generate.
"""

import os
import sys
import logging
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy

import h5py

from pycbc import DYN_RANGE_FAC
from pycbc.io import FieldArray
from pycbc.types import FrequencySeries
from pycbc.waveform import parameters as wfparams

from .. import sampler as gwin_sampler

class BaseInferenceFile(h5py.File):
    """Base class for all inference hdf files.
    
    This is a subclass of the h5py.File object. It adds functions for
    handling reading and writing the samples from the samplers.

    Parameters
    -----------
    path : str
        The path to the HDF file.
    mode : {None, str}
        The mode to open the file, eg. "w" for write and "r" for read.
    """
    __metaclass__ = ABCMeta

    name = None
    samples_group = 'samples'
    sampler_group = 'sampler_info'
    data_group = 'data'
    injections_group = 'injections'

    def __init__(self, path, mode=None, **kwargs):
        super(BaseInferenceFile, self).__init__(path, mode, **kwargs)

    def __getattr__(self, attr):
        """Things stored in ``.attrs`` are promoted to instance attributes.
        
        Note that properties will be called before this, so if there are any
        properties that share the same name as something in ``.attrs``, that
        property will get returned.
        """
        return self.attrs[attr]

    @abstractmethod
    def write_samples(self, samples, **kwargs):
        """This should write all of the provided samples.

        This function should be used to write both samples and model stats.

        Parameters
        ----------
        fp : open hdf file
            The file to write to.
        samples : dict
            Samples should be provided as a dictionary of numpy arrays.
        \**kwargs :
            Any other keyword args the sampler needs to write data.
        """
        pass

    def parse_parameters(self, parameters, array_class=None):
        """Parses a parameters arg to figure out what fields need to be loaded.

        Parameters
        ----------
        parameters : (list of) strings
            The parameter(s) to retrieve. A parameter can be the name of any
            field in ``samples_group``, a virtual field or method of
            ``FieldArray`` (as long as the file contains the necessary fields
            to derive the virtual field or method), and/or a function of
            these.
        array_class : array class, optional
            The type of array to use to parse the parameters. The class must
            have a ``parse_parameters`` method. Default is to use a
            ``FieldArray``.

        Returns
        -------
        list :
            A list of strings giving the fields to load from the file.
        """
        # get the type of array class to use
        if array_class is None:
            array_class = FieldArray
        # get the names of fields needed for the given parameters
        possible_fields = self[self.samples_group].keys()
        return array_class.parse_parameters(parameters, possible_fields)

    def read_samples(self, parameters, array_class=None, **kwargs):
        """Reads samples for the given parameter(s).

        The ``parameters`` can be the name of any dataset in ``samples_group``,
        a virtual field or method of ``FieldArray`` (as long as the file
        contains the necessary fields to derive the virtual field or method),
        and/or any numpy function of these.

        The ``parameters`` are parsed to figure out what datasets are needed.
        Only those datasets will be loaded, and will be the base-level fields
        of the returned ``FieldArray``.

        The ``static_params`` are also added as attributes of the returned
        ``FieldArray``.

        Parameters
        -----------
        fp : InferenceFile
            An open file handler to read the samples from.
        parameters : (list of) strings
            The parameter(s) to retrieve.
        array_class : FieldArray-like class, optional
            The type of array to return. The class must have ``from_kwargs``
            and ``parse_parameters`` methods. If None, will return a
            ``FieldArray``.
        \**kwargs :
            All other keyword arguments are passed to ``read_raw_samples``.

        Returns
        -------
        FieldArray :
            The samples as a ``FieldArray``.
        """
        # get the type of array class to use
        if array_class is None:
            array_class = FieldArray
        # get the names of fields needed for the given parameters
        possible_fields = self[self.samples_group].keys()
        loadfields = array_class.parse_parameters(parameters, possible_fields)
        samples = self.read_raw_samples(loadfields, **kwargs)
        # convert to FieldArray
        samples = array_class.from_kwargs(**samples)
        # add the static params
        for p,val in self.static_params.items():
            setattr(samples, p, val)
        return samples

    @abstractmethod
    def read_raw_samples(self, fields, **kwargs):
        """Low level function for reading datasets in the samples group.

        This should return a dictionary of numpy arrays.
        """
        pass

    @abstractmethod
    def write_posterior(self, posterior_file, **kwargs):
        """This should write a posterior plus any other metadata to the given
        file.

        Parameters
        ----------
        posterior_file : str
            Name of the file to write to.
        \**kwargs :
            Any other keyword args the sampler needs to write the posterior.
        """
        pass

    @property
    def sampler_class(self):
        """Returns the sampler class that was used."""
        try:
            sampler = self.sampler_name
        except KeyError:
            return None
        return gwin_sampler.samplers[sampler]

    @property
    def static_params(self):
        """Returns a dictionary of the static_params. The keys are the argument
        names, values are the value they were set to.
        """
        return {arg: self.attrs[arg] for arg in self.attrs["static_params"]}

    @property
    def n_indep_samples(self):
        """Returns the number of independent samples stored in the file.
        """
        try:
            return self.attrs['n_indep_samples']
        except KeyError:
            return 0

    @property
    def cmd(self):
        """Returns the (last) saved command line.

        If the file was created from a run that resumed from a checkpoint, only
        the last command line used is returned.

        Returns
        -------
        cmd : string
            The command line that created this InferenceFile.
        """
        cmd = self.attrs["cmd"]
        if isinstance(cmd, numpy.ndarray):
            cmd = cmd[-1]
        return cmd

    def write_logevidence(self, lnz, dlnz):
        """Writes the given log evidence and its error.

        Results are saved to file's 'log_evidence' and 'dlog_evidence'
        attributes.

        Parameters
        ----------
        lnz : float
            The log of the evidence.
        dlnz : float
            The error in the estimate of the log evidence.
        """
        self.attrs['log_evidence'] = lnz
        self.attrs['dlog_evidence'] = dlnz

    @property
    def log_evidence(self):
        """Returns the log of the evidence and its error, if they exist in the
        file. Raises a KeyError otherwise.
        """
        return self.attrs["log_evidence"], self.attrs["dlog_evidence"]

    def write_random_state(self, group=None, state=None):
        """Writes the state of the random number generator from the file.

        The random state is written to ``sampler_group``/random_state.

        Parameters
        ----------
        group : str
            Name of group to write random state to.
        state : tuple, optional
            Specify the random state to write. If None, will use
            ``numpy.random.get_state()``.
        """
        group = self.sampler_group if group is None else group
        dataset_name = "/".join([group, "random_state"])
        if state is None:
            state = numpy.random.get_state()
        s, arr, pos, has_gauss, cached_gauss = state
        if group in self:
            self[dataset_name][:] = arr
        else:
            self.create_dataset(dataset_name, arr.shape, fletcher32=True,
                                dtype=arr.dtype)
            self[dataset_name][:] = arr
        self[dataset_name].attrs["s"] = s
        self[dataset_name].attrs["pos"] = pos
        self[dataset_name].attrs["has_gauss"] = has_gauss
        self[dataset_name].attrs["cached_gauss"] = cached_gauss

    def read_random_state(self, group=None):
        """Reads the state of the random number generator from the file.

        Parameters
        ----------
        group : str
            Name of group to read random state from.

        Returns
        -------
        tuple
            A tuple with 5 elements that can be passed to numpy.set_state.
        """
        group = self.sampler_group if group is None else group
        dataset_name = "/".join([group, "random_state"])
        arr = self[dataset_name][:]
        s = self[dataset_name].attrs["s"]
        pos = self[dataset_name].attrs["pos"]
        has_gauss = self[dataset_name].attrs["has_gauss"]
        cached_gauss = self[dataset_name].attrs["cached_gauss"]
        return s, arr, pos, has_gauss, cached_gauss

    def write_strain(self, strain_dict, group=None):
        """Writes strain for each IFO to file.

        Parameters
        -----------
        strain : {dict, FrequencySeries}
            A dict of FrequencySeries where the key is the IFO.
        group : {None, str}
            The group to write the strain to. If None, will write to the top
            level.
        """
        subgroup = self.data_group + "/{ifo}/strain"
        if group is None:
            group = subgroup
        else:
            group = '/'.join([group, subgroup])
        for ifo, strain in strain_dict.items():
            self[group.format(ifo=ifo)] = strain
            self[group.format(ifo=ifo)].attrs['delta_t'] = strain.delta_t
            self[group.format(ifo=ifo)].attrs['start_time'] = \
                float(strain.start_time)

    def write_stilde(self, stilde_dict, group=None):
        """Writes stilde for each IFO to file.

        Parameters
        -----------
        stilde : {dict, FrequencySeries}
            A dict of FrequencySeries where the key is the IFO.
        group : {None, str}
            The group to write the strain to. If None, will write to the top
            level.
        """
        subgroup = self.data_group + "/{ifo}/stilde"
        if group is None:
            group = subgroup
        else:
            group = '/'.join([group, subgroup])
        for ifo, stilde in stilde_dict.items():
            self[group.format(ifo=ifo)] = stilde
            self[group.format(ifo=ifo)].attrs['delta_f'] = stilde.delta_f
            self[group.format(ifo=ifo)].attrs['epoch'] = float(stilde.epoch)

    def write_psd(self, psds, group=None):
        """Writes PSD for each IFO to file.

        Parameters
        -----------
        psds : {dict, FrequencySeries}
            A dict of FrequencySeries where the key is the IFO.
        group : {None, str}
            The group to write the psd to. Default is ``data_group``.
        """
        subgroup = self.data_group + "/{ifo}/psds/0"
        if group is None:
            group = subgroup
        else:
            group = '/'.join([group, subgroup])
        for ifo in psds:
            self[group.format(ifo=ifo)] = psds[ifo]
            self[group.format(ifo=ifo)].attrs['delta_f'] = psds[ifo].delta_f

    def write_injections(self, injection_file):
        """Writes injection parameters from the given injection file.

        Everything in the injection file is copied to ``injections_group``.

        Parameters
        ----------
        injection_file : str
            Path to HDF injection file.
        """
        try:
            with h5py.File(injection_file, "r") as fp:
                super(BaseInferenceFile, self).copy(fp, self.injections_group)
        except IOError:
            logging.warn("Could not read %s as an HDF file", injection_file)

    def write_command_line(self):
        """Writes command line to attributes.

        The command line is written to the file's ``attrs['cmd']``. If this
        attribute already exists in the file (this can happen when resuming
        from a checkpoint), ``attrs['cmd']`` will be a list storing the current
        command line and all previous command lines.
        """
        cmd = [" ".join(sys.argv)]
        try:
            previous = self.attrs["cmd"]
            if isinstance(previous, str):
                # convert to list
                previous = [previous]
            elif isinstance(previous, numpy.ndarray):
                previous = previous.tolist()
        except KeyError:
            previous = []
        self.attrs["cmd"] = cmd + previous

    def get_slice(self, thin_start=None, thin_interval=None, thin_end=None):
        """Formats a slice using the given arguments that can be used to
        retrieve a thinned array from an InferenceFile.

        Parameters
        ----------
        thin_start : {None, int}
            The starting index to use. If None, will try to retrieve the
            `burn_in_iterations` from the given file. If no
            `burn_in_iterations` exists, will default to the start of the
            array.
        thin_interval : {None, int}
            The interval to use. If None, will try to retrieve the acl from the
            given file. If no acl attribute exists, will default to 1.
        thin_end : {None, int}
            The end index to use. If None, will retrieve to the end of the
            array.

        Returns
        -------
        slice :
            The slice needed.
        """

        # default is to skip burn in samples
        if thin_start is None:
            try:
                thin_start = self.burn_in_iterations
                # if the sampler hasn't burned in, the burn_in_iterations will
                # be the same as the number of iterations, which would result
                # in 0 samples. In that case, just use the last one
                if thin_start == self.niterations:
                    thin_start = thin_start - 1
            except KeyError:
                pass

        # default is to use stored ACL and accept every i-th sample
        if thin_interval is None:
            try:
                thin_interval = int(numpy.ceil(self.acl))
            except KeyError:
                pass
        return slice(thin_start, thin_end, thin_interval)

    def copy_metadata(self, other):
        """Copies all metadata from this file to the other file.

        Metadata is defined as everything in the top-level ``.attrs``.

        Parameters
        ----------
        other : InferenceFile
            An open inference file to write the data to.
        """
        logging.info("Copying metadata")
        # copy attributes
        for key in self.attrs.keys():
            other.attrs[key] = self.attrs[key]

    def copy_info(self, other, ignore=None):
        """Copies "info" from this file to the other.

        "Info" is defined all groups that are not the samples group.

        Parameters
        ----------
        other : output file
            The output file. Must be an hdf file.
        ignore : (list of) str
            Don't copy the given groups.
        """
        logging.info("Copying info")
        # copy non-samples/stats data
        if ignore is None:
            ignore = []
        if isinstance(ignore, (str, unicode)):
            ignore = [ignore]
        ignore = set(ignore + [self.samples_group])
        copy_groups = set(self.keys()) - ignore
        for key in copy_groups:
            super(BaseInferenceFile, self).copy(key, other)

    def copy_samples(self, other, parameters=None, parameter_names=None,
                     read_args=None, write_args=None):
        """Should copy samples to the other files.

        Parameters
        ----------
        other : InferenceFile
            An open inference file to write to.
        parameters : list of str, optional
            List of parameters to copy. If None, will copy all parameters.
        parameter_names : dict, optional
            Rename one or more parameters to the given name. The dictionary
            should map parameter -> parameter name. If None, will just use the
            original parameter names.
        read_args : dict, optional
            Arguments to pass to ``read_samples``.
        write_args : dict, optional
            Arguments to pass to ``write_samples``.
        """
        # select the samples to copy
        logging.info("Reading samples to copy")
        if parameters is None:
            parameters = self.variable_params
        # if list of desired parameters is different, rename
        if set(parameters) != set(self.variable_params):
            other.attrs['variable_params'] = parameters
        samples = self.read_samples(parameters, **read_args)
        logging.info("Copying {} samples".format(samples.size))
        # if different parameter names are desired, get them from the samples
        if parameter_names:
            arrs = {pname: samples[p] for p, pname in parameter_names.items()}
            arrs.update({p: samples[p] for p in parameters if
                         p not in parameter_names})
            samples = FieldArray.from_kwargs(**arrs)
            other.attrs['variable_params'] = samples.fieldnames
        logging.info("Writing samples")
        other.write_samples(other, samples, **write_args)

    def copy(self, other, ignore=None, parameters=None, parameter_names=None,
             read_args=None, write_args=None):
        """Copies metadata, info, and samples in this file to another file.

        Parameters
        ----------
        other : str or InferenceFile
            The file to write to. May be either a string giving a filename,
            or an open hdf file. If the former, the file will be opened with
            the write attribute (note that if a file already exists with that
            name, it will be deleted).
        ignore : (list of) strings
            Don't copy the given groups. If the samples group is included, no
            samples will be copied.
        parameters : list of str, optional
            List of parameters in the samples group to copy. If None, will copy
            all parameters.
        parameter_names : dict, optional
            Rename one or more parameters to the given name. The dictionary
            should map parameter -> parameter name. If None, will just use the
            original parameter names.
        read_args : dict, optional
            Arguments to pass to ``read_samples``.
        write_args : dict, optional
            Arguments to pass to ``write_samples``.

        Returns
        -------
        InferenceFile
            The open file handler to other.
        """
        if not isinstance(other, h5py.File):
            # check that we're not trying to overwrite this file
            if other == self.name:
                raise IOError("destination is the same as this file")
            other = InferenceFile(other, 'w')
        # metadata
        self.copy_metadata(other)
        # info
        if ignore is None:
            ignore = []
        if isinstance(ignore, (str, unicode)):
            ignore = [ignore]
        self.copy_info(other, ignore=ignore)
        # samples
        if self.samples_group not in ignore:
            self.copy_samples(other, parameters=parameters,
                              parameter_names=parameter_names,
                              read_args=read_args,
                              write_args=write_args)
        # if any down selection was done, re-set the burn in iterations and
        # the acl, and the niterations.
        # The last dimension of the samples returned by the sampler should
        # be the number of iterations.
        #if samples.shape[-1] != self.niterations:
        #    other.attrs['acl'] = 1
        #    other.attrs['burn_in_iterations'] = 0
        #    other.attrs['niterations'] = samples.shape[-1]
        #return other


def write_kwargs_to_hdf_attrs(attrs, **kwargs):
    """Writes the given keywords to the given ``attrs``.
    
    If any keyword argument points to a dict, the keyword will point to a
    list of the dict's keys. Each key is then written to the attrs with its
    corresponding value.

    Parameters
    ----------
    attrs : an HDF attrs
        Can be either the ``attrs`` of the hdf file, or any group in a file.
    \**kwargs :
        The keywords to write.
    """
    for arg, val in kwargs.items():
        if val is None:
            val = str(None)
        if isinstance(val, dict):
            attrs[arg] = val.keys()
            # just call self again with the dict as kwargs
            write_kwargs_to_hdf_attrs(attrs, **val)
        else:
            attrs[arg] = val


def check_integrity(filename):
    """Checks the integrity of an InferenceFile.

    Checks done are:

        * can the file open?
        * do all of the datasets in the samples group have the same shape?
        * can the first and last sample in all of the datasets in the samples
          group be read?

    If any of these checks fail, an IOError is raised.

    Parameters
    ----------
    filename: str
        Name of an InferenceFile to check.

    Raises
    ------
    ValueError
        If the given file does not exist.
    KeyError
        If the samples group does not exist.
    IOError
        If any of the checks fail.
    """
    # check that the file exists
    if not os.path.exists(filename):
        raise ValueError("file {} does not exist".format(filename))
    # if the file is corrupted such that it cannot be opened, the next line
    # will raise an IOError
    with InferenceFile(filename, 'r') as fp:
        # check that all datasets in samples have the same shape
        parameters = fp[fp.samples_group].keys()
        group = fp.samples_group + '/{}'
        # use the first parameter as a reference shape
        ref_shape = fp[group.format(parameters[0])].shape
        if not all(fp[group.format(param)].shape == ref_shape
                   for param in parameters):
            raise IOError("not all datasets in the samples group have the "
                          "same shape")
        # check that we can read the first/last sample
        firstidx = tuple([0]*len(ref_shape))
        lastidx = tuple([-1]*len(ref_shape))
        for param in parameters:
            fp[group.format(param)][firstidx]
            fp[group.format(param)][lastidx]
