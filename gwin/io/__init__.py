# Copyright (C) 2018  Collin Capano
#
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

"""I/O utilities for GWIn
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import argparse
import shutil
import textwrap
import numpy
import logging
import h5py as _h5py
from pycbc.io.record import FieldArray, _numpy_function_lib
from pycbc import waveform as _waveform

from .emcee import EmceeFile
from .txt import InferenceTXTFile

filetypes = {
    EmceeFile.name: EmceeFile,
}


def get_file_type(filename):
    """ Returns I/O object to use for file.

    Parameters
    ----------
    filename : str
        Name of file.

    Returns
    -------
    file_type : {InferenceFile, InferenceTXTFile}
        The type of inference file object to use.
    """
    txt_extensions = [".txt", ".dat", ".csv"]
    hdf_extensions = [".hdf", ".h5"]
    for ext in hdf_extensions:
        if filename.endswith(ext):
            with _h5py.File(path, 'r') as fp:
                filetype = fp.attrs['filetype']
            return filetypes[filetype]
    for ext in txt_extensions:
        if filename.endswith(ext):
            return InferenceTXTFile
    raise TypeError("Extension is not supported.")


def loadfile(path, mode=None, filetype=None, **kwargs):
    """Loads the given file using the appropriate InferenceFile class.

    If ``filetype`` is not provided, this will try to retreive the ``filetype``
    from the file's ``attrs``. If the file does not exist yet, an IOError will
    be raised if ``filetype`` is not provided.

    Parameters
    ----------
    path : str
        The filename to load.
    mode : str, optional
        What mode to load the file with, e.g., 'w' for write, 'r' for read,
        'a' for append. Default will default to h5py.File's mode, which is 'a'.
    filetype : str, optional
        Force the file to be loaded with the given class name. This must be
        provided if creating a new file.

    Returns
    -------
    filetype instance
        An open file handler to the file. The class used for IO with the file
        is determined by the ``filetype`` keyword (if provided) or the
        ``filetype`` stored in the file (if not provided).
    """
    if filetype is None:
        # try to read the file to get its filetype
        try:
            fileclass = get_file_type(path)
        except IOError:
            # file doesn't exist, filetype must be provided
            raise IOError("The file appears not to exist. In this case, "
                          "filetype must be provided.")
    else:
        fileclass = filetypes[filetype]
    return fileclass(path, mode=mode, **kwargs)


#
# =============================================================================
#
#                         HDF Utilities
#
# =============================================================================
#


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
    with loadfile(filename, 'r') as fp:
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


def validate_checkpoint_files(checkpoint_file, backup_file):
    """Checks if the given checkpoint and/or backup files are valid.

    The checkpoint file is considered valid if:

        * it passes all tests run by ``check_integrity``;
        * it has at least one sample written to it (indicating at least one
          checkpoint has happened).

    The same applies to the backup file. The backup file must also have the
    same number of samples as the checkpoint file, otherwise, the backup is
    considered invalid.

    If the checkpoint (backup) file is found to be valid, but the backup
    (checkpoint) file is not valid, then the checkpoint (backup) is copied to
    the backup (checkpoint). Thus, this function ensures that checkpoint and
    backup files are either both valid or both invalid.

    Parameters
    ----------
    checkpoint_file : string
        Name of the checkpoint file.
    backup_file : string
        Name of the backup file.

    Returns
    -------
    checkpoint_valid : bool
        Whether or not the checkpoint (and backup) file may be used for loading
        samples.
    """
    # check if checkpoint file exists and is valid
    try:
        check_integrity(checkpoint_file)
        checkpoint_valid = True
    except (ValueError, KeyError, IOError):
        checkpoint_valid = False
    # backup file
    try:
        check_integrity(backup_file)
        backup_valid = True
    except (ValueError, KeyError, IOError):
        backup_valid = False
    # check if there are any samples in the file; if not, we'll just start from
    # scratch
    if checkpoint_valid:
        with loadfile(checkpoint_file, 'r') as fp:
            try:
                group = '{}/{}'.format(fp.samples_group, fp.variable_params[0])
                nsamples = fp[group].size
                checkpoint_valid = nsamples != 0
            except KeyError:
                checkpoint_valid = False
    # check if there are any samples in the backup file
    if backup_valid:
        with loadfile(backup_file, 'r') as fp:
            try:
                group = '{}/{}'.format(fp.samples_group, fp.variable_params[0])
                backup_nsamples = fp[group].size
                backup_valid = backup_nsamples != 0
            except KeyError:
                backup_valid = False
    # check that the checkpoint and backup have the same number of samples;
    # if not, assume the checkpoint has the correct number
    if checkpoint_valid and backup_valid:
        backup_valid = nsamples == backup_nsamples
    # decide what to do based on the files' statuses
    if checkpoint_valid and not backup_valid:
        # copy the checkpoint to the backup
        logging.info("Backup invalid; copying checkpoint file")
        shutil.copy(checkpoint_file, backup_file)
        backup_valid = True
    elif backup_valid and not checkpoint_valid:
        logging.info("Checkpoint invalid; copying backup file")
        # copy the backup to the checkpoint
        shutil.copy(backup_file, checkpoint_file)
        checkpoint_valid = True
    return checkpoint_valid


#
# =============================================================================
#
#                         Command-line Utilities
#
# =============================================================================
#
class ParseParametersArg(argparse.Action):
    """Argparse action that will parse parameters and labels from an opton.

    This assumes that the values set on the command line for its assigned
    argument are strings formatted like ``PARAM[:LABEL]``. When the arguments
    are parsed, the ``LABEL`` bit is stripped off and added to a dictionary
    mapping ``PARAM -> LABEL``. This dictionary is stored to the parsed
    namespace called ``{dest}_labels``, where ``{dest}`` is the argument's
    ``dest`` setting (by default, this is the same as the option string).
    Likewise, the argument's ``dest`` in the parsed namespace is updated so
    that it is just ``PARAM``.

    If no ``LABEL`` is provided, then ``PARAM`` will be used for ``LABEL``.

    If ``LABEL`` is a known parameter in ``pycbc.waveform.parameters``, then
    the label attribute there will be used in the ``parameter_labels``.
    Otherwise, ``LABEL`` will be used.

    This action can work on arguments that have ``nargs != 0`` and ``type`` set
    to ``str``.

    Examples
    --------
    Create a parser and add two arguments that use this action (note that the
    first argument accepts multiple inputs while the second only accepts a
    single input):

    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument('--parameters', type=str, nargs="+",
                            action=ParseParametersArg)
    >>> parser.add_argument('--z-arg', type=str, action=ParseParametersArg)

    Parse a command line that uses these options:

    >>> import shlex
    >>> cli = "--parameters 'mass1+mass2:mtotal' ra ni --z-arg foo:bar"
    >>> opts = parser.parse_args(shlex.split(cli))
    >>> opts.parameters
    ['mass1+mass2', 'ra', 'ni']
    >>> opts.parameters_labels
    {'mass1+mass2': '$M~(\\mathrm{M}_\\odot)$', 'ni': 'ni', 'ra': '$\\alpha$'}
    >>> opts.z_arg
    'foo'
    >>> opts.z_arg_labels
    {'foo': 'bar'}

    In the above, the first argument to ``--parameters`` was ``mtotal``. Since
    this is a recognized parameter in ``pycbc.waveform.parameters``, the label
    dictionary contains the latex string associated with the ``mtotal``
    parameter. A label was not provided for the second argument, and so ``ra``
    was used. Since ``ra`` is also a recognized parameter, its associated latex
    string was used in the labels dictionary. Since ``ni`` and ``bar`` (the
    label for ``z-arg``) are not recognized parameters, they were just used
    as-is in the labels dictionaries.
    """
    def __init__(self, type=str, nargs=None, **kwargs):
        # check that type is string
        if type != str:
            raise ValueError("the type for this action must be a string")
        if nargs == 0:
            raise ValueError("nargs must not be 0 for this action")
        super(ParseParametersArg, self).__init__(type=type, nargs=nargs,
                                                 **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        singlearg = isinstance(values, (str, unicode))
        if singlearg:
            values = [values]
        params = []
        labels = {}
        for param in values:
            psplit = param.split(':')
            if len(psplit) == 2:
                param, label = psplit
            else:
                label = param
            # try to get the label from waveform.parameters
            try:
                label = getattr(_waveform.parameters, label).label
            except AttributeError:
                pass
            labels[param] = label
            params.append(param)
        # update the namespace
        if singlearg:
            params = params[0]
        setattr(namespace, self.dest, params)
        setattr(namespace, '{}_labels'.format(self.dest), labels)


class PrintFileParams(argparse.Action):
    """Argparse action that will load input files and print possible parameters
    to screen. Once this is done, the program is forced to exit immediately.

    The behvior is similar to --help, except that the input-file is read.
    """
    def __init__(self, nargs=0, **kwargs):
        if nargs != 0:
            raise ValueError("nargs for this action must be 0")
        super(PrintFileParams, self).__init__(nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # get the input file(s)
        input_files = namespace.input_file
        if input_files is None:
            raise ValueError("One ore more --input-file must be provided. "
                             "If input-file was specified, make sure the "
                             "option to print the file help is after "
                             "the list of files.")
        parameters = []
        filesbytype = {}
        fileparsers = {}
        for fn in input_files:
            fp = loadfile(fn, 'r')
            parameters.append(set(fp[fp.samples_group].keys()))
            try:
                filesbytype[fp.name].append(fn)
            except KeyError:
                filesbytype[fp.name] = [fn]
                # get any extra options
                fileparsers[fp.name] = fp.extra_args_parser(add_help=False)
            fp.close()
        # now print information about the parameters
        # take the intersection of all parameters
        parameters = set.intersection(*parameters)
        print("\n"+textwrap.fill("Parameters available with this (these) input "
                                 "file(s):"), end="\n\n")
        print(textwrap.fill(' '.join(sorted(parameters))),
              end="\n\n")
        # information about the pycbc functions
        pfuncs = sorted(FieldArray.functionlib.fget(FieldArray).keys())
        print(textwrap.fill("Available pycbc functions (see "
                            "http://pycbc.org/pycbc/latest/html for "
                            "more details):"), end="\n\n") 
        print(textwrap.fill(', '.join(pfuncs)), end="\n\n")
        # numpy funcs
        npfuncs = sorted([name for name,obj in _numpy_function_lib.items()
                          if isinstance(obj, numpy.ufunc)])
        print(textwrap.fill("Available numpy functions:"),
              end="\n\n")
        print(textwrap.fill(', '.join(npfuncs)), end="\n\n")
        # misc
        consts = "e euler_gamma inf nan pi"
        print(textwrap.fill("Recognized constants:"),
              end="\n\n")
        print(consts, end="\n\n")
        print(textwrap.fill("Python arthimetic (+ - * / // ** %), "
                            "binary (&, |, etc.), and comparison (>, <, >=, "
                            "etc.) operators may also be used."), end="\n\n")
        # print out the extra arguments that may be used
        outstr = textwrap.fill("The following are additional command-line "
                               "options that may be provided, along with the "
                               "input files that understand them:")
        print("\n"+outstr, end="\n\n")
        for ftype, fparser in fileparsers.items():
            fnames = ', '.join(filesbytype[ftype])
            if fparser is None:
                outstr = textwrap.fill(
                    "File(s) {} use no additional options.".format(fnames))
                print(outstr, end="\n\n")
            else:
                fparser.usage = fnames
                fparser.print_help()
        parser.exit(0)


def add_results_option_group(parser):
    """Adds the options used to call gwin.io.results_from_cli function
    to an argument parser.
    
    These are options releated to loading the results from a run of
    gwin, for purposes of plotting and/or creating tables.

    Parameters
    ----------
    parser : object
        ArgumentParser instance.
    """

    results_reading_group = parser.add_argument_group(
        title="Arguments for loading results",
        description="Additional, file-specific arguments "
        "may also be provided, depending on what input-files are given. See "
        "--file-help for details.")

    # required options
    results_reading_group.add_argument(
        "--input-file", type=str, required=True, nargs="+",
        help="Path to input HDF file(s).")
    results_reading_group.add_argument(
        "--parameters", type=str, nargs="+", metavar="PARAM[:LABEL]",
        action=ParseParametersArg,
        help="Name of parameters to load. If none provided will load all of "
             "the model params in the input-file. If provided, the "
             "parameters can be any of the model params or posterior stats "
             "(loglikelihood, logprior, etc.) in the input file(s), derived "
             "parameters from them, or any function of them. If multiple "
             "files are provided, any parameter common to all files may be "
             "used. Syntax for functions is python; any math functions in "
             "the numpy libary may be used. Can optionally also specify a "
             "LABEL for each parameter. If no LABEL is provided, PARAM will "
             "used as the LABEL. If LABEL is the same as a parameter in "
             "pycbc.waveform.parameters, the label property of that parameter "
             "will be used (e.g., if LABEL were 'mchirp' then {} would be "
             "used). To see all possible parameters that may be used with the "
             "given input file(s), as well as all avaiable functions, run "
             "--file-help, along with one or more input files.".format(
             _waveform.parameters.mchirp.label))
    # optionals
    results_reading_group.add_argument(
        "--thin-start", type=int, default=None,
        help="Sample number to start collecting samples to plot. If none "
             "provided, will use the input file's `thin_start` attribute.")
    results_reading_group.add_argument(
        "--thin-interval", type=int, default=None,
        help="Interval to use for thinning samples. If none provided, will "
             "use the input file's `thin_interval` attribute.")
    results_reading_group.add_argument(
        "--thin-end", type=int, default=None,
        help="Sample number to stop collecting samples to plot. If none "
             "provided, will use the input file's `thin_end` attribute.")
    # advanced help
    results_reading_group.add_argument(
        "-H", "--file-help", action=PrintFileParams,
        help="Based on the provided input-file(s), print all available "
             "parameters that may be retrieved and all possible functions on "
             "those parameters. Also print available additional arguments "
             "that may be passed. This option is like an "
             "advanced --help: if run, the program will just print the "
             "information to screen, then exit. NOTE: this option must be "
             "provided after the --input-file option.")
    return results_reading_group


def results_from_cli(opts, extra_opts=None, load_samples=True):
    """Loads an inference result file along with any labels associated with it
    from the command line options.

    Parameters
    ----------
    opts : ArgumentParser options
        The options from the command line.
    load_samples : bool, optional
        Load the samples from the file.

    Returns
    -------
    fp_all : (list of) BaseInferenceFile type
        The result file as an hdf file. If more than one input file,
        then it returns a list.
    parameters_all : (list of) list
        List of the parameters to use, parsed from the parameters option.
        If more than one input file, then it returns a list.
    labels_all : dict
        Dictionary of labels to associate with the parameters.
    samples_all : (list of) FieldArray(s) or None
        If load_samples, the samples as a FieldArray; otherwise, None.
        If more than one input file, then it returns a list.
    """

    # lists for files and samples from all input files
    fp_all = []
    parameters_all = []
    labels_all = {}
    samples_all = []

    input_files = opts.input_file
    if isinstance(input_files, str):
        input_files = [input_files]

    # loop over all input files
    for input_file in input_files:
        logging.info("Reading input file %s", input_file)

        # read input file
        fp = loadfile(input_file, "r")

        # load the samples
        if load_samples:
            logging.info("Loading samples")

            # check if need extra parameters for a non-sampling parameter
            file_parameters, ts = transforms.get_common_cbc_transforms(
                opts.parameters, fp.variable_params)

            # read samples from file
            samples = fp.samples_from_cli(opts, extra_opts=extra_opts,
                                          parameters=file_parameters)

            logging.info("Using {} samples".format(samples.size))

            # add parameters not included in file
            samples = transforms.apply_transforms(samples, ts)

        # else do not read samples
        else:
            samples = None

        # add results to lists from all input files
        if len(input_files) > 1:
            fp_all.append(fp)
            parameters_all.append(opts.parameters)
            labels_all.update(opts.parameters_labels)
            samples_all.append(samples)

        # else only one input file then do not return lists
        else:
            fp_all = fp
            parameters_all = opts.parameters
            labels_all = opts.parameters_labels
            samples_all = samples

    return fp_all, parameters_all, labels_all, samples_all


def injections_from_cli(opts):
    """Gets injection parameters from the inference file(s).

    Parameters
    ----------
    opts : argparser
        Argparser object that has the command-line objects to parse.

    Returns
    -------
    FieldArray
        Array of the injection parameters from all of the input files given
        by ``opts.input_file``.
    """
    input_files = opts.input_file
    if isinstance(input_files, str):
        input_files = [input_files]
    injections = None
    # loop over all input files getting the injection files
    for input_file in input_files:
        fp = loadfile(input_file, 'r')
        these_injs = fp.read_injections()
        if injections is None:
            injections = these_injs
        else:
            injections = injections.append(these_injs)
    # check if need extra parameters than parameters stored in injection file
    _, ts = transforms.get_common_cbc_transforms(opts.parameters,
                                                 injections.fieldnames)
    # add parameters not included in injection file
    injections = transforms.apply_transforms(injections, ts)
    return injections
