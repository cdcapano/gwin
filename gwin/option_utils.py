# Copyright (C) 2016 Collin Capano
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Generals
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""This module contains standard options used for inference-related programs.
"""

import logging
import shutil

from pycbc import (conversions, inject, transforms)
from pycbc.distributions import (bounded, constraints)
from pycbc.io.record import FieldArray
from pycbc.workflow import (ConfigParser, WorkflowConfigParser)
from pycbc.pool import choose_pool
from pycbc.psd import from_cli_multi_ifos as psd_from_cli_multi_ifos
from pycbc.strain import from_cli_multi_ifos as strain_from_cli_multi_ifos
from pycbc.strain import (gates_from_cli, psd_gates_from_cli,
                          apply_gates_to_td, apply_gates_to_fd)
from pycbc import waveform
from gwin import (burn_in, models, sampler)
from gwin.io.hdf import InferenceFile, check_integrity
from gwin.io.txt import InferenceTXTFile


# -----------------------------------------------------------------------------
#
#                   Utilities for loading config files
#
# -----------------------------------------------------------------------------

def add_config_opts_to_parser(parser):
    """Adds options for the configuration files to the given parser.
    """
    parser.add_argument("--config-files", type=str, nargs="+", required=True,
                        help="A file parsable by "
                             "pycbc.workflow.WorkflowConfigParser.")
    parser.add_argument("--config-overrides", type=str, nargs="+",
                        default=None, metavar="SECTION:OPTION:VALUE",
                        help="List of section:option:value combinations to "
                             "add into the configuration file.")


def config_parser_from_cli(opts):
    """Loads a config file from the given options, applying any overrides
    specified. Specifically, config files are loaded from the `--config-files`
    options while overrides are loaded from `--config-overrides`.
    """
    # read configuration file
    logging.info("Reading configuration file")
    if opts.config_overrides is not None:
        overrides = [override.split(":") for override in opts.config_overrides]
    else:
        overrides = None
    return WorkflowConfigParser(opts.config_files, overrides)


# -----------------------------------------------------------------------------
#
#                    Utilities for setting up a sampler
#
# -----------------------------------------------------------------------------

def add_sampler_option_group(parser):
    """Adds the options needed to set up an inference sampler.

    Parameters
    ----------
    parser : object
        ArgumentParser instance.
    """
    sampler_group = parser.add_argument_group(
        "Arguments for setting up a sampler")

    # required options
    sampler_group.add_argument(
        "--sampler", required=True, choices=sampler.samplers.keys(),
        help="Sampler class to use for finding posterior.")
    sampler_group.add_argument(
        "--niterations", type=int,
        help="Number of iterations to perform. If 'use_sampler' is given to "
             "burn-in-function, this will be counted after the sampler's burn "
             "function has run. Otherwise, this is the total number of "
             "iterations, including any burn in.")
    sampler_group.add_argument(
        "--n-independent-samples", type=int,
        help="Run the sampler until the specified number of "
             "independent samples is obtained, at minimum. Requires "
             "checkpoint-interval. At each checkpoint the burn-in iteration "
             "and ACL is updated. The number of independent samples is the "
             "number of samples across all walkers starting at the "
             "burn-in-iteration and skipping every `ACL`th iteration. "
             "Either this or niteration should be specified (but not both).")
    # sampler-specific options
    sampler_group.add_argument(
        "--nwalkers", type=int, default=None,
        help="Number of walkers to use in sampler. Required for MCMC "
             "samplers.")
    sampler_group.add_argument(
        "--ntemps", type=int, default=None,
        help="Number of temperatures to use in sampler. Required for parallel "
             "tempered MCMC samplers.")
    sampler_group.add_argument(
        "--burn-in-function", default=None, nargs='+',
        choices=burn_in.burn_in_functions.keys(),
        help="Use the given function to determine when chains are burned in. "
             "If none provided, no burn in will be estimated. "
             "If multiple functions are provided, will use the maximum "
             "iteration from all functions.")
    sampler_group.add_argument(
        "--min-burn-in", type=int, default=0,
        help="Force the burn-in to be at least the given number of "
             "iterations.")
    sampler_group.add_argument(
        "--update-interval", type=int, default=None,
        help="If using kombine, specify the number of steps to take between "
             "proposal updates. Note: for purposes of updating, kombine "
             "counts iterations since the last checkpoint. This interval "
             "should therefore be less than the checkpoint interval, else "
             "no updates will occur. To ensure that updates happen at equal "
             "intervals, make checkpoint-interval a multiple of "
             "update-interval.")
    sampler_group.add_argument(
        "--nprocesses", type=int, default=None,
        help="Number of processes to use. If not given then use maximum.")
    sampler_group.add_argument(
        "--use-mpi", action='store_true', default=False,
        help="Use MPI to parallelize the sampler")
    sampler_group.add_argument(
        "--logpost-function", default="logposterior",
        help="Which attribute of the model to use for the logposterior. "
             "The default is logposterior. For example, if using the "
             "gaussian_noise model, you may wish to set this to logplr, since "
             "the logposterior includes a large constant contribution from "
             "log noise likelihood.")

    return sampler_group


def sampler_from_cli(opts, model, pool=None):
    """Parses the given command-line options to set up a sampler.

    Parameters
    ----------
    opts : object
        ArgumentParser options.
    model : model
        The model to use with the sampler.

    Returns
    -------
    gwin.sampler
        A sampler initialized based on the given arguments.
    """
    # create a wrapper for the model
    model = models.CallModel(model, opts.logpost_function)

    # Used to help paralleize over multiple cores / MPI
    if opts.nprocesses > 1:
        models._global_instance = model
        model_call = models._call_global_model
    else:
        model_call = None

    sclass = sampler.samplers[opts.sampler]

    pool = choose_pool(mpi=opts.use_mpi, processes=opts.nprocesses)

    if pool is not None:
        pool.count = opts.nprocesses

    return sclass.from_cli(opts, model,
                           pool=pool, model_call=model_call)


# -----------------------------------------------------------------------------
#
#                       Utilities for loading data
#
# -----------------------------------------------------------------------------


def add_low_frequency_cutoff_opt(parser):
    """Adds the low-frequency-cutoff option to the given parser."""
    # FIXME: this just uses the same frequency cutoff for every instrument for
    # now. We should allow for different frequency cutoffs to be used; that
    # will require (minor) changes to the Likelihood class
    parser.add_argument("--low-frequency-cutoff", type=float,
                        help="Low frequency cutoff for each IFO.")


def low_frequency_cutoff_from_cli(opts):
    """Parses the low frequency cutoff from the given options.

    Returns
    -------
    dict
        Dictionary of instruments -> low frequency cutoff.
    """
    # FIXME: this just uses the same frequency cutoff for every instrument for
    # now. We should allow for different frequency cutoffs to be used; that
    # will require (minor) changes to the Likelihood class
    instruments = opts.instruments if opts.instruments is not None else []
    return {ifo: opts.low_frequency_cutoff for ifo in instruments}


def data_from_cli(opts):
    """Loads the data needed for a model from the given
    command-line options. Gates specifed on the command line are also applied.

    Parameters
    ----------
    opts : ArgumentParser parsed args
        Argument options parsed from a command line string (the sort of thing
        returned by `parser.parse_args`).

    Returns
    -------
    strain_dict : dict
        Dictionary of instruments -> `TimeSeries` strain.
    stilde_dict : dict
        Dictionary of instruments -> `FrequencySeries` strain.
    psd_dict : dict
        Dictionary of instruments -> `FrequencySeries` psds.
    """
    # get gates to apply
    gates = gates_from_cli(opts)
    psd_gates = psd_gates_from_cli(opts)

    # get strain time series
    instruments = opts.instruments if opts.instruments is not None else []
    strain_dict = strain_from_cli_multi_ifos(opts, instruments,
                                             precision="double")
    # apply gates if not waiting to overwhiten
    if not opts.gate_overwhitened:
        strain_dict = apply_gates_to_td(strain_dict, gates)

    # get strain time series to use for PSD estimation
    # if user has not given the PSD time options then use same data as analysis
    if opts.psd_start_time and opts.psd_end_time:
        logging.info("Will generate a different time series for PSD "
                     "estimation")
        psd_opts = opts
        psd_opts.gps_start_time = psd_opts.psd_start_time
        psd_opts.gps_end_time = psd_opts.psd_end_time
        psd_strain_dict = strain_from_cli_multi_ifos(psd_opts,
                                                     instruments,
                                                     precision="double")
        # apply any gates
        logging.info("Applying gates to PSD data")
        psd_strain_dict = apply_gates_to_td(psd_strain_dict, psd_gates)

    elif opts.psd_start_time or opts.psd_end_time:
        raise ValueError("Must give --psd-start-time and --psd-end-time")
    else:
        psd_strain_dict = strain_dict

    # FFT strain and save each of the length of the FFT, delta_f, and
    # low frequency cutoff to a dict
    stilde_dict = {}
    length_dict = {}
    delta_f_dict = {}
    low_frequency_cutoff_dict = low_frequency_cutoff_from_cli(opts)
    for ifo in instruments:
        stilde_dict[ifo] = strain_dict[ifo].to_frequencyseries()
        length_dict[ifo] = len(stilde_dict[ifo])
        delta_f_dict[ifo] = stilde_dict[ifo].delta_f

    # get PSD as frequency series
    psd_dict = psd_from_cli_multi_ifos(
        opts, length_dict, delta_f_dict, low_frequency_cutoff_dict,
        instruments, strain_dict=psd_strain_dict, precision="double")

    # apply any gates to overwhitened data, if desired
    if opts.gate_overwhitened and opts.gate is not None:
        logging.info("Applying gates to overwhitened data")
        # overwhiten the data
        for ifo in gates:
            stilde_dict[ifo] /= psd_dict[ifo]
        stilde_dict = apply_gates_to_fd(stilde_dict, gates)
        # unwhiten the data for the model
        for ifo in gates:
            stilde_dict[ifo] *= psd_dict[ifo]

    return strain_dict, stilde_dict, psd_dict


# -----------------------------------------------------------------------------
#
#                Utilities for plotting results
#
# -----------------------------------------------------------------------------

def add_plot_posterior_option_group(parser):
    """Adds the options needed to configure plots of posterior results.

    Parameters
    ----------
    parser : object
        ArgumentParser instance.
    """
    pgroup = parser.add_argument_group("Options for what plots to create and "
                                       "their formats.")
    pgroup.add_argument('--plot-marginal', action='store_true', default=False,
                        help="Plot 1D marginalized distributions on the "
                             "diagonal axes.")
    pgroup.add_argument('--marginal-percentiles', nargs='+', default=None,
                        type=float,
                        help="Percentiles to draw lines at on the 1D "
                             "histograms.")
    pgroup.add_argument("--plot-scatter", action='store_true', default=False,
                        help="Plot each sample point as a scatter plot.")
    pgroup.add_argument("--plot-density", action="store_true", default=False,
                        help="Plot the posterior density as a color map.")
    pgroup.add_argument("--plot-contours", action="store_true", default=False,
                        help="Draw contours showing the 50th and 90th "
                             "percentile confidence regions.")
    pgroup.add_argument('--contour-percentiles', nargs='+', default=None,
                        type=float,
                        help="Percentiles to draw contours if different "
                             "than 50th and 90th.")
    # add mins, maxs options
    pgroup.add_argument('--mins', nargs='+', metavar='PARAM:VAL', default=[],
                        help="Specify minimum parameter values to plot. This "
                             "should be done by specifying the parameter name "
                             "followed by the value. Parameter names must be "
                             "the same as the PARAM argument in --parameters "
                             "(or, if no parameters are provided, the same as "
                             "the parameter name specified in the variable "
                             "args in the input file. If none provided, "
                             "the smallest parameter value in the posterior "
                             "will be used.")
    pgroup.add_argument('--maxs', nargs='+', metavar='PARAM:VAL', default=[],
                        help="Same as mins, but for the maximum values to "
                             "plot.")
    # add expected parameters options
    pgroup.add_argument('--expected-parameters', nargs='+',
                        metavar='PARAM:VAL',
                        default=[],
                        help="Specify expected parameter values to plot. If "
                             "provided, a cross will be plotted in each axis "
                             "that an expected parameter is provided. "
                             "Parameter names must be "
                             "the same as the PARAM argument in --parameters "
                             "(or, if no parameters are provided, the same as "
                             "the parameter name specified in the variable "
                             "args in the input file.")
    pgroup.add_argument('--expected-parameters-color', default='r',
                        help="What to color the expected-parameters cross. "
                             "Default is red.")
    pgroup.add_argument('--plot-injection-parameters', action='store_true',
                        default=False,
                        help="Get the expected parameters from the injection "
                             "in the input file. There must be only a single "
                             "injection in the file to work. Any values "
                             "specified by expected-parameters will override "
                             "the values obtained for the injection.")
    return pgroup


def plot_ranges_from_cli(opts):
    """Parses the mins and maxs arguments from the `plot_posterior` option
    group.

    Parameters
    ----------
    opts : ArgumentParser
        The parsed arguments from the command line.

    Returns
    -------
    mins : dict
        Dictionary of parameter name -> specified mins. Only parameters that
        were specified in the --mins option will be included; if no parameters
        were provided, will return an empty dictionary.
    maxs : dict
        Dictionary of parameter name -> specified maxs. Only parameters that
        were specified in the --mins option will be included; if no parameters
        were provided, will return an empty dictionary.
    """
    mins = {}
    for x in opts.mins:
        x = x.split(':')
        if len(x) != 2:
            raise ValueError("option --mins not specified correctly; see help")
        mins[x[0]] = float(x[1])
    maxs = {}
    for x in opts.maxs:
        x = x.split(':')
        if len(x) != 2:
            raise ValueError("option --maxs not specified correctly; see help")
        maxs[x[0]] = float(x[1])
    return mins, maxs


def expected_parameters_from_cli(opts):
    """Parses the --expected-parameters arguments from the `plot_posterior`
    option group.

    Parameters
    ----------
    opts : ArgumentParser
        The parsed arguments from the command line.

    Returns
    -------
    dict
        Dictionary of parameter name -> expected value. Only parameters that
        were specified in the --expected-parameters option will be included; if
        no parameters were provided, will return an empty dictionary.
    """
    expected = {}
    for x in opts.expected_parameters:
        x = x.split(':')
        if len(x) != 2:
            raise ValueError("option --expected-paramters not specified "
                             "correctly; see help")
        expected[x[0]] = float(x[1])
    return expected


def add_scatter_option_group(parser):
    """Adds the options needed to configure scatter plots.

    Parameters
    ----------
    parser : object
        ArgumentParser instance.
    """
    scatter_group = parser.add_argument_group("Options for configuring the "
                                              "scatter plot.")

    scatter_group.add_argument(
        '--z-arg', type=str, default=None,
        help='What to color the scatter points by. Syntax is the same as the '
             'parameters option.')
    scatter_group.add_argument(
        "--vmin", type=float, help="Minimum value for the colorbar.")
    scatter_group.add_argument(
        "--vmax", type=float, help="Maximum value for the colorbar.")
    scatter_group.add_argument(
        "--scatter-cmap", type=str, default='plasma',
        help="Specify the colormap to use for points. Default is plasma.")

    return scatter_group


def add_density_option_group(parser):
    """Adds the options needed to configure contours and density colour map.

    Parameters
    ----------
    parser : object
        ArgumentParser instance.
    """
    density_group = parser.add_argument_group("Options for configuring the "
                                              "contours and density color map")

    density_group.add_argument(
        "--density-cmap", type=str, default='viridis',
        help="Specify the colormap to use for the density. "
             "Default is viridis.")
    density_group.add_argument(
        "--contour-color", type=str, default=None,
        help="Specify the color to use for the contour lines. Default is "
             "white for density plots and black for scatter plots.")
    density_group.add_argument(
        '--use-kombine-kde', default=False, action="store_true",
        help="Use kombine's KDE for determining contours. "
             "Default is to use scipy's gaussian_kde.")

    return density_group
