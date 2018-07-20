# Copyright (C) 2017  Collin Capano
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
This modules provides classes and functions for determining when Markov Chains
have burned in.
"""

import numpy
from scipy.stats import ks_2samp

from pycbc.filter import autocorrelation


def ks_test(samples1, samples2, threshold=0.9):
    """Applies a KS test to determine if two sets of samples are the same.

    The ks test is applied parameter-by-parameter. If the two-tailed p-value
    returned by the test is greater than ``threshold``, the samples are
    considered to be the same.

    Parameters
    ----------
    samples1 : dict
        Dictionary of mapping parameters to the first set of samples.
    samples2 : dict
        Dictionary of mapping parameters to the second set of samples.
    threshold : float
        The thershold to use for the p-value. Default is 0.9.

    Returns
    -------
    dict :
        Dictionary mapping parameter names to booleans indicating whether the
        given parameter passes the KS test.
    """
    is_the_same = {}
    assert set(samples1.keys()) == set(samples2.keys()), (
        "samples1 and 2 must have the same parameters")
    # iterate over the parameters
    for param in samples1:
        s1 = samples1[param]
        s2 = samples2[param]
        _, p_value = ks_2samp(samples_last_iter, samples_chain_midpt)
        is_the_same[param] = p_value > threshold
    return is_the_same


def n_acl(chain, nacls=5):
    """Burn in based on ACL.

    This applies the following test to determine burn in:

    1. The first half of the chain is ignored.

    2. An ACL is calculated from the second half.

    3. If ``nacls`` times the ACL is < the number of iterations / 2,
       the chain is considered to be burned in at the half-way point.

    Parameters
    ----------
    chain : array
        The chain of samples to apply the test to. Must be 1D.
    nacls : int, optional
        Number of ACLs to use for burn in. Default is 5.

    Returns
    -------
    burn_in_idx : int
        The burn in index. If the chain is not burned in, will be equal to the
        length of the chain.
    is_burned_in : bool
        Whether or not the chain is burned in.
    acl : int
        The ACL that was estimated.
    """
    kstart = int(len(chain)/2.)
    acl = autocorrelation.calculate_acl(chain[kstart:])
    is_burned_in = nacls * acl < kstart
    if is_burned_in:
        burn_in_idx = kstart
    else:
        burn_in_idx = len(chain)
    return burn_in_idx, is_burned_in, acl


def max_posterior(lnps_per_walker, dim):
    """Burn in based on samples being within dim/2 of maximum posterior.

    Parameters
    ----------
    lnps_per_walker : 2D array
        Array of values that are proportional to the log posterior values. Must
        have shape ``nwalkers x niterations``.
    dim : float
        The dimension of the parameter space.

    Returns
    -------
    burn_in_idx : array of int
        The burn in indices of each walker. If a walker is not burned in, its
        index will be be equal to the length of the chain.
    is_burned_in : array of bool
        Whether or not a walker is burned in.
    """
    if len(lnps_per_walker.shape) != 2:
        raise ValueError("lnps_per_walker must have shape "
                         "nwalkers x niterations")
    # find the value to compare against
    max_p = lnps_per_walker.max()
    criteria = max_p - dim/2.
    nwalkers, niterations = lnps_per_walker.shape
    burn_in_idx = numpy.empty(nwalkers, dtype=int)
    is_burned_in = numpy.empty(nwalkers, dtype=bool)
    # find the first iteration in each chain where the logpost has exceeded
    # max_p - dim/2
    for ii in range(nwalkers):
        chain = lnps_per_walker[ii,:]
        passedidx = numpy.where(chain >= criteria)[0]
        is_burned_in[ii] = is_burned_in = passedidx.size > 0
        if is_burned_in:
            burn_in_idx[ii] = passedidx[0]
        else:
            burn_in_idx[ii] = niterations
    return burn_in_idx, is_burned_in


def posterior_step(logposts, dim):
    """Finds the last time a chain made a jump > dim/2.

    Parameters
    ----------
    logposts : array
        1D array of values that are proportional to the log posterior values.
    dim : float
        The dimension of the parameter space.

    Returns
    -------
    int
        The index of the last time the logpost made a jump > dim/2. If that
        never happened, returns 0.
    """
    if logposts.ndim > 1:
        raise ValueError("logposts must be a 1D array")
    criteria = dim/2.
    dp = numpy.diff(logposts)
    indices = numpy.where(dp >= criteria)[0]
    if indices.size > 0:
        idx = indices[-1] + 1
    else:
        idx = 0
    return idx
