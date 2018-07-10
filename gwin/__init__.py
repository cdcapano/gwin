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

"""GWIn is a package for parameter estimation of gravitational-wave data
using Bayesian Inference.
"""

from ._version import get_versions
import io
import models
import sampler
import burn_in

__author__ = 'Collin Capano <collin capano@ligo.org>'
__version__ = get_versions()['version']

del get_versions
