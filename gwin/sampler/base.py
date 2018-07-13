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
"""
Defines the base sampler class to be inherited by all samplers.
"""

from abc import ABCMeta, abstractmethod, abstractproperty
import numpy
from pycbc.io import FieldArray
from pycbc.filter import autocorrelation
import h5py
import logging


#
# =============================================================================
#
#                                   Samplers
#
# =============================================================================
#

class BaseSampler(object):
    """Base container class for inference samplers.

    Parameters
    ----------
    model : Model
        An instance of a model from ``gwin.models``.
    """
    __metaclass__ = ABCMeta
    name = None

    def __init__(self, model):
        self.model = model

    #@classmethod # uncomment when we move to python 3.3
    @abstractmethod
    def from_config(cls, cp, model, pool=None, model_call=None, **kwargs):
        """This should initialize the sampler given a config file.
        """
        pass

    @property
    def variable_params(self):
        """Returns the parameters varied in the model.
        """
        return self.model.variable_params

    @property
    def sampling_params(self):
        """Returns the sampling params used by the model.
        """
        return self.model.sampling_params

    @property
    def static_params(self):
        """Returns the model's fixed parameters.
        """
        return self.model.static_params

    @abstractproperty
    def samples(self):
        """Should return all of the samples currently stored in memory as a
        numpy structure array or FieldArray.
        """
        pass

    @abstractproperty
    def model_stats(self):
        """Should return all of the model's metadata currently stored in
        memory as a numpy structure array or FieldArray.
        """
        pass

    @abstractmethod
    def run(self):
        """This function should run the sampler.
        
        Any checkpointing should be done internally in this function.
        """
        pass

    @abstractproperty
    def io(self):
        """A class that inherits from ``BaseInferenceFile`` to handle IO with
        an hdf file.
        
        This should be a class, not an instance of class, so that the sampler
        can initialize it when needed.
        """
        pass
