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
#                           Base Sampler definition
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
    def from_config(cls, cp, model, nprocesses=1, use_mpi=False,
                    **kwargs):
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
        """A dict mapping variable_params to arrays of samples currently
        in memory. The dictionary may also contain sampling_params.
        
        The sample arrays may have any shape, and may or may not be thinned.
        """
        pass

    @abstractproperty
    def model_stats(self):
        """A dict mapping model's metadata fields to arrays of values for
        each sample in ``raw_samples``.

        The arrays may have any shape, and may or may not be thinned.
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

    @abstractmethod
    def set_initial_conditions(self, initial_distribution=None,
                               samples_file=None):
        """Sets up the starting point for the sampler.
        
        Should also set the sampler's random state.
        """
        pass

    @abstractmethod
    def checkpoint(self):
        """The sampler must have a checkpoint method for dumping raw samples
        and stats to the file type defined by ``io``.
        """
        pass

    @abstractmethod
    def finalize(self):
        """Do any finalization to the samples file before exiting."""
        pass

    def write_metadata(self, fp):
        """Writes metadata about the sampler to the given filehandler."""
        fp.attrs['sampler'] = self.name
        # write the model's metadata
        self.model.write_metadata(fp)
        self._write_more_metadata(fp)
        
    def _write_more_metadata(self, fp):
        """Optional method that can be implemented if a sampler wants to write
        more metadata than just its name and the model's metadata.
        """
        pass

    def setup_output(self, output_file, force=False, injection_file=None):
        """Sets up the sampler's checkpoint and output files.

        The checkpoint file has the same name as the output file, but with
        ``.checkpoint`` appended to the name. A backup file will also be
        created.

        If the output file already exists, an ``OSError`` will be raised.
        This can be overridden by setting ``force`` to ``True``.
        
        Parameters
        ----------
        sampler : sampler instance
            Sampler
        output_file : str
            Name of the output file.
        force : bool, optional
            If the output file already exists, overwrite it.
        injection_file : str, optional
            If an injection was added to the data, write its information.
        """
        # check for backup file(s)
        checkpoint_file = output_file + '.checkpoint'
        backup_file = output_file + '.bkup'
        # check if we have a good checkpoint and/or backup file
        checkpoint_valid = validate_checkpoint_files(checkpoint_file,
                                                     backup_file)
        # Create a new file if the checkpoint doesn't exist, or if it is
        # corrupted
        if not checkpoint_valid:
            self.create_new_output_file(checkpoint_file, force=force,
                                        injection_file=injection_file)
            # now the checkpoint is valid
            checkpoint_valid = True
            # copy to backup
            shutil.copy(checkpoint_file, backup_file)
        # write the command line
        for fn in [checkpoint_file, backup_file]:
            with sampler.io(fn, "a") as fp:
                fp.write_command_line()
        # store
        self.checkpoint_file = checkpoint_file
        self.backup_file = backup_file
        self.checkpoint_valid = checkpoint_valid

    def set_target(self, nsamples, require_independent=False):
        """Sets the number of samples the sampler should try to acquire.

        If the ``must_be_independent`` flag is set, then the number of samples
        must be independent. This means, for example, that MCMC chains are
        thinned by their ACL before counting samples. Otherwise, the sampler
        will just run until it has the requested number of samples, regardless
        of thinning.

        Parameters
        ----------
        nsamples : int
            The number of samples to acquire.
        must_be_independent : bool, optional
            Add the requirement that the target number of samples be
            independent. Default is False.
        """
        self.target_nsamples = nsamples
        self.require_indep_samples = require_independent


#
# =============================================================================
#
#                           Convenience functions
#
# =============================================================================
#

def create_new_output_file(sampler, filename, force=False, injection_file=None,
                           **kwargs):
    """Creates a new output file.

    If the output file already exists, an ``OSError`` will be raised. This can
    be overridden by setting ``force`` to ``True``.
    
    Parameters
    ----------
    sampler : sampler instance
        Sampler
    filename : str
        Name of the file to create.
    force : bool, optional
        Create the file even if it already exists. Default is False.
    injection_file : str, optional
        If an injection was added to the data, write its information.
    \**kwargs :
        All other keyword arguments are passed through to the file's
        ``write_metadata`` function.
    """
    if os.path.exists(filename):
        if force:
            os.remove(filename)
        else:
            raise OSError("output-file already exists; use force if you "
                          "wish to overwrite it.")
    logging.info("Creating file {}".format(filename))
    with sampler.io(filename, "w") as fp:
        # save the sampler's metadata
        sampler.write_metadata(fp)
        # save injection parameters
        if injection_file is not None:
            logging.info("Writing injection file to output")
            # just use the first one
            fp.write_injections(injection_file)

def intial_dist_from_config(cp):
    """Loads a distribution for the sampler start from the given config file.

    A distribution will only be loaded if the config file has a [initial-*]
    section(s).

    Parameters
    ----------
    cp : Config parser
        The config parser to try to load from.

    Returns
    -------
    JointDistribution or None :
        The initial distribution. If no [initial-*] section found in the
        config file, will just return None.
    """
    if len(cp.get_subsections("initial")):
        logging.info("Using a different distribution for the starting points "
                     "than the prior.")
        initial_dists = distributions.read_distributions_from_config(
            cp, section="initial")
        constraints = distributions.read_constraints_from_config(cp,
            constraint_section="initial_constraint")
        init_dist = distributions.JointDistribution(sampler.variable_params,
            *initial_dists, **{"constraints" : constraints})
    else:
        init_dist = None
    return init_dist
