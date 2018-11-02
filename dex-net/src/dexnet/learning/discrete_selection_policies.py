# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Policies for selecting the next point in discrete solvers

Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import logging
import numpy as np
import scipy.io
import scipy.stats as ss

from dexnet.learning import DiscreteModel, BetaBernoulliModel, GaussianModel
import IPython

# class DiscreteSelectionPolicy(metaclass=ABCMeta):
class DiscreteSelectionPolicy:

    __metaclass__ = ABCMeta
    def __init__(self, model = None):
        self.model_ = model

    @abstractmethod
    def choose_next(self):
        """
        Choose the next index of the model to sample 
        """
        pass

    def set_model(self, model):
        if not isinstance(model, DiscreteModel):
            raise ValueError('Must supply a discrete predictive model')
        self.model_ = model

class UniformSelectionPolicy(DiscreteSelectionPolicy):
    def choose_next(self):
        """ Returns an index uniformly at random"""
        if self.model_ is None:
            raise ValueError('Must set predictive model')
        num_vars = self.model_.num_vars()
        next_index = np.random.choice(num_vars)
        return next_index

class MaxDiscreteSelectionPolicy(DiscreteSelectionPolicy):
    def choose_next(self):
        """ Returns the index of the maximal variable, breaking ties uniformly at random"""
        if self.model_ is None:
            raise ValueError('Must set predictive model')
        max_indices, _, _ = self.model_.max_prediction()
        num_max_indices = max_indices.shape[0]
        next_index = np.random.choice(num_max_indices)
        return max_indices[next_index]

class ThompsonSelectionPolicy(DiscreteSelectionPolicy):
    """ Chooses the next point using the Thompson sampling selection policy"""
    def choose_next(self, stop = False):
        """ Returns the index of the maximal random sample, breaking ties uniformly at random"""
        if self.model_ is None:
            raise ValueError('Must set predictive model')
        sampled_values = self.model_.sample()
        if stop:
            IPython.embed()
        max_indices = np.where(sampled_values == np.max(sampled_values))[0]
        num_max_indices = max_indices.shape[0]
        next_index = np.random.choice(num_max_indices)
        return max_indices[next_index]        

class BetaBernoulliGittinsIndex98Policy(DiscreteSelectionPolicy):
    """ Chooses the next point using the BetaBernoulli gittins index policy with gamma = 0.98"""
    def __init__(self, model = None):
        self.indices_ = scipy.io.loadmat('data/bandits/gittins_indices_98.mat')
        self.indices_ = self.indices_['indices']
        DiscreteSelectionPolicy.__init__(self, model)

    def choose_next(self):
        """ Returns the index of the maximal random sample, breaking ties uniformly at random"""
        if self.model_ is None:
            raise ValueError('Must set predictive model')
        if not isinstance(self.model_, BetaBernoulliModel):
            raise ValueError('Gittins index policy can only be used with Beta-bernoulli models')
        
        alphas = self.model_.posterior_alphas.astype(np.uint64)
        betas = self.model_.posterior_betas.astype(np.uint64)

        # subtract one, since the indices are intended for matlab 1 indexing
        alphas = alphas - 1
        betas = betas - 1

        # snap alphas and betas to boundaries of index matrix
        alphas[alphas >= self.indices_.shape[0]] = self.indices_.shape[0] - 1
        betas[betas >= self.indices_.shape[1]] = self.indices_.shape[1] - 1
        np.round(alphas)
        np.round(betas)

        # find maximum of gittins indices
        gittins_indices = self.indices_[alphas, betas]

        max_indices = np.where(gittins_indices == np.max(gittins_indices))[0]
        num_max_indices = max_indices.shape[0]
        next_index = np.random.choice(num_max_indices)
        return max_indices[next_index]        

class BetaBernoulliBayesUCBPolicy(DiscreteSelectionPolicy):
    """ Chooses the next point using the Bayes UCB selection policy"""
    def __init__(self, horizon=1000, c=6, model=None):
        self.t_ = 1
        self.n_ = horizon
        self.c_ = c
        DiscreteSelectionPolicy.__init__(self, model)        
    
    def choose_next(self, stop = False):
        """ Returns the index of the maximal random sample, breaking ties uniformly at random"""
        if self.model_ is None:
            raise ValueError('Must set predictive model')
        gamma = 1.0 - (1.0 / (self.t_ * np.log(self.n_)**self.c_))
        alphas = self.model_.posterior_alphas
        betas = self.model_.posterior_betas
        intervals = ss.beta.interval(gamma, alphas, betas)
        ucbs = intervals[1]

        max_indices = np.where(ucbs == np.max(ucbs))[0]
        num_max_indices = max_indices.shape[0]
        next_index = np.random.choice(num_max_indices)
        self.t_ += 1
        return max_indices[next_index]        

class GaussianUCBPolicy(DiscreteSelectionPolicy):
    def __init__(self, beta=1.0):
        self.beta_ = beta

    def choose_next(self, stop=False):
        """Returns the index of the variable with the highest UCB, breaking ties
        uniformly at random."""
        if self.model_ is None:
            raise ValueError('Must set predictive model')
        if not isinstance(self.model_, GaussianModel):
            raise ValueError('GP-UCB can only be used with Gaussian models')

        ucb = self.model_.means + self.beta_ * np.sqrt(self.model_.variances)
        max_ucb = np.max(ucb)
        max_indices = np.where(ucb == max_ucb)[0]
        num_max_indices = max_indices.shape[0]
        next_index = np.random.choice(num_max_indices)
        return max_indices[next_index]
