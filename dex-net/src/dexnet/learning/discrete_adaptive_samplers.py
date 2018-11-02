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
# -*- coding: utf-8 -*-

"""
Classes for selecting a candidate that maximizes some objective over a discrete set of candidates

Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.stats
import time

from dexnet.constants import DEF_MAX_ITER

from dexnet.learning import DiscreteSelectionPolicy, UniformSelectionPolicy, ThompsonSelectionPolicy, BetaBernoulliGittinsIndex98Policy, BetaBernoulliBayesUCBPolicy, GaussianUCBPolicy
from dexnet.learning import DiscreteModel, BetaBernoulliModel, GaussianModel, CorrelatedBetaBernoulliModel
from dexnet.learning import DiscreteSamplingSolver
from dexnet.learning import MaxIterTerminationCondition

import IPython

class AdaptiveSamplingResult:
    """
    Struct to store the results of sampling / optimization.

    Attributes
    ----------
    best_candidates : list of candidate objects
        list of the best candidates as estimated by the optimizer
    best_pred_means : list of floats
        list of the predicted mean objective value for the best candidates
    best_pred_vars : list of floats
        list of the variance in the predicted objective value for the best candidates
    total_time : float
        the total optimization time
    checkpt_times : list of floats
        the time since start at which the snapshots were taken
    iters : list of ints
        the iterations at which snapshots were taked
    indices : list of ints
        the indices of the candidates selected at each snapshot iteration
    vals : list of objective output values
        the value returned by the evaluated candidate at each snapshot iteration
    models : list of :obj:`Model`
        the state of the current candidate objective value predictive model at each snapshot iteration
    best_pred_ind : list of int
        the indices of the candidate predicted to be the best by the model at each snapshot iteration
    """
    def __init__(self, best_candidates, best_pred_means, best_pred_vars, total_time, checkpt_times, iters, indices, vals, models):
        self.best_candidates = best_candidates
        self.best_pred_means = best_pred_means
        self.best_pred_vars = best_pred_vars
        self.total_time = total_time
        self.checkpt_times = checkpt_times
        self.iters = iters
        self.indices = indices
        self.vals = vals
        self.models = models
        self.best_pred_ind = [m.best_pred_ind for m in models]

    def shrink(self):
        self.models = self.models[-1:]

# class DiscreteAdaptiveSampler(DiscreteSamplingSolver, metaclass=ABCMeta):
class DiscreteAdaptiveSampler(DiscreteSamplingSolver):
    """
    Abstract class for an adaptive sampler to maximize some objective over some discrete set of candidates.
    NOTE: Should NOT be instantiated directly. Always use a subclass that fixes the model and selection policy

    Attributes
    ----------
    objective : :obj:`Objective`
        the objective to optimize via sampling
    candidates : :obj:`list` of arbitrary objects that can be evaluted by the objective
        the list of candidates to optimize over
    model : :obj:`Model`
        a model of the objective values for each candidate based on the samples so far
    selection_policy : :obj:`DiscreteSelectionPolicy`
        a policy to use to select the next candidate to evaluate
    """
    __metaclass__ = ABCMeta

    def __init__(self, objective, candidates, model, selection_policy):
        self.model_ = model
        self.selection_policy_ = selection_policy
        DiscreteSamplingSolver.__init__(self, objective, candidates)

    @abstractmethod
    def reset_model(self, candidates):
        """ Reset model with the new candidates.

        Parameters
        ----------
        candidates : list of arbitrary objects
            a new set of candidates, for resetting the model
        """
        # feels a little hacky, but maybe we can make it work down the road
        pass

    def discrete_maximize(self, candidates, termination_condition = MaxIterTerminationCondition(DEF_MAX_ITER),
                          snapshot_rate = 1):
        """
        Maximizes a function over a discrete set of variables by
        iteratively predicting the best point (using some model and policy).

        Parameters
        ---------
        candidates : list of arbitrary objects that can be evaluted by the objective
            the list of candidates to optimize over
        termination_condition : :obj:`TerminationCondition`
            called on each iteration to determine whether or not to terminate
        snapshot_rate : int
            how often to store the state of the optimizer

        Returns
        ------
        result : :obj:`AdaptiveSamplingResult`
            the result of the optimization
        """
        # check input
        if len(candidates) == 0:
            raise ValueError('No candidates specified')

        if not isinstance(self.model_, DiscreteModel):
            logging.error('Illegal model specified')
            raise ValueError('Illegitimate model used in DiscreteAdaptiveSampler')

        # init vars
        terminate = False
        k = 0 # cur iter
        num_candidates = len(candidates)
        self.reset_model(candidates) # update model with new candidates

        # logging
        times = []
        iters = []
        iter_indices = []
        iter_vals = []
        iter_models = []
        start_time = time.clock()
        next_ind_val = 0

        while not terminate:
            # get next point to sample
            next_ind = self.selection_policy_.choose_next()

            # evaluate the function at the given point (can be nondeterministic)
            prev_ind_val = next_ind_val
            next_ind_val = self.objective_.evaluate(candidates[next_ind])

            # snapshot the model and whatnot
            if (k % snapshot_rate) == 0:
                logging.debug('Iteration %d' %(k))

                # log time and stuff
                checkpt = time.clock()
                times.append(checkpt - start_time)
                iters.append(k)
                iter_indices.append(next_ind)
                iter_vals.append(next_ind_val)
                iter_models.append(self.model_.snapshot())

            # update the model (e.g. posterior update, grasp pruning)
            self.model_.update(next_ind, next_ind_val)

            # check termination condiation
            k = k + 1
            terminate = termination_condition(k, cur_val = next_ind_val, prev_val = prev_ind_val, model = self.model_)

        # log final values
        checkpt = time.clock()
        times.append(checkpt - start_time)
        iters.append(k)
        iter_indices.append(next_ind)
        iter_vals.append(next_ind_val)
        iter_models.append(self.model_.snapshot())

        # log total runtime
        end_time = time.clock()
        total_duration = end_time - start_time

        # log results and return
        best_indices, best_pred_means, best_pred_vars = self.model_.max_prediction()
        best_candidates = []
        num_best = best_indices.shape[0]
        for i in range(num_best):
            best_candidates.append(candidates[best_indices[i]])
        return AdaptiveSamplingResult(best_candidates, best_pred_means, best_pred_vars, total_duration,
                                      times, iters, iter_indices, iter_vals, iter_models)


# Beta-Bernoulli bandit models: so easy!
class BetaBernoulliBandit(DiscreteAdaptiveSampler):
    """
    Class for running Beta Bernoulli Multi-Armed Bandits

    Attributes
    ----------
    objective : :obj:`Objective`
        the objective to optimize via sampling
    candidates : :obj:`list` of arbitrary objects that can be evaluted by the objective
        the list of candidates to optimize over
    policy : :obj:`DiscreteSelectionPolicy`
        a policy to use to select the next candidate to evaluate (e.g. ThompsonSampling)
    alpha_prior : float
        the prior to use on the alpha parameter 
    beta_prior : float
        the prior to use on the beta parameter
    """
    def __init__(self, objective, candidates, policy, alpha_prior = 1.0, beta_prior = 1.0):
        self.num_candidates_ = len(candidates)
        self.model_ = BetaBernoulliModel(self.num_candidates_, alpha_prior, beta_prior)
        self.selection_policy_ = policy
        self.selection_policy_.set_model(self.model_)

        DiscreteAdaptiveSampler.__init__(self, objective, candidates, self.model_, self.selection_policy_)

    def reset_model(self, candidates):
        """ Needed to independently maximize over subsets of data """
        num_subcandidates = len(candidates)
        self.model_ = BetaBernoulliModel(self.num_candidates_, self.model_.alpha_prior_, self.model_.beta_prior_)
        self.selection_policy_.set_model(self.model_) # always update the selection policy!

class UniformAllocationMean(BetaBernoulliBandit):
    """
    Uniform Allocation with Beta Bernoulli Multi-Armed Bandits

    Attributes
    ----------
    objective : :obj:`Objective`
        the objective to optimize via sampling
    candidates : :obj:`list` of arbitrary objects that can be evaluted by the objective
        the list of candidates to optimize over
    alpha_prior : float
        the prior to use on the alpha parameter 
    beta_prior : float
        the prior to use on the beta parameter
    """
    def __init__(self, objective, candidates, alpha_prior = 1.0, beta_prior = 1.0):
        self.selection_policy_ = UniformSelectionPolicy()
        BetaBernoulliBandit.__init__(self, objective, candidates, self.selection_policy_, alpha_prior, beta_prior)

class ThompsonSampling(BetaBernoulliBandit):
    """
    Thompson Sampling with Beta Bernoulli Multi-Armed Bandits

    Attributes
    ----------
    objective : :obj:`Objective`
        the objective to optimize via sampling
    candidates : :obj:`list` of arbitrary objects that can be evaluted by the objective
        the list of candidates to optimize over
    alpha_prior : float
        the prior to use on the alpha parameter 
    beta_prior : float
        the prior to use on the beta parameter
    """
    def __init__(self, objective, candidates, alpha_prior = 1.0, beta_prior = 1.0):
        self.selection_policy_ = ThompsonSelectionPolicy()
        BetaBernoulliBandit.__init__(self, objective, candidates, self.selection_policy_, alpha_prior, beta_prior)

class GittinsIndex98(BetaBernoulliBandit):
    """
    Gittins Index Policy using gamma = 0.98 with Beta Bernoulli Multi-Armed Bandits

    Attributes
    ----------
    objective : :obj:`Objective`
        the objective to optimize via sampling
    candidates : :obj:`list` of arbitrary objects that can be evaluted by the objective
        the list of candidates to optimize over
    alpha_prior : float
        the prior to use on the alpha parameter 
    beta_prior : float
        the prior to use on the beta parameter
    """
    def __init__(self, objective, candidates, alpha_prior = 1.0, beta_prior = 1.0):
        # NOTE: priors will be rounded to the nearest integers
        self.selection_policy_ = BetaBernoulliGittinsIndex98Policy()
        BetaBernoulliBandit.__init__(self, objective, candidates, self.selection_policy_, alpha_prior, beta_prior)

# Gaussian bandit models
class GaussianBandit(DiscreteAdaptiveSampler):
    """
    Multi-Armed Bandit class using and independent Gaussian random variables to model the objective value of each candidate.

    Attributes
    ----------
    objective : :obj:`Objective`
        the objective to optimize via sampling
    candidates : :obj:`list` of arbitrary objects that can be evaluted by the objective
        the list of candidates to optimize over
    policy : :obj:`DiscreteSelectionPolicy`
        a policy to use to select the next candidate to evaluate (e.g. ThompsonSampling)
    """
    def __init__(self, objective, candidates, policy):
        self.num_candidates_ = len(candidates)
        self.model_ = GaussianModel(self.num_candidates_)
        self.selection_policy_ = policy
        self.selection_policy_.set_model(self.model_)

        DiscreteAdaptiveSampler.__init__(self, objective, candidates, self.model_, self.selection_policy_)

    def reset_model(self, candidates):
        self.model_ = GaussianModel(self.num_candidates_)
        self.selection_policy_.set_model(self.model_) # always update the selection policy!

class GaussianUniformAllocationMean(GaussianBandit):
    """
    Uniform Allocation with Independent Gaussian Multi-Armed Bandit model

    Attributes
    ----------
    objective : :obj:`Objective`
        the objective to optimize via sampling
    candidates : :obj:`list` of arbitrary objects that can be evaluted by the objective
        the list of candidates to optimize over
    """
    def __init__(self, objective, candidates):
        GaussianBandit.__init__(self, objective, candidates, UniformSelectionPolicy())

class GaussianThompsonSampling(GaussianBandit):
    """
    Thompson Sampling with Independent Gaussian Multi-Armed Bandit model

    Attributes
    ----------
    objective : :obj:`Objective`
        the objective to optimize via sampling
    candidates : :obj:`list` of arbitrary objects that can be evaluted by the objective
        the list of candidates to optimize over
    """
    def __init__(self, objective, candidates):
        GaussianBandit.__init__(self, objective, candidates, ThompsonSelectionPolicy())

class GaussianUCBSampling(GaussianBandit):
    """
    UCB with Independent Gaussian Multi-Armed Bandit model

    Attributes
    ----------
    objective : :obj:`Objective`
        the objective to optimize via sampling
    candidates : :obj:`list` of arbitrary objects that can be evaluted by the objective
        the list of candidates to optimize over
    """
    def __init__(self, objective, candidates):
        GaussianBandit.__init__(self, objective, candidates, GaussianUCBPolicy())

# Correlated Beta-Bernoulli bandit models
class CorrelatedBetaBernoulliBandit(DiscreteAdaptiveSampler):
    """
    Multi-Armed Bandit class using Continuous Correlated Beta Processes (CCBPs) to model the objective value of each candidate.

    Attributes
    ----------
    objective : :obj:`Objective`
        the objective to optimize via sampling
    candidates : :obj:`list` of arbitrary objects that can be evaluted by the objective
        the list of candidates to optimize over
    policy : :obj:`DiscreteSelectionPolicy`
        a policy to use to select the next candidate to evaluate (e.g. ThompsonSampling)
    nn : :obj:`NearestNeighbor`
        nearest neighbor structure for fast lookups during module updates
    kernel : :obj:`Kernel`
        kernel to use in CCBP model
    alpha_prior : float
        the prior to use on the alpha parameter 
    beta_prior : float
        the prior to use on the beta parameter
    p : float
        the lower confidence bound used for best arm prediction (e.g. 0.95 -> return the 5th percentile of the belief distribution as the estimated objective value for each candidate)
    """
    def __init__(self, objective, candidates, policy, nn, kernel, tolerance=1e-4, alpha_prior=1.0, beta_prior=1.0, p=0.95):
        self.num_candidates_ = len(candidates)
        self.model_ = CorrelatedBetaBernoulliModel(candidates, nn, kernel, tolerance, alpha_prior, beta_prior, p)
        self.selection_policy_ = policy
        self.selection_policy_.set_model(self.model_)

        DiscreteAdaptiveSampler.__init__(self, objective, candidates, self.model_, self.selection_policy_)

    def reset_model(self, candidates):
        """ Needed to independently maximize over subsets of data """
        self.model_ = CorrelatedBetaBernoulliModel(
            self.candidates_, self.model_.nn_, self.model_.kernel_,
            self.model_.tolerance_, self.model_.alpha_prior_, self.model_.beta_prior_, p=self.model_.p_
        )
        self.selection_policy_.set_model(self.model_) # always update the selection policy!

class CorrelatedThompsonSampling(CorrelatedBetaBernoulliBandit):
    """
    Thompson Sampling with CCBP Multi-Armed Bandit model

    Attributes
    ----------
    objective : :obj:`Objective`
        the objective to optimize via sampling
    candidates : :obj:`list` of arbitrary objects that can be evaluted by the objective
        the list of candidates to optimize over
    nn : :obj:`NearestNeighbor`
        nearest neighbor structure for fast lookups during module updates
    kernel : :obj:`Kernel`
        kernel to use in CCBP model
    alpha_prior : float
        the prior to use on the alpha parameter 
    beta_prior : float
        the prior to use on the beta parameter
    p : float
        the lower confidence bound used for best arm prediction (e.g. 0.95 -> return the 5th percentile of the belief distribution as the estimated objective value for each candidate)
    """
    def __init__(self, objective, candidates, nn, kernel,
                 tolerance=1e-4, alpha_prior=1.0, beta_prior=1.0, p=0.95):
        CorrelatedBetaBernoulliBandit.__init__(
            self, objective, candidates, ThompsonSelectionPolicy(),
            nn, kernel, tolerance, alpha_prior, beta_prior, p
        )

class CorrelatedBayesUCB(CorrelatedBetaBernoulliBandit):
    """
    Bayes UCB with CCBP Multi-Armed Bandit model (see "On Bayesian Upper Confidence Bounds for Bandit Problems" by Kaufmann et al.)

    Attributes
    ----------
    objective : :obj:`Objective`
        the objective to optimize via sampling
    candidates : :obj:`list` of arbitrary objects that can be evaluted by the objective
        the list of candidates to optimize over
    nn : :obj:`NearestNeighbor`
        nearest neighbor structure for fast lookups during module updates
    kernel : :obj:`Kernel`
        kernel to use in CCBP model
    tolerance : float
        TODO
    alpha_prior : float
        the prior to use on the alpha parameter 
    beta_prior : float
        the prior to use on the beta parameter
    horizon : int
        horizon parameter for Bayes UCB
    c : int
        quantile parameter for Bayes UCB
    p : float
        the lower confidence bound used for best arm prediction (e.g. 0.95 -> return the 5th percentile of the belief distribution as the estimated objective value for each candidate)
    """
    def __init__(self, objective, candidates, nn, kernel, tolerance=1e-4,
                 alpha_prior=1.0, beta_prior=1.0, horizon=1000, c=6, p=0.95):
        policy = BetaBernoulliBayesUCBPolicy(horizon=horizon, c=c)
        CorrelatedBetaBernoulliBandit.__init__(
            self, objective, candidates, policy,
            nn, kernel, tolerance, alpha_prior, beta_prior, p
        )

class CorrelatedGittins(CorrelatedBetaBernoulliBandit):
    """"
    Gittins Index Policy for gamma=0.98 with CCBP Multi-Armed Bandit model

    Attributes
    ----------
    objective : :obj:`Objective`
        the objective to optimize via sampling
    candidates : :obj:`list` of arbitrary objects that can be evaluted by the objective
        the list of candidates to optimize over
    nn : :obj:`NearestNeighbor`
        nearest neighbor structure for fast lookups during module updates
    kernel : :obj:`Kernel`
        kernel to use in CCBP model
    alpha_prior : float
        the prior to use on the alpha parameter 
    beta_prior : float
        the prior to use on the beta parameter
    p : float
        the lower confidence bound used for best arm prediction (e.g. 0.95 -> return the 5th percentile of the belief distribution as the estimated objective value for each candidate)
    """
    def __init__(self, objective, candidates, nn, kernel, tolerance=1e-4,
                 alpha_prior=1.0, beta_prior=1.0, p=0.95):
        policy = BetaBernoulliGittinsIndex98Policy()
        CorrelatedBetaBernoulliBandit.__init__(
            self, objective, candidates, policy,
            nn, kernel, tolerance, alpha_prior, beta_prior, p
        )

