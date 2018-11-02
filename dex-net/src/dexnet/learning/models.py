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
Classes for predicting some (possibly non-deterministic) value over a set of discrete candidates or continuous space

Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import copy
import numpy as np
import scipy.stats
import numbers

import IPython

# class Model(metaclass=ABCMeta):
class Model:
    """
    A predictor of some value of the input data
    """
    __metaclass__ = ABCMeta

    def __call__(self, x):
        return self.predict(x)

    @abstractmethod
    def predict(self, x):
        """
        Predict the function of the data at some point x. For probabilistic models this returns the mean prediction
        """
        pass

    @abstractmethod
    def update(self):
        """
        Update the model based on current data
        """
        pass

    @abstractmethod
    def snapshot(self):
        """
        Returns a concise description of the current model for debugging and logging purposes
        """
        pass

class DiscreteModel(Model):
    """
    Maintains a prediction over a discrete set of points
    """
    @abstractmethod
    def max_prediction(self):
        """
        Returns the index (or indices), posterior mean, and posterior variance of the variable(s) with the
        maximal mean predicted value
        """
        pass

    @abstractmethod
    def sample(self):
        """
        Sample discrete predictions from the model. For deterministic models, returns the deterministic prediction
        """
        pass

    def num_vars(self):
        """Returns the number of variables in the model"""
        return self.num_vars_


# class Snapshot(metaclass=ABCMeta):
class Snapshot:
    """ Abstract class for storing the current state of a model """
    __metaclass__ = ABCMeta

    def __init__(self, best_pred_ind, num_obs):
        self.best_pred_ind = best_pred_ind
        self.num_obs = copy.copy(num_obs)

class BernoulliSnapshot(Snapshot):
    """ Stores the current state of a Bernoulli model """
    def __init__(self, best_pred_ind, means, num_obs):
        Snapshot.__init__(self, best_pred_ind, num_obs)
        self.means = copy.copy(means)

class BetaBernoulliSnapshot(Snapshot):
    """ Stores the current state of a Beta Bernoulli model """
    def __init__(self, best_pred_ind, alphas, betas, num_obs):
        Snapshot.__init__(self, best_pred_ind, num_obs)
        self.alphas = copy.copy(alphas)
        self.betas = copy.copy(betas)

class GaussianSnapshot(Snapshot):
    """ Stores the current state of a Gaussian model """
    def __init__(self, best_pred_ind, means, variances, sample_vars, num_obs):
        Snapshot.__init__(self, best_pred_ind, num_obs)
        self.means = copy.copy(means)
        self.variances = copy.copy(variances)
        self.sample_vars = copy.copy(sample_vars)

class BernoulliModel(DiscreteModel):
    """
    Standard bernoulli model for predictions over a discrete set of candidates
    
    Attributes
    ----------
    num_vars: :obj:`int`
        the number of variables to track
    prior_means: (float) prior on mean probabilty of success for candidates
    """
    def __init__(self, num_vars, mean_prior = 0.5):
        if num_vars <= 0:
            raise ValueError('Must provide at least one variable to BetaBernoulliModel')

        self.num_vars_ = num_vars
        self.mean_prior_  = mean_prior

        self._init_model_params()

    def _init_model_params(self):
        """
        Allocates numpy arrays for the estimated alpha and beta values for each variable,
        and the number of observations for each
        """
        self.pred_means_ = self.mean_prior_ * np.ones(self.num_vars_)
        self.num_observations_ = np.zeros(self.num_vars_)

    @staticmethod
    def bernoulli_mean(p):
        """ Mean of the beta distribution with params alpha and beta """
        return p

    @staticmethod
    def bernoulli_variance(p, n):
        """ Uses Wald interval for variance prediction """
        sqrt_p_n = np.sqrt(p * (1 - p) / n)
        z = scipy.stats.norm.cdf(0.68) # cdf for standard Gaussian, 1 -sigma deviation
        return 2 * z * sqrt_p_n

    def predict(self, index):
        """
        Predicts the probability of success for the variable indexed by index
        """
        return BernoulliModel.bernoulli_mean(self.pred_means_[index])

    def max_prediction(self):
        """
        Returns the index (or indices), posterior mean, and posterior variance of the variable(s) with the
        maximal mean probaiblity of success
        """
        mean_posteriors = BernoulliModel.bernoulli_mean(self.pred_means_)
        max_indices = np.where(mean_posteriors == np.max(mean_posteriors))[0]
        max_posterior_means = mean_posteriors[max_indices]
        max_posterior_vars = BernoulliModel.bernoulli_variance(self.pred_means_[max_indices], self.num_observations_[max_indices])

        return max_indices, max_posterior_means, max_posterior_vars

    def update(self, index, value):
        """
        Update the model based on an observation of value at index index
        """
        if value < 0 or value > 1:
            raise ValueError('Values must be between 0 and 1')

        self.pred_means_[index] = self.pred_means_[index] * (self.num_observations_[index] / (self.num_observations_[index] + 1)) + \
            value * (1.0 / (self.num_observations_[index] + 1));
        self.num_observations_[index] = self.num_observations_[index] + 1

    def snapshot(self):
        """
        Return copys of the model params
        """
        ind, mn, var = self.max_prediction()
        return BernoulliSnapshot(ind[0], self.pred_means_, self.num_observations_)

    def sample(self):
        """
        Samples probabilities of success from the given values
        """
        return self.pred_means_

class BetaBernoulliModel(DiscreteModel):
    """
    Beta-Bernoulli model for predictions over a discrete set of candidates
    Attributes
    ----------
    num_vars : int
        the number of variables to track
    alpha_prior : float
        prior alpha parameter of the Beta distribution 
    beta_prior : float
        prior beta parameter of the Beta distribution 
    """
    def __init__(self, num_vars, alpha_prior = 1., beta_prior = 1.):
        if num_vars <= 0:
            raise ValueError('Must provide at least one variable to BetaBernoulliModel')

        self.num_vars_ = num_vars
        self.alpha_prior_  = alpha_prior
        self.beta_prior_  = beta_prior

        self._init_model_params()

    def _init_model_params(self):
        """
        Allocates numpy arrays for the estimated alpha and beta values for each variable, and the number of observations for each
        """
        if isinstance(self.alpha_prior_, numbers.Number):
            self.posterior_alphas_ = self.alpha_prior_ * np.ones(self.num_vars_)
        else:
            self.posterior_alphas_ = np.array(self.alpha_prior_)

        if isinstance(self.alpha_prior_, numbers.Number):
            self.posterior_betas_ = self.beta_prior_ * np.ones(self.num_vars_)
        else:
            self.posterior_betas_ = np.array(self.beta_prior_)
        
        self.num_observations_ = np.zeros(self.num_vars_)

    @staticmethod
    def beta_mean(alpha, beta):
        """ Mean of the beta distribution with params alpha and beta """
        return alpha / (alpha + beta)

    @staticmethod
    def beta_variance(alpha, beta):
        """ Mean of the beta distribution with params alpha and beta """
        return (alpha * beta) / ( (alpha + beta)**2 * (alpha + beta + 1))

    @staticmethod
    def sample_variance(alpha, beta):
        """ Mean of the beta distribution with params alpha and beta """
        mean = BetaBernoulliModel.beta_mean(alpha, beta)
        sample_variance = (1.0 / (alpha + beta)) * (alpha * (1 - mean)**2 + beta * mean**2)
        return sample_variance

    @property
    def posterior_alphas(self):
        return self.posterior_alphas_

    @property
    def posterior_betas(self):
        return self.posterior_betas_

    def predict(self, index):
        """
        Predicts the probability of success for the variable indexed by index """
        return BetaBernoulliModel.beta_mean(self.posterior_alphas_[index], self.posterior_betas_[index])

    def max_prediction(self):
        """
        Returns the index (or indices), posterior mean, and posterior variance of the variable(s) with the
        maximal mean probaiblity of success
        """
        mean_posteriors = BetaBernoulliModel.beta_mean(self.posterior_alphas_, self.posterior_betas_)
        max_indices = np.where(mean_posteriors == np.max(mean_posteriors))[0]
        max_posterior_means = mean_posteriors[max_indices]
        max_posterior_vars = BetaBernoulliModel.beta_variance(self.posterior_alphas_[max_indices], self.posterior_betas_[max_indices])

        return max_indices, max_posterior_means, max_posterior_vars

    def update(self, index, value):
        """
        Update the model based on an observation of value at index index
        """
        if value < 0 or value > 1:
            raise ValueError('Values must be between 0 and 1')

        self.posterior_alphas_[index] = self.posterior_alphas_[index] + value
        self.posterior_betas_[index] = self.posterior_betas_[index] + (1.0 - value)
        self.num_observations_[index] = self.num_observations_[index] + 1

    def snapshot(self):
        """
        Return copies of the model params
        """
        ind, mn, var = self.max_prediction()
        return BetaBernoulliSnapshot(ind[0], self.posterior_alphas_, self.posterior_betas_, self.num_observations_)

    def sample(self, vis = False, stop = False):
        """
        Samples probabilities of success from the given values
        """
        #samples = np.random.beta(self.posterior_alphas_, self.posterior_betas_)
        samples = scipy.stats.beta.rvs(self.posterior_alphas_, self.posterior_betas_)
        if stop:
            IPython.embed()
        if vis:
            print('Samples')
            print(samples)
            print('Estimated mean')
            print((BetaBernoulliModel.beta_mean(self.posterior_alphas_, self.posterior_betas_)))
            print('At best index')
            print((BetaBernoulliModel.beta_mean(self.posterior_alphas_[21], self.posterior_betas_[21])))
        return samples

class GaussianModel(DiscreteModel):
    """
    Gaussian model for predictions over a discrete set of candidates.

    Attributes
    ----------
    num_vars : int
        the number of variables to track
    """
    def __init__(self, num_vars):
        if num_vars <= 0:
            raise ValueError('Must provide at least one variable to GaussianModel')

        self.num_vars_ = num_vars
        self._init_model_params()

    def _init_model_params(self):
        self.means_ = np.zeros(self.num_vars_)
        self.squared_means_ = np.zeros(self.num_vars_)
        self.num_observations_ = np.zeros(self.num_vars_)

    @property
    def means(self):
        return self.means_

    @property
    def variances(self):
        """ Confidence bounds on the mean """
        if np.max(self.num_observations_) == 0:
            return self.sample_vars
        return self.sample_vars / np.sqrt(self.num_observations_)

    @property
    def sample_vars(self):
        return self.squared_means_ - self.means_**2

    def predict(self, index):
        """Predict the value of the index'th variable.

        Parameters
        ----------
        index : int
            the variable to find the predicted value for
        """
        return self.means_[index]

    def max_prediction(self):
        """Returns the index, mean, and variance of the variable(s) with the
        maximal predicted value.
        """
        max_mean = np.max(self.means_)
        max_indices = np.where(self.means_ == max_mean)[0]
        max_posterior_means = self.means[max_indices]
        max_posterior_vars = self.variances[max_indices]

        return max_indices, max_posterior_means, max_posterior_vars

    def update(self, index, value):
        """Update the model based on current data.

        Parameters
        ----------
        index : int
            the index of the variable that was evaluated
        value : float
            the value of the variable
        """
        old_mean = self.means_[index]
        old_squared_mean = self.squared_means_[index]
        n = self.num_observations_[index]

        self.means_[index] = (old_mean * n + value) / (n + 1)
        self.squared_means_[index] = (old_squared_mean * n + value**2) / (n + 1)
        self.num_observations_[index] += 1

    def sample(self, stop=False):
        """Sample discrete predictions from the model. Mean follows a t-distribution"""
        samples = scipy.stats.t.rvs(self.num_observations_ - np.ones(self.num_vars_),
                                    self.means, self.variances)
        return samples

    def snapshot(self):
        """Returns a concise description of the current model for debugging and
        logging purposes.
        """
        ind, mn, var = self.max_prediction()
        return GaussianSnapshot(ind[0], self.means, self.variances,
                                self.sample_vars,
                                self.num_observations_)

class CorrelatedBetaBernoulliModel(BetaBernoulliModel):
    """Correlated Beta-Bernoulli model for predictions over a discrete set of
    candidates.

    Attributes
    ----------
    candidates : :obj:`list`
        the objects to track
    nn : :obj:`NearestNeighbor`
        nearest neighbor structure to use for neighborhood lookups
    kernel : :obj:`Kernel`
        kernel instance to measure similarities
    tolerance : float
        for computing radius of neighborhood, between 0 and 1
    alpha_prior : float
        prior alpha parameter of the Beta distribution 
    beta_prior : float
        prior beta parameter of the Beta distribution 
    """
    def __init__(self, candidates, nn, kernel, tolerance=1e-2,
                 alpha_prior=1.0, beta_prior=1.0, p=0.5):
        BetaBernoulliModel.__init__(self, len(candidates), alpha_prior, beta_prior)
        self.candidates_ = candidates

        self.kernel_ = kernel
        self.tolerance_ = tolerance
        self.error_radius_ = kernel.error_radius(tolerance)
        self.kernel_matrix_ = None
        self.p_ = p

        self.nn_ = nn
        self.nn_.train(candidates)

    @property
    def kernel_matrix(self):
        """
        Create the full kernel matrix for debugging purposes
        """
        if self.kernel_matrix_ is None:
            self.kernel_matrix_ = self.kernel_.matrix(self.candidates_)
        return self.kernel_matrix_

    def update(self, index, value):
        """Update the model based on current data

        Parameters
        ----------
        index : int
            the index of the variable that was evaluated
        value : float
            the value of the variable
        """
        if not (0 <= value <= 1):
            raise ValueError('Values must be between 0 and 1')

        # find neighbors within radius
        candidate = self.candidates_[index]
        neighbor_indices, _ = self.nn_.within_distance(candidate, self.error_radius_,
                                                       return_indices=True)
        # create array of correlations
        correlations = np.zeros(self.num_vars_)
        for neighbor_index in neighbor_indices:
            neighbor = self.candidates_[neighbor_index]
            correlations[neighbor_index] = self.kernel_(candidate, neighbor)

        self.posterior_alphas_ = self.posterior_alphas_ + value * correlations
        self.posterior_betas_ = self.posterior_betas_ + (1.0 - value) * correlations

        # TODO: should num_observations_ be updated by correlations instead?
        self.num_observations_[index] += 1.0

    def lcb_prediction(self, p=0.95):
        """ Return the index with the highest lower confidence bound """
        lcb, ucb = scipy.stats.beta.interval(p, self.posterior_alphas_, self.posterior_betas_)
        max_indices = np.where(lcb == np.max(lcb))[0]
        posterior_means = BetaBernoulliModel.beta_mean(self.posterior_alphas_[max_indices], self.posterior_betas[max_indices])
        posterior_vars = BetaBernoulliModel.beta_variance(self.posterior_alphas_[max_indices], self.posterior_betas[max_indices])

        return max_indices, posterior_means, posterior_vars        

    def snapshot(self):
        """
        Return copys of the model params
        """
        #ind, mn, var = self.max_prediction()
        ind, mn, var = self.lcb_prediction(self.p_)
        return BetaBernoulliSnapshot(ind[0], self.posterior_alphas_, self.posterior_betas_, self.num_observations_)
