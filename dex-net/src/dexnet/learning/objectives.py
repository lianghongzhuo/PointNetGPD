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
Objectives that place some value on a set on input points
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import numbers
import numpy as np
import scipy.stats as ss

# class Objective(metaclass=ABCMeta):
class Objective:
    """ Acts as a function that returns a numeric value for classes of input data, with checks for valid input. """
    __metaclass__ = ABCMeta

    def __call__(self, x):
        return self.evaluate(x)

    @abstractmethod
    def evaluate(self, x):
        """ Evaluates a function to be maximized at some point x.

        Parameters
        ----------
        x : :obj:`object`
            point at which to evaluate the objective
        """
        pass

    @abstractmethod
    def check_valid_input(self, x):
        """ Return whether or not a point is valid for the objective.

        Parameters
        ----------
        x : :obj:`object`
            point at which to evaluate the objective
        """
        pass

# class DifferentiableObjective(Objective, metaclass=ABCMeta):
class DifferentiableObjective(Objective):
    """ Objectives that are at least two-times differentable. """
    __metaclass__ = ABCMeta

    @abstractmethod
    def gradient(self, x):
        """ Evaluate the gradient at x.

        Parameters
        ----------
        x : :obj:`object`
            point at which to evaluate the objective
        """
        pass

    @abstractmethod
    def hessian(self, x):
        """ Evaluate the hessian at x.

        Parameters
        ----------
        x : :obj:`object`
            point at which to evaluate the objective
        """
        pass

class MaximizationObjective(DifferentiableObjective):
    """ Wrapper for maximization of some supplied objective function. Actually not super important, here for symmetry.

    Attributes
    ----------
    obj : :obj:`Objective`
        objective function to maximize
    """
    def __init__(self, obj):
        """ obj is the objective to call """
        if not isinstance(obj, Objective):
            raise ValueError("Function must be a single argument objective")
        self.obj_ = obj

    def check_valid_input(self, x):
        self.obj_.check_valid_input(x)

    def evaluate(self, x):
        return self.obj_(x)

    def gradient(self, x):
        if not isinstance(self.obj_, DifferentiableObjective):
            raise ValueError("Objective is non-differentiable")
        return self.obj_.gradient(x)

    def hessian(self, x):
        if not isinstance(self.obj_, DifferentiableObjective):
            raise ValueError("Objective is non-differentiable")
        return self.obj_.hessian(x)

class MinimizationObjective(DifferentiableObjective):
    """ Wrapper for minimization of some supplied objective function. Used because internally all solvers attempt to maximize by default.

    Attributes
    ----------
    obj : :obj:`Objective`
        objective function to minimize
    """
    def __init__(self, obj):
        """ obj is the objective to call """
        if not isinstance(obj, Objective):
            raise ValueError("Function must be a single argument objective")
        self.obj_ = obj

    def check_valid_input(self, x):
        self.obj_.check_valid_input(x)

    def evaluate(self, x):
        """ Return negative, as all solvers will be assuming a maximization """
        return -self.obj_(x)

    def gradient(self, x):
        if not isinstance(self.obj_, DifferentiableObjective):
            raise ValueError("Objective is non-differentiable")
        return -self.obj_.gradient(x)

    def hessian(self, x):
        if not isinstance(self.obj_, DifferentiableObjective):
            raise ValueError("Objective is non-differentiable")
        return -self.obj_.hessian(x)

class NonDeterministicObjective(Objective):
    """ Wrapper for non-deterministic objective function evaluations. Samples random values of the input data x.

    Attributes
    ----------
    det_objective : :obj:`Objective`
        deterministic objective function to optimize
    """
    def __init__(self, det_objective):
        self.det_objective_ = det_objective

    def evaluate(self, x):
        """ Evaluates a function to be maximized at some point x.

        Parameters
        ----------
        x : :obj:`object` with a sample() function
            point at which to evaluate the nondeterministic objective
        """
        if not hasattr(x, "sample"):
            raise ValueError("Data points must have a sampling function returning a 0 or 1")

        x_val = x.sample()
        return self.det_objective_.evaluate(x_val)

class ZeroOneObjective(Objective):
    """ Zero One Loss based on thresholding.

    Attributes
    ----------
    b : :obj:`int`
        threshold value, 1 iff x > b, 0 otherwise
    """
    def __init__(self, b = 0):
        self.b_ = b

    def check_valid_input(self, x):
        """ Check whether or not input is valid for the objective """
        if not isinstance(x, numbers.Number):
            raise ValueError("Zero-One objective can only be evaluated on numbers")

    def evaluate(self, x):
        self.check_valid_input(x)
        return x >= self.b_

class IdentityObjective(Objective):
    """ Just returns the value x """
    def check_valid_input(self, x):
        """ Check whether or not input is valid for the objective """
        if not isinstance(x, numbers.Number):
            raise ValueError("Zero-One objective can only be evaluated on numbers")

    def evaluate(self, x):
        self.check_valid_input(x)
        return x

class RandomBinaryObjective(NonDeterministicObjective):
    """
    Returns a 0 or 1 based on some underlying random probability of success for the data points
    Evaluated data points must have a sample_success method that returns 0 or 1
    """
    def __init__(self):
        NonDeterministicObjective.__init__(self, ZeroOneObjective(0.5))

    def check_valid_input(self, x):
        """ Check whether or not input is valid for the objective """
        if not isinstance(x, numbers.Number):
            raise ValueError("Random binary objective can only be evaluated on numbers")

class RandomContinuousObjective(NonDeterministicObjective):
    """
    Returns a continuous value based on some underlying random probability of success for the data points
    Evaluated data points must have a sample method
    """
    def __init__(self):
        NonDeterministicObjective.__init__(self, IdentityObjective())

    def check_valid_input(self, x):
        """ Check whether or not input is valid for the objective """
        if not isinstance(x, numbers.Number):
            raise ValueError("Random continuous objective can only be evaluated on numbers")

class LeastSquaresObjective(DifferentiableObjective):
    """ Classic least-squares loss 0.5 * norm(Ax - b)**2

    Attributes
    ----------
    A : :obj:`numpy.ndarray`
        A matrix in least squares 0.5 * norm(Ax - b)**2
    b : :obj:`numpy.ndarray`
        b vector in least squares 0.5 * norm(Ax - b)**2
    """
    def __init__(self, A, b):
        self.A_ = A
        self.b_ = b

        self.x_dim_ = A.shape[1]
        self.b_dim_ = A.shape[0]
        if self.b_dim_ != b.shape[0]:
            raise ValueError('A and b must have same dimensions')

    def check_valid_input(self, x):
        if not isinstance(x, np.ndarray):
            raise ValueError('Least squares objective only works with numpy ndarrays!')
        if x.shape[0] != self.x_dim_:
            raise ValueError('x values must have same dimensions as number of columns of A')

    def evaluate(self, x):
        self.check_valid_input(x)
        return 0.5 * (x.T.dot(self.A_.T).dot(self.A_).dot(x) - 2 * self.b_.T.dot(self.A_).dot(x) + self.b_.T.dot(self.b_))

    def gradient(self, x):
        self.check_valid_input(x)
        return self.A_.T.dot(self.A_).dot(x) - self.A_.T.dot(self.b_)

    def hessian(self, x):
        self.check_valid_input(x)
        return self.A_.T.dot(self.A_)

class LogisticCrossEntropyObjective(DifferentiableObjective):
    """ Logistic cross entropy loss.

    Attributes
    ----------
    X : :obj:`numpy.ndarray`
        X matrix in logistic function 1 / (1 + exp(- X^T beta)
    y : :obj:`numpy.ndarray`
        y vector, true labels
    """
    def __init__(self, X, y):
        self.X_ = X
        self.y_ = y

    def check_valid_input(self, beta):
        if not isinstance(beta, np.ndarray):
            raise ValueError('Logistic cross-entropy objective only works with np.ndarrays!')
        if self.X_.shape[1] != beta.shape[0]:
            raise ValueError('beta dimension mismatch')

    def _mu(self, X, beta):
        return 1.0 / (1.0 + np.exp(-np.dot(X, beta)))

    def evaluate(self, beta):
        self.check_valid_input(beta)
        mu = self._mu(self.X_, beta)
        return -np.sum(self.y_ * np.log(mu) + (1 - self.y_) * np.log(1 - mu))

    def gradient(self, beta):
        self.check_valid_input(beta)
        mu = self._mu(self.X_, beta)
        return 2 * beta - np.dot(self.X_.T, self.y_ - mu)

    def hessian(self, beta):
        self.check_valid_input(beta)
        mu = self._mu(self.X_, beta)
        return 2 - np.dot(np.dot(self.X_.T, np.diag(mu * (1 - mu))), self.X_)

class CrossEntropyLoss(Objective):
    """ Cross entropy loss.

    Attributes
    ----------
    true_p : :obj:`numpy.ndarray`
        the true probabilities for all admissible datapoints
    """
    def __init__(self, true_p):
        self.true_p_ = true_p
        self.N_ = true_p.shape[0]

    def evaluate(self, est_p):
        self.check_valid_input(est_p)
        return -1.0 / self.N_ * np.sum((self.true_p_ * np.log(est_p) + (1.0 - self.true_p_) * np.log(1.0 - est_p)))
    
    def check_valid_input(self, est_p):
        if not isinstance(est_p, np.ndarray):
            raise ValueError('Cross entropy must be called with ndarray')
        if est_p.shape[0] != self.N_:
            raise ValueError('Must supply same number of datapoints as true P')

class SquaredErrorLoss(Objective):
    """ Squared error (x - x_true)**2

    Attributes
    ----------
    true_p : :obj:`numpy.ndarray`
        the true labels for all admissible inputs
    """
    def __init__(self, true_p):
        self.true_p_ = true_p
        self.N_ = true_p.shape[0]

    def evaluate(self, est_p):
        self.check_valid_input(est_p)
        return 1.0 / self.N_ * np.sum((self.true_p_ - est_p)**2)
    
    def check_valid_input(self, est_p):
        if not isinstance(est_p, np.ndarray):
            raise ValueError('Cross entropy must be called with ndarray')
        if est_p.shape[0] != self.N_:
            raise ValueError('Must supply same number of datapoints as true P')

class WeightedSquaredErrorLoss(Objective):
    """ Weighted squared error w * (x - x_true)**2

    Attributes
    ----------
    true_p : :obj:`numpy.ndarray`
        the true labels for all admissible inputs
    """
    def __init__(self, true_p):
        self.true_p_ = true_p
        self.N_ = true_p.shape[0]

    def evaluate(self, est_p, weights):
        """ Evaluates the squared loss of the estimated p with given weights

        Parameters
        ----------
        est_p : :obj:`list` of :obj:`float`
            points at which to evaluate the objective
        """
        self.check_valid_input(est_p)
        return np.sum(weights * (self.true_p_ - est_p)**2) * (1.0 / np.sum(weights))
    
    def check_valid_input(self, est_p):
        if not isinstance(est_p, np.ndarray):
            raise ValueError('Cross entropy must be called with ndarray')
        if est_p.shape[0] != self.N_:
            raise ValueError('Must supply same number of datapoints as true P')

class CCBPLogLikelihood(Objective):
    """ CCBP log likelihood of the true params under a current posterior distribution

    Attributes
    ----------
    true_p : :obj:`list` of :obj:`Number`
        true probabilities of datapoints
    """
    def __init__(self, true_p):
        self.true_p_ = true_p
        self.N_ = true_p.shape[0]

    def evaluate(self, alphas, betas):
        """ Evaluates the CCBP likelihood of the true data under estimated CCBP posterior parameters alpha and beta

        Parameters
        ----------
        alphas : :obj:`list` of :obj:`Number`
            posterior alpha values
        betas : :obj:`list` of :obj:`Number`
            posterior beta values
        """
        self.check_valid_input(alphas)
        self.check_valid_input(betas)
        log_density = ss.beta.logpdf(self.true_p_, alphas, betas)
        return (1.0 / self.N_) * np.sum(log_density)
    
    def check_valid_input(self, alphas):
        if not isinstance(alphas, np.ndarray):
            raise ValueError('CCBP ML must be called with ndarray')
        if alphas.shape[0] != self.N_:
            raise ValueError('Must supply same number of datapoints as true P')

