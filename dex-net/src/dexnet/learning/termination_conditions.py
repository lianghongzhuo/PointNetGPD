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
Classes for termination conditions for optimization modules
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

# class TerminationCondition(metaclass=ABCMeta):
class TerminationCondition:
    """
    Returns true when a condition is satisfied. Used for supplying different termination conditions to optimization algorithms
    """
    __metaclass__ = ABCMeta
    @abstractmethod
    def __call__(self, k, cur_val = None, prev_val = None, cur_grad = None, cur_hess = None, model = None):
        """
        Returns true or false based on whether or not a termination condition was met

        Parameters
        ----------
        k : :obj:`int`
            current iteration
        cur_val : :obj:`Number`
            most recent result of objective evaluation
        prev_val : :obj:`Number`
            previous result of objective evaluation
        cur_grad : :obj:`Number` or numpy :obj:`ndarray`
            gradient of objective at most recent input
        cur_hess : :obj:`Number` or numpy :obj:`ndarray`
            hessian of objective at most recent input
        model : :obj:`Model`
            the model being used

        Returns
        -------
        :obj:`bool`
            True if the condition is satisfied, False otherwise
        """
        pass

class MaxIterTerminationCondition(TerminationCondition):
    """
    Terminate based on reaching a maximum number of iterations.

    Attributes
    ----------
    max_iters : :obj:`int`
        the maximum number of allowed iterations
    """
    def __init__(self, max_iters):
        self.max_iters_ = max_iters

    def __call__(self, k, cur_val, prev_val, cur_grad = None, cur_hess = None, model = None):
        return (k >= self.max_iters_)

class ProgressTerminationCondition(TerminationCondition):
    """
    Terminate based on lack of progress.

    Attributes
    ----------
    eps : :obj:`float`
        the minimum admissible progress that must be made on each iteration to continue
    """
    def __init__(self, eps):
        self.eps_ = eps

    def __call__(self, k, cur_val, prev_val, cur_grad = None, cur_hess = None, model = None):
        return (abs(cur_val - prev_val) < self.eps_)

class ConfidenceTerminationCondition(TerminationCondition):
    """
    Terminate based on model confidence.

    Attributes
    ----------
    eps : :obj:`float`
        the amount of confidence in the predicted objective value that the model must have to terminate
    """
    def __init__(self, eps):
        self.eps_ = eps

    def __call__(self, k, cur_val, prev_val, cur_grad = None, cur_hess = None, model = None):
        max_ind, max_mean, max_var = model.max_prediction()
        return (max_var[0] < self.eps_)
    
class OrTerminationCondition(TerminationCondition):
    """
    Terminate based on the OR of several termination conditions

    Attributes
    ----------
    term_conditions : :obj:`list` of :obj:`TerminationCondition`
        termination conditions that are ORed to get the final termination results     
    """
    def __init__(self, term_conditions):
        self.term_conditions_ = term_conditions

    def __call__(self, k, cur_val, prev_val, cur_grad = None, cur_hess = None, model = None):
        terminate = False
        for term_condition in self.term_conditions_:
            terminate = terminate or term_condition(k, cur_val, prev_val, cur_grad, cur_hess, model)
        return terminate

class AndTerminationCondition(TerminationCondition):
    """
    Terminate based on the AND of several termination conditions

    Attributes
    ----------
    term_conditions : :obj:`list` of :obj:`TerminationCondition`
        termination conditions that are ANDed to get the final termination results     
    """
    def __init__(self, term_conditions):
        self.term_conditions_ = term_conditions

    def __call__(self, k, cur_val, prev_val, cur_grad = None, cur_hess = None, model = None):
        terminate = True
        for term_condition in self.term_conditions_:
            terminate = terminate and term_condition(k, cur_val, prev_val, cur_grad, cur_hess, model)
        return terminate
