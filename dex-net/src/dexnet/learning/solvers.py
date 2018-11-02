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
Abstract classes for solvers

Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import numpy as np

from dexnet.constants import DEF_MAX_ITER
from dexnet.learning import MaxIterTerminationCondition
import IPython

# class Solver(metaclass=ABCMeta):
class Solver:
    __metaclass__ = ABCMeta

    def __init__(self, objective):
        self.objective_ = objective

    @abstractmethod
    def solve(self, termination_condition = MaxIterTerminationCondition(DEF_MAX_ITER),
              snapshot_rate = 1):
        '''
        Solves for the maximal / minimal point
        '''
        pass

class TopKSolver(Solver):
    def __init__(self, objective):
        Solver.__init__(self, objective)

    @abstractmethod
    def top_K_solve(self, K, termination_condition = MaxIterTerminationCondition(DEF_MAX_ITER),
                    snapshot_rate = 1):
        '''
        Solves for the top K maximal / minimal points
        '''
        pass

# class SamplingSolver(TopKSolver, metaclass=ABCMeta):
class SamplingSolver(TopKSolver):
    """ Optimization methods based on a sampling strategy"""
    __metaclass__ = ABCMeta


# class DiscreteSamplingSolver(SamplingSolver, metaclass=ABCMeta):
class DiscreteSamplingSolver(SamplingSolver):

    __metaclass__ = ABCMeta

    def __init__(self, objective, candidates):
        """
        Initialize a solver with a discrete set of candidate points
        specified in a list object
        """
        self.candidates_ = candidates # discrete candidates
        self.num_candidates_ = len(candidates)
        TopKSolver.__init__(self, objective)

    @abstractmethod
    def discrete_maximize(self, candidates, termination_condition, snapshot_rate):
        """
        Main loop for sampling-based solvers
        """
        pass

    def partition(self, K):
        """
        Partition the input space into K bins uniformly at random
        """
        candidate_bins = []
        indices = np.linspace(0, self.num_candidates_)
        indices_shuff = np.random.shuffle(indices) 
        candidates_per_bin = np.floor(float(self.num_candidates_) / float(K))

        # loop through bins, adding candidates at random
        start_i = 0
        end_i = min(start_i + candidates_per_bin, self.num_candidates_ - 1)
        for k in range(K-1):
            candidate_bins.push_back(self.candidates_[indices_shuff[start_i:end_i]])

            start_i = start_i + candidates_per_bin
            end_i = min(start_i + candidates_per_bin, self.num_candidates_ - 1)
            
        candidate_bins.push_back(self.candidates_[indices_shuff[start_i:end_i]])
        return candidate_bins

    def solve(self, termination_condition = MaxIterTerminationCondition(DEF_MAX_ITER),
              snapshot_rate = 1):
        """ Call discrete maxmization function with all candidates """
        return self.discrete_maximize(self.candidates_, termination_condition, snapshot_rate)

    def top_K_solve(self, K, termination_condition = MaxIterTerminationCondition(DEF_MAX_ITER),
                    snapshot_rate = 1):
        """ Solves for the top K maximal / minimal points """
        # partition the input space
        if K == 1:
            candidate_bins = [self.candidates_]
        else:
            candidate_bins = self.partition(K)

        # maximize over each bin
        top_K_results = []
        for k in range(K):
            top_K_results.append(self.discrete_maximize(candidate_bins[k], termination_condition, snapshot_rate))
        return top_K_results


class OptimizationSolver(Solver):
    def __init__(self, objective, ineq_constraints = None, eq_constraints = None, eps_i = 1e-2, eps_e = 1e-2):
        """
        Inequality constraints: g_i(x) <= 0
        Equality constraints: h_i(x) <= 0
        """
        self.ineq_constraints_ = ineq_constraints
        self.eq_constraints_ = eq_constraints        
        self.eps_i_ = eps_i
        self.eps_e_ = eps_e
        Solver.__init__(self, objective)

    def is_feasible(self, x):
        """ Check feasibility of a given point """
        try:
            self.objective_.check_valid_input(x)
        except ValueError as e:
            return False

        if self.ineq_constraints_ is not None:
            for g in self.ineq_constraints_:
                if np.sum(g(x) > eps_i * np.ones(g.num_outputs())) > 0:
                    return False

        if self.eq_constraints_ is not None:
            for h in self.eq_constraints_:
                if np.sum(np.abs(h(x)) > eps_e * np.ones(h.num_outputs())) > 0:
                    return False            
        return True
