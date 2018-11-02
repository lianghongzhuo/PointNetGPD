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
from dexnet.learning.models import Model, DiscreteModel, Snapshot, BernoulliSnapshot, BetaBernoulliSnapshot, \
    GaussianSnapshot, BernoulliModel, BetaBernoulliModel, GaussianModel, CorrelatedBetaBernoulliModel
from dexnet.learning.termination_conditions import TerminationCondition, MaxIterTerminationCondition, \
    ProgressTerminationCondition, ConfidenceTerminationCondition, OrTerminationCondition, AndTerminationCondition
from dexnet.learning.discrete_selection_policies import DiscreteSelectionPolicy, UniformSelectionPolicy, \
    MaxDiscreteSelectionPolicy, ThompsonSelectionPolicy, BetaBernoulliGittinsIndex98Policy, \
    BetaBernoulliBayesUCBPolicy, GaussianUCBPolicy
from dexnet.learning.objectives import Objective, DifferentiableObjective, MaximizationObjective, \
    MinimizationObjective, NonDeterministicObjective, ZeroOneObjective, IdentityObjective, \
    RandomBinaryObjective, RandomContinuousObjective, LeastSquaresObjective, \
    LogisticCrossEntropyObjective, CrossEntropyLoss, SquaredErrorLoss, WeightedSquaredErrorLoss, CCBPLogLikelihood
from dexnet.learning.solvers import Solver, TopKSolver, SamplingSolver, DiscreteSamplingSolver, OptimizationSolver
from dexnet.learning.discrete_adaptive_samplers import AdaptiveSamplingResult, DiscreteAdaptiveSampler, \
    BetaBernoulliBandit, UniformAllocationMean, ThompsonSampling, GittinsIndex98, GaussianBandit, \
    GaussianUniformAllocationMean, GaussianThompsonSampling, GaussianUCBSampling, \
    CorrelatedBetaBernoulliBandit, CorrelatedThompsonSampling, CorrelatedBayesUCB, CorrelatedGittins
from dexnet.learning.analysis import ConfusionMatrix, ClassificationResult, RegressionResult

from dexnet.learning.tensor_dataset import Tensor, TensorDataset

__all__ = ['Model', 'DiscreteModel', 'Snapshot', 'BernoulliSnapshot', 'BetaBernoulliSnapshot', 'GaussianSnapshot',
           'BernoulliModel', 'BetaBernoulliModel', 'GaussianModel', 'CorrelatedBetaBernoulliModel',
           'TerminationCondition', 'MaxIterTerminationCondition', 'ProgressTerminationCondition',
           'ConfidenceTerminationCondition', 'OrTerminationCondition', 'AndTerminationCondition',
           'DiscreteSelectionPolicy', 'UniformSelectionPolicy', 'MaxDiscreteSelectionPolicy',
           'ThompsonSelectionPolicy', 'BetaBernoulliGittinsIndex98Policy', 'BetaBernoulliBayesUCBPolicy',
           'GaussianUCBPolicy',
           'Objective', 'DifferentiableObjective', 'MaximizationObjective', 'MinimizationObjective',
           'NonDeterministicObjective', 'ZeroOneObjective', 'IdentityObjective', 'RandomBinaryObjective',
           'RandomContinuousObjective', 'LeastSquaresObjective', 'LogisticCrossEntropyObjective',
           'CrossEntropyLoss', 'SquaredErrorLoss', 'WeightedSquaredErrorLoss', 'CCBPLogLikelihood',
           'Solver', 'TopKSolver', 'SamplingSolver', 'DiscreteSamplingSolver', 'OptimizationSolver',
           'AdaptiveSamplingResult', 'DiscreteAdaptiveSampler', 'BetaBernoulliBandit',
           'UniformAllocationMean', 'ThompsonSampling', 'GittinsIndex98', 'GaussianBandit',
           'GaussianUniformAllocationMean', 'GaussianThompsonSampling', 'GaussianUCBSampling',
           'CorrelatedBetaBernoulliBandit', 'CorrelatedThompsonSampling', 'CorrelatedBayesUCB',
           'CorrelatedGittins',
           'ConfusionMatrix', 'ClassificationResult', 'RegressionResult',
           'Tensor', 'TensorDataset']
