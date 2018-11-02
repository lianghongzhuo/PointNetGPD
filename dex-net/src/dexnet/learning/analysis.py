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
Helper classes for analyzing machine learning output
Author: Jeff Mahler
"""
import IPython
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.metrics as sm

class ConfusionMatrix(object):
    """ Confusion matrix for classification errors """
    def __init__(self, num_categories):
        self.num_categories = num_categories

        # organized as row = true, column = pred
        self.matrix = np.zeros([num_categories, num_categories])

    def update(self, predictions, labels):
        num_pred = predictions.shape[0]
        for i in range(num_pred):
            self.matrix[labels[i].astype(np.uint16), predictions[i].astype(np.uint16)] += 1

class ClassificationResult(object):
    def __init__(self, pred_probs_list, labels_list):
        self.pred_probs = None
        self.labels = None
        
        for pred_probs, labels in zip(pred_probs_list, labels_list):
            if self.pred_probs is None:
                self.pred_probs = pred_probs
                self.labels = labels
            else:
                self.pred_probs = np.r_[self.pred_probs, pred_probs]
                self.labels = np.r_[self.labels, labels]

    @property
    def error_rate(self):
        return 100.0 - (
            100.0 *
            np.sum(self.predictions == self.labels) /
            self.num_datapoints)

    def top_k_error_rate(self, k):
        predictions_arr = self.top_k_predictions(k)
        labels_arr = np.zeros(predictions_arr.shape)
        for i in range(k):
            labels_arr[:,i] = self.labels

        return 100.0 - (
            100.0 *
            np.sum(predictions_arr == labels_arr) /
            self.num_datapoints)

    @property
    def fpr(self):
        if np.sum(self.labels == 0) == 0:
            return 0.0
        return float(np.sum((self.predictions == 1) & (self.labels == 0))) / np.sum(self.labels == 0)

    @property
    def precision(self):
        if np.sum(self.predictions == 1) == 0:
            return 1.0
        return float(np.sum((self.predictions == 1) & (self.labels == 1))) / np.sum(self.predictions == 1)

    @property
    def recall(self):
        if np.sum(self.predictions == 1) == 0:
            return 1.0
        return float(np.sum((self.predictions == 1) & (self.labels == 1))) / np.sum(self.labels == 1)

    @property
    def num_datapoints(self):
        return self.pred_probs.shape[0]

    @property
    def num_categories(self):
        return self.pred_probs.shape[1]
        
    @property
    def predictions(self):
        return np.argmax(self.pred_probs, 1)

    def top_k_predictions(self, k):
        return np.argpartition(self.pred_probs, -k, axis=1)[:, -k:]

    @property
    def confusion_matrix(self):
        cm = ConfusionMatrix(self.num_categories)
        cm.update(self.predictions, self.labels)
        return cm

    def mispredicted_indices(self):
        return np.where(self.predictions != self.labels)[0]

    def correct_indices(self):
        return np.where(self.predictions == self.labels)[0]

    def convert_labels(self, mapping):
        new_num_categories = len(set(mapping.values()))
        new_probs = np.zeros([self.num_datapoints, new_num_categories])
        new_labels = np.zeros(self.num_datapoints)
        for i in range(self.num_datapoints):
            for j in range(self.num_categories):
                new_probs[i,mapping[j]] += self.pred_probs[i,j]
            new_labels[i] = mapping[self.labels[i]]
        return ClassificationResult([new_probs], [new_labels])

    def label_vectors(self):
        return self.pred_probs[:,1], self.labels

    def multiclass_label_vectors(self):
        label_mat = np.zeros(self.pred_probs.shape)
        for i in range(self.num_datapoints):
            label_mat[i, self.labels[i]] = 1

        pred_probs_vec = self.pred_probs.ravel()
        labels_vec = label_mat.ravel()
        return pred_probs_vec, labels_vec

    def precision_recall_curve(self, plot=False, line_width=2, font_size=15, color='b', style='-', label='', marker=None):
        pred_probs_vec, labels_vec = self.label_vectors()
        precision, recall, thresholds = sm.precision_recall_curve(labels_vec, pred_probs_vec)
        if plot:
            plt.plot(recall, precision, linewidth=line_width, color=color, linestyle=style, label=label, marker=marker)
            plt.xlabel('Recall', fontsize=font_size)
            plt.ylabel('Precision', fontsize=font_size)
        return precision, recall, thresholds

    def roc_curve(self, plot=False, line_width=2, font_size=15, color='b', style='-', label=''):
        pred_probs_vec, labels_vec = self.label_vectors()
        fpr, tpr, thresholds = sm.roc_curve(labels_vec, pred_probs_vec)

        if plot:
            plt.plot(fpr, tpr, linewidth=line_width, color=color, linestyle=style, label=label)
            plt.xlabel('FPR', fontsize=font_size)
            plt.ylabel('TPR', fontsize=font_size)
        return fpr, tpr, thresholds

    @property
    def ap_score(self):
        pred_probs_vec, labels_vec = self.label_vectors()
        return sm.average_precision_score(labels_vec, pred_probs_vec)        

    @property
    def auc_score(self):
        pred_probs_vec, labels_vec = self.label_vectors()
        return sm.roc_auc_score(labels_vec, pred_probs_vec)        

    def save(self, filename):
        if not os.path.exists(filename):
            os.mkdir(filename)
        
        pred_filename = os.path.join(filename, 'predictions.npz')
        np.savez_compressed(pred_filename, self.pred_probs)

        labels_filename = os.path.join(filename, 'labels.npz')
        np.savez_compressed(labels_filename, self.labels)

    @staticmethod
    def load(filename):
        if not os.path.exists(filename):
            raise ValueError('File %s does not exists' %(filename))

        pred_filename = os.path.join(filename, 'predictions.npz')
        pred_probs = np.load(pred_filename)['arr_0']

        labels_filename = os.path.join(filename, 'labels.npz')
        labels = np.load(labels_filename)['arr_0']
        return ClassificationResult([pred_probs], [labels])

class RegressionResult(object):
    def __init__(self, predictions_list, labels_list):
        self.predictions = None
        self.labels = None
        
        for predictions, labels in zip(predictions_list, labels_list):
            if self.predictions is None:
                self.predictions = predictions
                self.labels = labels
            else:
                self.predictions = np.r_[self.predictions, predictions]
                self.labels = np.r_[self.labels, labels]

    @property
    def error_rate(self):
        return np.sum((self.predictions - self.labels)**2) / (float(self.num_datapoints  * self.predictions.shape[1]))

    @property
    def num_datapoints(self):
        return self.predictions.shape[0]

    def save(self, filename):
        if not os.path.exists(filename):
            os.mkdir(filename)
        
        pred_filename = os.path.join(filename, 'predictions.npz')
        np.savez_compressed(pred_filename, self.predictions)

        labels_filename = os.path.join(filename, 'labels.npz')
        np.savez_compressed(labels_filename, self.labels)

    @staticmethod
    def load(filename):
        if not os.path.exists(filename):
            raise ValueError('File %s does not exists' %(filename))

        pred_filename = os.path.join(filename, 'predictions.npz')
        predictions = np.load(pred_filename)['arr_0']

        labels_filename = os.path.join(filename, 'labels.npz')
        labels = np.load(labels_filename)['arr_0']
        return RegressionResult([predictions], [labels])
