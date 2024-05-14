#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 00:41:55 2021

@author: shariba
"""

import numpy as np
from sklearn import metrics
import torch 
from scipy.ndimage import distance_transform_edt

def _assert_valid_lists(groundtruth_list, predicted_list):
    assert len(groundtruth_list) == len(predicted_list)
    for unique_element in np.unique(groundtruth_list).tolist():
        assert unique_element in [0, 1]

def _all_class_1_predicted_as_class_1(groundtruth_list, predicted_list):
    _assert_valid_lists(groundtruth_list, predicted_list)
    return np.unique(groundtruth_list).tolist() == np.unique(predicted_list).tolist() == [1]

def _all_class_0_predicted_as_class_0(groundtruth_list, predicted_list):
    _assert_valid_lists(groundtruth_list, predicted_list)
    return np.unique(groundtruth_list).tolist() == np.unique(predicted_list).tolist() == [0]

def get_confusion_matrix_elements(groundtruth_list, predicted_list):
    """returns confusion matrix elements i.e TN, FP, FN, TP as floats
	See example code for helper function definitions
    """
    _assert_valid_lists(groundtruth_list, predicted_list)
    if _all_class_1_predicted_as_class_1(groundtruth_list, predicted_list) is True:
        tn, fp, fn, tp = 0, 0, 0, np.float64(len(groundtruth_list))
    elif _all_class_0_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        tn, fp, fn, tp = np.float64(len(groundtruth_list)), 0, 0, 0
    else:
        tn, fp, fn, tp = metrics.confusion_matrix(groundtruth_list, predicted_list).ravel()
        tn, fp, fn, tp = np.float64(tn), np.float64(fp), np.float64(fn), np.float64(tp)
    return tn, fp, fn, tp

def get_confusion_matrix_torch(y_true, y_pred):
    N = max(max(y_true), max(y_pred)) + 1
    y_true = torch.tensor(y_true, dtype=torch.long)
    y_pred = torch.tensor(y_pred, dtype=torch.long)
    y = N * y_true + y_pred
    y = torch.bincount(y)
    if len(y) < N * N:
        y = torch.cat(y, torch.zeros(N * N - len(y), dtype=torch.long))
    y = y.reshape(N, N)
    return y

def precision(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)
    
def recall(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)
    
def F2(y_true, y_pred, beta=2):
    
    p = precision(y_true,y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15) 


def PPV(y_true,y_pred):
    # TP/(TP + FP)
    TP = (y_true * y_pred).sum()
    FP = np.sum(y_true[y_pred>0]==0)
    
    return TP / float(TP+FP+1e-15)

def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def hausdorff_distance(y_true, y_pred):
    ref_distances = distance_transform_edt(np.logical_not(y_true).astype(int))
    pred_distances = distance_transform_edt(np.logical_not(y_pred).astype(int))
    hausdorff_dist_ref_to_pred = np.max(ref_distances * y_pred)
    hausdorff_dist_pred_to_ref = np.max(pred_distances * y_true)
    hausdorff_distance = max(hausdorff_dist_ref_to_pred, hausdorff_dist_pred_to_ref)
    return hausdorff_distance