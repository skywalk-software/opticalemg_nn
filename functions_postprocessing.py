# -*- coding: utf-8 -*-
"""
Created on Tues Mar  8 13:09:52 2022

@author: tyler
"""

# %% Functions to import from this file
# from functions_postprocessing import plot_predictions
# from functions_postprocessing import apply_flag_pulses

# %% Top-Level Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% Local Imports
from functions_general import get_rising_edge_indices
from functions_general import get_falling_edge_indices

# %% FUNCTION - CHECK ACCURACY OF THE MODEL (currently unused, defaulting to apply_flag_pulses)
# TODO needs to output RMS and max onset timing error, RMS and max offset timing error
# TODO needs to output # true positives, # false positives, # false negatives, and # false releases (midway through event)
# TODO needs to do this quickly

# def check_accuracy(model, test_dataset, test_labels_array, sequence_length, correct_tolerance):
#     predictions_test = model.predict(test_dataset)
#     predictions = np.argmax(np.array(predictions_test), axis=1)
#     matched_test_labels_array = test_labels_array[0:predictions.shape[0]]

#     # Get actual edges
#     rising_edge_timestamps, rising_edge_indices = get_rising_edge_indices(matched_test_labels_array)
#     falling_edge_timestamps, falling_edge_indices = get_falling_edge_indices(matched_test_labels_array)

#     correct_array = np.zeros(rising_edge_indices.shape)

#     # Get predicted edges
#     pred_rising_edge_timestamps, pred_rising_edge_indices = get_rising_edge_indices(predictions)
#     pred_falling_edge_timestamps, pred_falling_edge_indices = get_falling_edge_indices(predictions)

#     # Check if any edges would be combined by the correct tolerance
#     if (max(abs(falling_edge_indices[:falling_edge_indices.shape[0]-1] - rising_edge_indices[1:])) <
#             2*correct_tolerance):
#         print("One or more events would be combined in this accuracy metric, defaulting to zero tolerance")
#     # Expand events if it won't break stuff
#     else:
#         for ind in rising_edge_indices:
#             if (ind - correct_tolerance) <= 0:
#                 matched_test_labels_array[0:ind] = matched_test_labels_array[ind]
#             else:
#                 matched_test_labels_array[ind-correct_tolerance:ind] = matched_test_labels_array[ind]

#         for ind in falling_edge_indices:
#             if (ind + correct_tolerance) >= matched_test_labels_array.shape[0]:
#                 matched_test_labels_array[ind:matched_test_labels_array.shape[0]] = matched_test_labels_array[ind]
#             else:
#                 matched_test_labels_array[ind:ind+correct_tolerance] = matched_test_labels_array[ind]

#     # Multiply matched_test_labels_array by predictions to get overlap
#     mult_array = matched_test_labels_array * predictions
#     mult_rising_edge_timestamps, mult_rising_edge_indices = get_rising_edge_indices(mult_array)
#     mult_falling_edge_timestamps, mult_falling_edge_indices = get_falling_edge_indices(mult_array)

#     # Number of correct predictions (need to verify 1:1 correspondence between prediction and actual)
#     num_correct = mult_rising_edge_indices.shape[0]

#     for i in range(rising_edge_indices.shape[0]):
#         for j in range(pred_rising_edge_indices.shape[0]):
#             if pred_rising_edge_indices[j] - rising_edge_indices


#     # for i in range(rising_edge_indices.shape[0]):
#     #     mult_rising_edge_timestamps, mult_rising_edge_indices =
#           get_rising_edge_indices(mult_array[rising_edge_indices[i]:falling_edge_indices[i]])
#     #     mult_falling_edge_timestamps, mult_falling_edge_indices =
#           get_falling_edge_indices(mult_array[rising_edge_indices[i]:falling_edge_indices[i]])
#     #     if (mult_falling_edge_indices.shape[0] >= 1):
#     #         if (mult_rising_edge_indices.shape[0] >=1):
#     #             for ind in mult_falling_edge_indices:
#     #                 if ind <

#     # Add matched_test_labels_array to predictions to check for false releases
#     sum_array = matched_test_labels_array + predictions


#     # metrics

#     # call out if num_pred_rising_edges != num_correct_predictions (e.g. two or more predictions got combined)?
#     # can also get handled in the latency calc actually...

#     # expand actual events

# %% Flag Pulses (apply_flag_pulses)

def apply_flag_pulses(predicted, actual):
    correct_pulses = 0
    false_pos_pulses = 0
    false_neg_pulses = 0
    n = len(predicted)
    pred_on = False
    true_on = False
    caught = False
    caught_vec_pulses = []
    true_on_vec_pulses = []
    pred_on_vec_pulses = []

    for i_pulses in range(n):
        caught_vec_pulses.append(caught)
        true_on_vec_pulses.append(true_on)
        pred_on_vec_pulses.append(pred_on)

        if true_on and pred_on:  # both are high
            if actual[i_pulses] == 0 and predicted[i_pulses] == 0:
                true_on = False
                pred_on = False
                correct_pulses += 1  # increase number correct
                caught = True  # the press has been caught
            elif actual[i_pulses] == 0:
                true_on = False
                correct_pulses += 1  # increase number correct
                caught = True  # the press has been caught
            elif predicted[i_pulses] == 0:
                pred_on = False
                correct_pulses += 1  # increase number correct
                caught = True  # the press has been caught

        elif true_on and not pred_on:
            if actual[i_pulses] == 0:  # see if we missed a pulse (false negative)
                true_on = False
                if not caught:  # we haven't caught the pulse yet
                    false_neg_pulses += 1  # increase false negative if true_on and not pred_on and actual = 0
                    true_on = False
            if predicted[i_pulses] == 1:
                pred_on = True
        elif pred_on and not true_on:
            if predicted[i_pulses] == 0:  # see if we detected a fake pulse (false positive)
                if not caught:
                    false_pos_pulses += 1  # increase false positive if pred_on and not true_on and actual = 0
                    # print("false positive at:", i)
                pred_on = False
            if actual[i_pulses] == 1:
                true_on = True

        else:  # both are low
            caught = False  # waiting for the next pulse, so haven't caught it yet
            if actual[i_pulses] == 1:
                true_on = True
            if predicted[i_pulses] == 1:
                pred_on = True

    return correct_pulses, false_pos_pulses, false_neg_pulses, caught_vec_pulses, true_on_vec_pulses, pred_on_vec_pulses

# %% PLOT PREDICTED VS. ACTUAL DATA / LABELS
# plot_predictions(predictions, test_labels_array, test_data_array)
def plot_predictions(pred, labels_array, data_array):
    pred_rising_edges = get_rising_edge_indices(pred, which_column=None)
    pred_falling_edges = get_falling_edge_indices(pred, which_column=None)
    for i_e in range(len(pred_rising_edges)):
        plt.axvline(pred_rising_edges[i_e], color="red", ymin=.8, ymax=1, alpha=0.1)
        plt.axvspan(pred_rising_edges[i_e], pred_falling_edges[i_e], ymin=.8, ymax=1, facecolor='r', alpha=0.3,
                    label="_" * i_e + "prediction")
        plt.axvline(pred_falling_edges[i_e], color="red", ymin=.8, ymax=1, alpha=0.1)

    real_rising_edges = get_rising_edge_indices(labels_array, which_column=None)
    real_falling_edges = get_falling_edge_indices(labels_array, which_column=None)
    for i_e in range(len(real_rising_edges)):
        plt.axvline(real_rising_edges[i_e], color="blue", ymin=0, ymax=.8, alpha=0.1)
        plt.axvspan(real_rising_edges[i_e], real_falling_edges[i_e], ymin=0, ymax=.8, facecolor='b', alpha=0.3,
                    label="_" * i_e + "actual")
        plt.axvline(real_falling_edges[i_e], color="blue", ymin=0, ymax=.8, alpha=0.1)

    plt.plot(data_array)
    plt.legend()
    return
