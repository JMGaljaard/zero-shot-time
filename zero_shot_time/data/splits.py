import logging
import warnings

import datasets
import numpy as np


python

def create_validation_split(train_series: np.array, test_length: int) -> (np.array, np.array):
    """
    Constructs a validation time series from the last `T` observations in the training series, where `T` is
    the length of the test series. If the training series is shorter than `2T`, the last half of the
    training series is taken as the validation series.

    Args:
        train_series (np.array): The input training time series.
        test_length (int): The length of the test series.

    Returns:
        np.array: Hyperparameter training split consisting of the first part of the provided training dataset targets.
        np.array: Hyperparameter validation split consisting of the remaining splits.
    """


    if len(train_series) < 2 * test_length:
        half = len(train_series) // 2
        return train_series[:half], train_series[half:]
    else:
        return train_series[:-test_length], train_series[-test_length:]


def create_train_test_split(train_dataset: datasets.Dataset, test_dataset: datasets.Dataset, target: str = 'target',
                            max_length: int = 400) -> ((np.array, np.array), np.array, np.array):
    """Create train/train_val and test splits to use for a datasets.

    Returns:
        List[(np.array, np.array)]: List of tuples with a hyper-parameter optimization train and test list for each
            provided training set.
        List[np.array]: List of training sets limited in maximum length by provided `max_length`.
        List[np.array]: List of testing sets corresponding to each training set.

    """

    if not isinstance(train_dataset[target], list):
        train_dataset = [train_dataset]
        test_dataset = [test_dataset]

    param_sets, train_sets, test_sets = [], [], []

    for train_set, test_set in zip(train_dataset, test_dataset):
        # Limit the maximum length of the dataset to `max_length`
        limit_train_set = train_set[-max_length:]
        if len(test_set) > max_length:
            logging.fatal("Length of test dataset exceeds maximum length also!")

        # Create hyper-parameter split for zero-shot training
        train_h, val_h = create_validation_split(limit_train_set, len(test_set))

        # Append dataset to the required number of sets.
        param_sets.append((train_h, val_h))
        train_sets.append(limit_train_set)
        test_sets.append(test_set)

    return param_sets, train_sets, test_sets
