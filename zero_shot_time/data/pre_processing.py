import logging
import typing as tp

import datasets
import numpy as np
import transformers
from sklearn.preprocessing import MinMaxScaler
from transformers import BatchEncoding


def pre_process_values(values, **scaler_kwargs):
    """Helper function to apply a scaler to a time-series, to map values within a specific range. By default, this
        processing function will apply a `MinMaxScaler` to the data, to map the values to the domain (0, 1).

    Args:
        values (np.array): Array of values from the time series. Make sure that 'future' values from a time-series are
            not passed to the pre-processing function.
        scaler_kwargs (kwargs): Kwargs to pass to the standard scaler in case required.
    Returns:
        MinMaxScaler fitted to the original data that was passed.
        Transformed data according to the fitted scaler.


    """
    logging.warning("Calling the pre_process_values function with a complete timeseries may leak information about the"
                    "future. To prevent leakage of information, only provide the values till the point that you want to"
                    "start generating from.")
    scaler = MinMaxScaler(**scaler_kwargs)
    scaler.fit(values[:, None])

    transformed_values = scaler.transform(values[:, None])

    return scaler, transformed_values.flatten()


def convert_timeseries_to_fixed_precision(dataset: datasets.Dataset, tokenizer: transformers.PreTrainedTokenizerFast, target: str):
    """Helper function to convert a time-series dataset (split) from numerical representation to a fixed precision
    representation.

    Args:
        dataset (Dataset): Huggingface (compatible) dataset of a time series (split).
        target (str): Target (column) name of which to convert to a specific timeseries.

    """
    values = dataset[target]

    pre_processor, pre_processed_values = pre_process_values(np.array(values))
    _, stringified_values = stringify_values(pre_processed_values, precision=4, value_mapper=map_substring_to_token)
    time_series_tokens = tokenize_values(stringified_values, tokenizer)

    return pre_processor, pre_processed_values, time_series_tokens


def map_substring_to_token(value: tp.List[str], sign=None) -> tp.List[str]:
    """Helper function to map a 'stringified' representation of a number to its corresponding 'token string'. E.g. to
    map '0' to ' 0', in case of the default algorithm.

    Args:
        value (tp.List[str]): List of strings with individual value represenetation.
    """
    ret = []
    if sign is not None:
        ret += '+' if sign else '-'
    return ret + [f" {char}" for char in value]


def convert_values(values: tp.Union[np.array], base=10, abs_max=None, precision=2, mapper_function=None):
    """

    Args:
        values: Set/vector of values to map to 'integer' representation (with lagging zeros)
        base: Base to convert base10 values to.
        abs_max: Absolute maximum value to 'clip' values to.
        precision: Numerical precision to adhere to

    Returns:

    """
    signs = values < 0.0
    if np.sum(signs) == 0:
        signs = [None] * len(signs)
    if base != 10:
        raise NotImplementedError("Currently does not support generation of 'rounded' values in arbitrary base.")
    # TODO: Generatize beyond base 10 encoding
    if precision > 0:
        values = values * base ** precision
    representation = np.round(values).astype(np.int32)
    stringified = [mapper_function(list(str(value)), sign=sign) for value, sign in zip(representation, signs)]
    return signs, stringified


def stringify_values(
    values, base=10, abs_max=None, precision=2, value_mapper: tp.Callable[[tp.List[str]], tp.List[str]] = None
):
    """Helper function to convert a series of base10 values into their string counterparts to be used
    Args:
        values:
        base:
        abs_max:
        precision:
        value_mapper: Helper function to map individual characters in a value to the corresponding string representation
            to use from a tokenizer.

    Returns:
        List containg lists of strings, each list representing a value from the provided value list
    """
    sign, converted_values = convert_values(values, base, abs_max, precision, mapper_function=value_mapper)

    # TODO Convert converted values to string representation.
    return sign, converted_values


def tokenize_values(
    values: tp.Union[tp.List[tp.List[str]]], tokenizer: transformers.PreTrainedTokenizerFast, seperator: str = ", "
):
    """Helper function to convert stringified values to input_ids for a (pre-trained) autoregressive LLM. Requires
    the PADDING ID to be set of the tokenizer (e.g. the EOS token of the model, if no PAD ID is set).

    Args:
        values: Stringified representations of sampels
        tokenizer:

    Returns:

    """
    pad_id = tokenizer.pad_token_id
    # Pretend that the list of lists is a batch of sentences
    # Hack to make the tokenizer agree with batch_encodign
    prefix_space_config = tokenizer.add_prefix_space
    tokenizer.add_prefix_space = True

    batch_encoded_values: BatchEncoding = tokenizer.batch_encode_plus(
        [value + [seperator] for value in values],
        padding=True,
        max_length=None,
        is_split_into_words=True,
        return_tensors="pt",
    )

    tokenizer.add_prefix_space = prefix_space_config

    # [num_values, max_encoding_length]
    input_ids = batch_encoded_values["input_ids"]
    # [num_values * E[encoding_length]
    values_representation = input_ids[input_ids != pad_id]

    # One very long tensor with the corresponding input_ids', further handling should be done by the caller.
    return values_representation