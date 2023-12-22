import typing as tp
from functools import partial
from itertools import chain

import numpy as np
import torch
import transformers
from sklearn.base import TransformerMixin
from transformers import BatchEncoding

import datasets

from zero_shot_time.data.scaler import Scaler, get_scaler


def pre_process_llm_values(values, **scaler_kwargs):
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
    scaler: Scaler = get_scaler(values, **scaler_kwargs)

    transformed_values = scaler.transform(values)

    return scaler, transformed_values.flatten()


def convert_timeseries_to_fixed_precision(
    dataset: datasets.Dataset,
        tokenizer: transformers.PreTrainedTokenizerFast,
        values: tp.Optional[np.array] = None,
        target: str = 'target',
        precision: int = 5,
        pre_processor: tp.Optional[tp.Callable] = None,
        seperator: str = ' ,',
        form: str = ' {}',
        **pre_processor_kwargs
) -> (TransformerMixin, np.array, torch.LongTensor):
    """Helper function to convert a time-series dataset (split) from numerical representation to a fixed precision
    representation.

    Args:
        dataset (Dataset): Huggingface (compatible) dataset of a time series (split).
        target (str): Target (column) name of which to convert to a specific timeseries.
        precision (int): Number of precision tokens to use.
        pre_processor (tp.Optional[tp.Callable]): Optional pre-processor in case a pre-fitted scaler is to be used.
    """
    if values is None:
        values = dataset[target]

    if pre_processor is None:
        pre_processor, pre_processed_values = pre_process_llm_values(np.array(values), **pre_processor_kwargs)
    else:
        pre_processor, pre_processed_values = pre_processor, pre_processor.transform(np.array(values))
    _, stringified_values = stringify_values(
        pre_processed_values, precision=precision,value_mapper=partial(map_substring_to_tokens, form=form)
    )
    time_series_tokens = tokenize_values(stringified_values, tokenizer,  seperator=seperator)

    return pre_processor, pre_processed_values, time_series_tokens


def map_substring_to_tokens(value: tp.List[str], sign=None, form = ' {}') -> tp.List[str]:
    """Helper function to map a 'stringified' representation of a number to its corresponding 'token string'. E.g. to
    map '0' to ' 0', in case of the default algorithm.

    Args:
        value (tp.List[str]): List of strings with individual value represenetation.
    """
    ret = []
    if sign is not None:
        ret += "+" if sign else "-"
    # TODO: Link this back to configurable
    return ret + [form.format(char) for char in value]


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
        values = values * base**precision
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
    values: tp.Union[tp.List[tp.List[str]]], tokenizer: transformers.PreTrainedTokenizerFast, seperator: str = ' ,'
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
    if hasattr(tokenizer, 'add_prefix_space'):
        # only for gpt2 model
        prefix_space_config = tokenizer.add_prefix_space
        tokenizer.add_prefix_space = True

    if 'llama' in tokenizer.__class__.__name__.lower():
        # Hacky solution to abuse tokenizer to represent values in the way we want
        # batch_encoded_values: tp.List[tp.List[int]] = [
        #    tokenizer.convert_tokens_to_ids(
        #         value + [seperator],
        #     )    for value in values]
        # # To get attention_mask
        # encoded_batch: BatchEncoding = tokenizer.pad(
        #         BatchEncoding({'input_ids': torch.tensor(list(chain(*batch_encoded_values)), dtype=torch.long)}))
        # Note that LLama(2) requires `<s> ` to be part of the prompt :)
        encoded_batch = tokenizer.batch_encode_plus(
                [''.join([
                    ''.join(value + [seperator]) for value in (values)
                ])],
                return_tensors='pt'
        )
    else:
        encoded_batch: BatchEncoding = tokenizer.batch_encode_plus(
                [value + [seperator] for value in values],
                padding=True,
                max_length=None,
                is_split_into_words=True,
                add_special_tokens=False,
                return_tensors="pt",
        )

    if hasattr(tokenizer, 'add_prefix_space'):
        # Only for gpt2 model
        tokenizer.add_prefix_space = prefix_space_config

    # [num_values, max_encoding_length]
    input_ids = encoded_batch["input_ids"]
    # [num_values * E[encoding_length]
    values_representation = input_ids[input_ids != pad_id]

    # One very long tensor with the corresponding input_ids', further handling should be done by the caller.
    return values_representation


def limit_token_input_length(input_ids: torch.LongTensor, max_samples: int, delimiter_id: int):
    """Helper function to limit the length of input base on a maximum number of samples. Note this function assumes that
    provided data is always a batch of the same data, i.e. no different time-series, or different time-series lenghts
    are provided!

    Args:
        input_ids (torch.LongTensor): Tensor containing the input identifiers of encoded input.
        max_samples (int): Maximum number of samples to select from an encoded list.

    Returns:
        torch.LongTensor of input ids containig the `min(len(input_ids[0], max_samples))` last samples of a time-series,
        using the encoded/input ids from a generative's models output.
    """
    indices = (input_ids[0] == delimiter_id).nonzero(as_tuple=True)
    if indices.size(-1) < max_samples:
        return input_ids

    else:
        return input_ids[:, indices[-max_samples] :]


def limit_series_length(values: np.array, max_history_length: int) -> np.array:
    """Helper function to limit the series length.

    Args:
        values (np.array): Numpy array containing the original or transformed values before tokenization.
        max_samples (int): Maximum number of samples to select from an encoded list.

    Returns:
        torch.LongTensor of input ids containig the `min(len(input_ids[0], max_samples))` last samples of a time-series,
        using the encoded/input ids from a generative's models output.

    """
    # In case max_history_length exceeds the passed list length, the original list will be returned.
    return values[-max_history_length:]
