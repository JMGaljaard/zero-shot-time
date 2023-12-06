
import typing as tp

import numpy as np
import transformers
from transformers import BatchEncoding


def convert_values(values: tp.Union[np.array], base=10, abs_max=None, precision = 2):
    """

    Args:
        values: Set/vector of values to map to 'integer' representation (with lagging zeros)
        base: Base to convert base10 values to.
        abs_max: Absolute maximum value to 'clip' values to.
        precision: Numerical precision to adhere to

    Returns:

    """
    sign = values < 0.0
    if base != 0:
        raise NotImplementedError("Currently does not support generation of 'rounded' values in arbitrary base.")
    # TODO: Generatize beyond base 10 encoding
    representation = np.round(values * 10** precision).astype(np.int32)
    raise NotImplementedError("Not yet implement")

    return sign, representation

def stringify_values(values, base, abs_max, precision=2):
    """Helper function to convert a series of base10 values into their string counterparts to be used
    Args:
        values:
        base:
        abs_max:
        precision:

    Returns:

    """
    sign, converted_values = convert_values(values ,base, abs_max, precision)

    # TODO Convert converted values to string representation.

    raise NotImplementedError("Not yet implemented")

def tokenize_values(values: tp.Union[tp.List[tp.List[str]]], tokenizer: transformers.PreTrainedTokenizerFast, seperator: str = ', '):
    """Helper function to convert stringified values to input_ids for a (pre-trained) autoregressive LLM. Requires
    the PADDING ID to be set of the tokenizer (e.g. the EOS token of the model, if no PAD ID is set).

    Args:
        values: Stringified representations of smapels
        tokenizer:

    Returns:

    """
    pad_id = tokenizer.pad_token_id
    # Pretend that the list of lists is a batch of sentences
    batch_encoded_values: BatchEncoding = tokenizer.batch_encode_plus(
            [
                value + [seperator] for value in values
            ],
            padding=True,
            max_length=None,
            is_split_into_words=True,
            return_tensors='pt'
    )

    # [num_values, max_encoding_length]
    input_ids = batch_encoded_values['input_ids']
    # [num_values * E[encoding_length]
    values_representation = input_ids[input_ids != pad_id]

    # One very long tensor with the corresponding input_ids', further handling should be done by the caller.
    return values_representation
