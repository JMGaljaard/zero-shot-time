import typing as tp

import numpy as np
import torch
import transformers


def base_transformation(base=10, precision=3, seperator=" "):
    """Helper function with partial application (currying) to transform an encoded value to a floating point value.

    Args:
         base (int): Base in which the number is encoded, bases other than 10 are currently not supported.
         precision (int): Precision of the number to re-construct, i.e. maximum token lenght of an encoded number.
         seperator (str): 'Bit' seperator used in the encoding scheme, defaults to ' ' as described in original work.

    Returns:
        floating point number representation of the encoded value.

    """
    local_precision = precision
    local_base = float(base)
    local_seperator = seperator

    if base != 10:
        raise NotImplementedError("Support for bases other than 10 is not yet implemented!")

    def curried_transform(value: str):
        # Reverse the order from MSB to LSB
        if len(seperator) > 0:
            value_list = value.split(local_seperator)
        else:
            # Note that this will only work if value components can be represented in single digits/characters.
            value_list = list(value)
        value = np.flip(np.array(value_list, dtype=np.int32))
        D = len(value)
        # Generate a list of powers of 10 with [10^-1, 10^-2, ..., 10^-precision+1, 10^-precision]
        powers = np.arange(-local_precision, -local_precision + D)
        # Calculateteh
        val = np.sum((value * local_base**powers)[:local_precision])
        return val

    return curried_transform


def convert_tokens_to_timeseries(
    token_ids: torch.LongTensor,
    tokenizer: transformers.PreTrainedTokenizerFast,
    mapping_function: tp.Callable = None,
    seperator: str = " ,",
    dtype=np.float64,
) -> np.array:
    """Function to transfrom encoded text back to  tokenized data back to floating points

    Args:
        token_ids: Predicted token_ids by the language model


    """

    # 1. Convert tokens to text

    texts = tokenizer.decode(token_ids,
                 skip_special_tokens=True,
                 clean_up_tokenization_spaces=False
                 )

    # 2. Split values to individual samples
    # For specific edge case, we need to strip delimiter
    string_values = [text.strip(seperator) for text in texts.split(seperator)]


    # 3. Convert values to values
    # TODO: Properly handle invalid values
    values = [mapping_function(val) for value in string_values if (val := value.strip()) != ""]

    return np.array(values, dtype=dtype)
