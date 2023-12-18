import typing as tp

import transformers


def set_padding_or_none(tokenizer: transformers.PreTrainedTokenizerFast, set_padding="eos"):
    """Helper function to initialze the tokenization model when not set during initialization."""

    if tokenizer.pad_token_id is None and set_padding:
        tokenizer.pad_token_id = getattr(tokenizer, f"{set_padding}_token_id")
        tokenizer.pad_token = getattr(tokenizer, f"{set_padding}_token")

    return tokenizer


def get_token_ids_for_numerical(tokens: tp.Union[str, tp.List[str]], tokenizer: transformers.PreTrainedTokenizerFast) -> \
tp.Union[int, tp.List[int]]:
    """
    Common helper function to retrieve token id(s) from an input string. Note that it assumes provided numerical/
    separator tokens can be encoded in a single token.

    Args:
        tokens (Union[str, List[str]]):
            The input string or list of strings for which token ids need to be retrieved.
        tokenizer (transformers.PreTrainedTokenizerFast):
            The pre-trained tokenizer object used for encoding the input tokens.

    Returns:
        Union[int, List[int]]:
            If the input is a single string, returns the token id as an integer.
            If the input is a list of strings, returns a list of corresponding token ids.
    """
    if isinstance(tokens, str):
        return tokenizer.encode(tokens)[0]
    elif isinstance(tokens, tp.List):
        if hasattr(tokenizer, 'batch_encode_plus'):
            return [
                token[0]
                for token in tokenizer.batch_encode_plus(tokens)['input_ids']
            ]
        else:
            return [
                get_token_ids_for_numerical(token, tokenizer)
                for token in tokens
            ]
