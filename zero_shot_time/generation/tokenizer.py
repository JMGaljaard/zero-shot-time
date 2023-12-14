import transformers


def set_padding_or_none(tokenizer: transformers.PreTrainedTokenizerFast, set_padding="eos"):
    """Helper function to initialze the tokenization model when not set during initialization."""

    if tokenizer.pad_token_id is None and set_padding:
        tokenizer.pad_token_id = getattr(tokenizer, f"{set_padding}_token_id")
        tokenizer.pad_token = getattr(tokenizer, f"{set_padding}_token")

    return tokenizer
