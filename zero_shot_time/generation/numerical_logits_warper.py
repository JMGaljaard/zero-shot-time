import torch
import transformers
from transformers import LogitsWarper, add_start_docstrings
import typing as tp

from transformers.generation.logits_process import LOGITS_PROCESSOR_INPUTS_DOCSTRING


class NumericalLogitsWarper(LogitsWarper):
    """Simple logits processor implementation required for generation with a 'token mask', i.e. to limit the model to
    predict to generate with only numerical, padding, and seperator tokens.

    This will set all `scores` of dissallowed tokens to `float(-inf)`, making them impossible to generate after a
    softmax.

    Args:
        vocabulary_size (`int`):
            Total vocabulary size of the model which tokens are to be warped during generation.
        numerical_token_ids (`Union[torch.LongTensor, tp.List[int]]`):
            The token ids of allowed numerical tokens.
        padding_token_id (`int`, default=None):
            Optional padding token id, incase padding (before or after) is required by the strategy.
        seperator_token_id (`int`, default=None):
            Optional seperator token id, in case the strategy requireds values to be seperated by an additional seperator.


    """

    def __init__(
        self,
        vocabulary_size: int,
        numerical_token_ids: tp.Union[torch.LongTensor, tp.List[int]],
        padding_token_id: tp.Optional[int],
        seperator_token_id: tp.Optional[int],
        device="cuda",
    ):
        assert isinstance(vocabulary_size, int), "Vocabulary cardinality needs to be int"
        assert isinstance(numerical_token_ids, list), (
            "Numerical tokens are to be provided in a tensor or " "List."
        )
        assert len(numerical_token_ids) > 0, "Numerical generation requires a non-zero number of token ids"

        self.device = device
        self.mask = torch.ones((vocabulary_size), device=device, dtype=torch.bool)
        self.numerical_token_ids = torch.tensor(numerical_token_ids, dtype=torch.int32)
        self.padding_token_id = padding_token_id
        self.seperator_token_id = seperator_token_id

        self.init_mask()

    def init_mask(self):
        """Initialization function to be called during construction, will initialize the token mask to be used during
        generation.
        """
        self.mask[self.numerical_token_ids] = False

        if self.padding_token_id is not None:
            self.mask[self.padding_token_id] = False

        if self.seperator_token_id is not None:
            self.mask[self.seperator_token_id] = False

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # inputs [batch_size, vocabulary]
        # scores [batch_size, vocabulary]

        scores[:, self.mask] = -float("inf")

        return scores


def get_token_masks(seperator: str, padding: str, numerical_encodings: tp.List[str], tokenizer: transformers.GPT2TokenizerFast):
    """ Helper function to construct a mask to mask out part of the predictions' logits output. Note that
    seperators and paddings will not be in the set of ;allowable tokesn;, i.e., they will be assigned a `1` value
    in generated mask.
    Args:
        seperator ():
        padding ():
        numerical_encodings ():
        tokenizer ():

    Returns:

    """
    [seperator_token_id] = [None] if seperator is None else tokenizer.encode(
        seperator,
        add_special_tokens=False
    )
    [padding_token_id] = [None] if len(padding) == 0 else tokenizer.encode(
        padding,
        add_special_tokens=False
    )

    allowable_token_ids = tokenizer.batch_encode_plus(
        numerical_encodings,
        return_tensors='pt',
        add_special_tokens=False
    )['input_ids'].flatten()

    allowable_mask = torch.ones(
        (tokenizer.vocab_size),
        dtype=torch.bool
    )
    allowable_mask[allowable_token_ids] = False

    return seperator_token_id, padding_token_id, allowable_mask


class EarlyTerminationWarper(LogitsWarper):

    def __init__(
        self,
        seperator_token_id: int,
        max_sep_count: int,
        device="cuda",
    ):


        self.device = device
        self.seperator_token_id = seperator_token_id
        self.required_count = None

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        pass
        # inputs [batch_size, vocabulary]
        # scores [batch_size, vocabulary]
        # [bs, lenght] =
