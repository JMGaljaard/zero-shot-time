import torch
import transformers
from transformers import LogitsWarper
from transformers.generation.utils import GenerateOutput


def calculate_negative_log_likelihood(
    model: transformers.GPT2Model,
    input_token_ids: torch.LongTensor,
    target_token_ids: torch.LongTensor,
    separator_token_id: int,
    padding_token_id: int,
    logits_warper: LogitsWarper,
    generation_config: transformers.GenerationConfig,
    token_mask: torch.BoolTensor,
    dy_dx: float,
    precision: int = 3,
    base: int = 10,
):
    """
    Calculate the Negative Log-Likelihood (NLL) per dimension using the output logits of a Language Model (LM)
    as conditional probability.

    Args:
        model (transformers.GPT2Model): Language Model to perform predictions with.
        input_token_ids (torch.LongTensor): Tensor containing input tokens (history) for the model.
        target_token_ids (torch.LongTensor): Tensor containing target tokens (ground truth) for evaluation.
        separator_token_id (int): Token ID representing separators in the token vocabulary.
        padding_token_id (int): Token ID representing padding in the token vocabulary.
        logits_warper (LogitsWarper): Object responsible for warping the logits during the calculation.
        generation_config (transformers.GenerationConfig): Configuration for text generation.
        token_mask (torch.BoolTensor): Boolean tensor indicating which tokens should be masked during calculation.
        dy_dx (float): Derivative (intercept) of scaling function, this can be calculated with JAX.
        precision (int, optional): Precision of the calculation. Defaults to 3.
        base (int, optional): Base for logarithmic calculations. Defaults to 10.

    Returns:
        float: Calculated Negative Log-Likelihood (NLL) per dimension.
    """

    assert torch.all(input_token_ids[:,-1] == separator_token_id), 'Provided input requires to end with seperator' \
                                                                   ' before concatenating'

    full_series = torch.concat([input_token_ids, target_token_ids], dim=1)
    # TODO: Ensure that hyper-parameters are set in a logits warper list.

    # Use generate function to return
    response: GenerateOutput = model.generate(
        input_token_ids=full_series,
        max_new_tokens=0,
        scores=True,
        generation_config=generation_config
    )

    logits = response.scores
    # Set logits to -100 (ignore) for dissallowed tokens
    logits[token_mask] = -100.0

    # get log probabilties over output dimension, skip the EOS token
    logp_num_tokens = torch.log_softmax(logits, dim=-1)[0, :-1, token_mask]

    # Get sequence length
    history_len = input_token_ids.size(1)

    # Take only the future / predictions tokens into account.
    logp_future_tokens = logp_num_tokens[history_len:]

    base = - logp_future_tokens.sum() / len(target_token_ids)
    bin_offset = - precision * torch.log(torch.tensor(base, device='cuda'))

    # Calculate component to map back to feature space.
    scaling_offset = - torch.log(torch.tensor(dy_dx))

    # Add components  (they are already negative)
    nll_res = base + bin_offset + scaling_offset

    return nll_res
