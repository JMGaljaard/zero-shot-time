import torch
import transformers
from transformers.generation.utils import GenerateOutput


def calculate_negative_log_likelihood(
    model: transformers.PreTrainedModel,
    input_token_ids: torch.LongTensor,
    target_token_ids: torch.LongTensor,
    separator_token_id: int,
    padding_token_id: int,
    token_mask: torch.BoolTensor,
    dy_dx: float,
    precision: int = 3,
    base: int = 10,
    temperature: float = 0.7,
    pre_computed_logits=None,
    offset=0,
    prediction_len=8,
):
    """
    Calculate the Negative Log-Likelihood (NLL) per dimension using the output logits of a Language Model (LM)
    as conditional probability.

    This function curries the application of the acutal objective, so that hyper-parameters can be 'provided' outside
    of calling the optimization function!

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
        temperature (float): Temperature to scale distribution of logits (hyper-parameter).
    Returns:
        float: Calculated Negative Log-Likelihood (NLL) per dimension.
        torch.FloatTensor containing pre-scaling logits.
    """

    assert torch.all(input_token_ids[-1] == separator_token_id), (
        "Provided input requires to end with seperator" " before concatenating"
    )
    assert torch.all(target_token_ids[[0, -1]] != separator_token_id), {
        "Provided targets should not start with seperator! Check your encoding schema!"
    }
    # Note: train_input_ids and target_input_ids share the same start, but the targets can be seen as the through thruth
    # completion of the prompt!
    full_series = target_token_ids
    # TODO: Ensure that hyper-parameters are set in a logits warper list.

    # Use generatino function to compute logits scores on input 'prompt'. leverage max_new_tokens=0 to effectively
    #   perform a forward with the underlying model.
    with torch.no_grad():
        if not isinstance(pre_computed_logits, torch.Tensor):
            # Grads shall not pass!
            response: GenerateOutput = model.forward(
                full_series.unsqueeze(0),  # Get whole series, create 'virtual batch'
            )
            # TODO: Check wether we have to re-scale by the temperature.
            logits = response.logits / temperature
        else:
            logits = pre_computed_logits / temperature
        # Set logits to -100 (ignore) for dissallowed tokens
        logits[0, token_mask[None, :].repeat(logits.size(1), 1)] = -100
        # get log probabilties over output dimension, skip the EOS token, get only the required values
        logp_num_tokens = torch.log_softmax(logits, dim=-1)[0, torch.arange(len(full_series)), full_series]

        # Get sequence length, keep in mind that some models have some starting tokens (e.g. LLama2)
        history_len = len(input_token_ids)

        # Take only the future / predictions tokens into account.
        logp_future_tokens = logp_num_tokens[history_len:]

        base_nll = -logp_future_tokens.sum() / prediction_len

        # Average is equal to the product itself
        bin_offset = -precision * torch.log(torch.tensor(base, device=model.device, dtype=torch.float32))

        # Calculate component to map back to feature space.
        scaling_offset = -torch.log(torch.tensor(dy_dx, device=model.device, dtype=torch.float32))

        # Add components  (they are already negative)
        nll_res = base_nll + bin_offset + scaling_offset
        if not isinstance(pre_computed_logits, torch.Tensor):
            return response.logits, nll_res
        else:
            return pre_computed_logits, nll_res
