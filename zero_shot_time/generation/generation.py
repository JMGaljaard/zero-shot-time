import logging
import typing as tp

import torch
import transformers


@torch.no_grad()
def get_generated_completions(
    train_sets: tp.List[torch.Tensor],
    test_sets: tp.List[torch.Tensor],
    model: transformers.PreTrainedModel,
    logits_warper_constraint: transformers.LogitsWarper,
    generation_config: transformers.GenerationConfig = None,
    completions: int = 20,
    seperator_token_id: int = None,
    attempts=3,
    numerical_token_mask: torch.BoolTensor = None,
    parallel: int = 1,
    to_generate: int = 5,
) -> tp.Tuple[tp.List[tp.List[torch.LongTensor]], tp.List[tp.List[torch.FloatTensor]]]:
    """Generate completions for a train-set dataset based on a given model.

    This function generates completions for a train-set dataset, and the test sets are required to establish the length
    (i.e., the number of additional separators - 1) to be generated.

    Args:
        train_sets (List[torch.Tensor]): List of numpy arrays representing the training datasets.
        test_sets (List[torch.Tensor]): List of numpy arrays representing the test datasets, only used for variable length
            generation checking.
        generation_config (transformers.GenerationConfig): Configuration for decoding/search strategy. Paper proposes
            nucleus sampling in combination with a limited vocabulary.
        model (transformers.PreTrainedModel): Pre-trained transformer model for completion generation.
        logits_warper_constraint (transformers.LogitsWarper): Logits warper constraint for guiding the generation process.
        completions (int, optional): Number of completions to generate. Defaults to 20.
        seperator_token_id (int, optional): Separator ID used to determine the length of generated completions. Defaults to None.
        generate_kwargs (Dict[str, Any], optional): Additional configuration options for generation. Defaults to None.

    Returns:
        List[torch.Tensor]: List of List of arrays representing the number of samples that were generated by sampling the
            LLM with historical (train) data. Note not every list of arrays is of same length if the model was not
            able to generate `completions` within the provided `attempts`.
    """
    predictions = []
    scores = []
    to_gen = completions
    # TODO: Implement parallel decoding :)
    for train_set, test_set in zip(train_sets, test_sets):
        # Make sure that it is completed
        min_samples = torch.sum(test_set == seperator_token_id) + 1
        prediction = []
        score = []
        train_view = train_set.view(1, train_set.size(-1))
        for _ in range(attempts * completions // parallel):
            if len(prediction) >= completions:
                break
            # Create parallel responses
            response = model.generate(
                input_ids=train_view,
                num_return_sequences=min(parallel, to_gen),
                attention_mask=torch.ones_like(train_view),
                generation_config=generation_config,
                logits_processor=transformers.LogitsProcessorList([logits_warper_constraint]),
            )
            local_scores = torch.stack([score[:, ~numerical_token_mask] for score in response.scores]).to("cpu")
            for samp_indx in range(response.sequences.size(0)):
                if (sample := torch.sum(response.sequences[samp_indx] == seperator_token_id)) >= min_samples:
                    prediction.append(response.sequences[samp_indx].detach().to("cpu"))
                    score.append(local_scores[:, samp_indx].detach().to("cpu"))
                else:
                    logging.warning(
                        f"Generated insufficient samples! Required: {min_samples.item()} but got: {sample.item()}!"
                        f"If you see this many times, you may want to increase the context-length"
                    )
            to_gen = completions - len(prediction)
            if to_gen == 0:
                break
        predictions.append(prediction)
        scores.append(score)

    return predictions, scores
