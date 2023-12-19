import os
import pickle
import typing as tp

import argparse
import logging

import numpy as np
import torch
import tqdm
import transformers
from transformers import GenerationConfig

from zero_shot_time.data.post_processing import convert_tokens_to_timeseries, base_transformation
from zero_shot_time.data.scaler import Scaler
from zero_shot_time.generation import NumericalLogitsWarper
from zero_shot_time.data import get_dataset, pre_processing
from zero_shot_time.data.splits import create_train_test_split
from zero_shot_time.generation import set_padding_or_none
from zero_shot_time.generation.generation import get_generated_completions
from zero_shot_time.generation.numerical_logits_warper import get_token_masks
from zero_shot_time.generation.tokenizer import get_token_ids_for_numerical
from zero_shot_time.optimization import perform_hyper_parameter_tuning
from zero_shot_time.optimization.hyper_optimization import process_sets


def main_baseline(args: argparse.Namespace):
    # TODO: Implement baserunning:
    pass


# Define search spaces for different models
SEARCH_SPACES = {
    'distilgpt2': {
        'precision': [2, 3],
        'tau': [0.7],
        'alpha': [0.5, 0.7, 0.9, 0.99],
        'beta': [0.0, 0.14, 0.3, 0.5]
    },
    'gpt2': {
        'precision': [2, 3],
        'tau': [0.7],
        'alpha': [0.5, 0.7, 0.9, 0.99],
        'beta': [0.0, 0.14, 0.3, 0.5]
    },
    'meta-llama/Llama-2-7b-hf': {
        'precision': [3],
        'nucleus': [0.9],
        'tau': [0.2, 0.4, 0.6, 0.8],
        'alpha': [0.99],
        'beta': [0.3]
    },
    'llama13b': {
        'precision': [3],
        'nucleus': [0.9],
        'tau': [0.2, 0.4, 0.6, 0.8],
        'alpha': [0.99],
        'beta': [0.3]
    }
}


def post_process_responses(
        predictions: tp.List[tp.List[torch.LongTensor]],
        scores: tp.List[tp.List[torch.FloatTensor]],
        scaler: Scaler,
        tokenizer: transformers.PreTrainedTokenizerFast,
        mapping_function: tp.Callable,
        seperator_id: int,
        seperator: str,
        train_len: int,
        test_len: int) -> tp.Tuple[tp.List[tp.List[np.array]], tp.List[tp.List[np.array]]]:

    """
    Perform post-processing on model predictions and scores.

    Args:
        predictions (List[List[torch.LongTensor]]): List of lists containing model predictions.
        scores (List[List[torch.FloatTensor]]): List of lists containing model scores.
        scaler (Scaler): Scaler function to perform the inverse transformation.
        test_set_length (int): Length of the test set.

    Returns:
        List[List[np.array]]: List of lists containing post-processed responses.

    Example:
        predictions, scores = get_generated_completions(
            train_sets=train_sets_tokens,
            test_sets=test_sets_tokens,
            model=model,
            logits_warper_constraint=warper,
            generation_config=generation_config,
            separator_token_id=sep_token_id,
            completions=20,
        )
        results = post_process_responses(predictions, scores, scaler, len(test_set))
    """
    ret_pred = []
    ret_score = []
    for sample_predictions, sample_scores in zip(predictions, scores):
        # prediction [1, pred_len]
        # scores [1, pred_len, vocabulary]
        sample_pred_res = []
        sample_score_res = []

        for prediction, score in zip(sample_predictions, sample_scores):
            # Get input_ids and sc

            # Create flat tensor
            input_ids = prediction
            # Drop batch dimension
            score = score.squeeze(0)
            # Get sub-tokens index using seperator to superfluous index (
            indices = (input_ids == seperator_id).nonzero().squeeze()
            # Require n - 1 seperators to compute
            if len(indices) >= (test_len - 1):
                temp = test_len - 1
                if temp >= len(indices):
                    index = -1
                else:
                    index = indices[test_len - 1]
            else:
                # Otherwise there are no super-flo=uous tokens, i.e. just long enough
                index = -1
            # Drop additional predictions and scores
            input_ids = input_ids[:index]
            # TODO: Only get relevant scores?
            score = score[:, :index, ]

            # TODO: Shouldn't we do this in the generation part?
            scaled_representation = convert_tokens_to_timeseries(input_ids, tokenizer, mapping_function=mapping_function, seperator=seperator)
            values = scaler.inverse_transform(scaled_representation)

            sample_pred_res.append(values)
            sample_score_res.append(score.numpy())
        ret_pred.append(sample_pred_res)
        ret_score.append(sample_score_res)
    return ret_pred, ret_score


def main_llm(args: argparse.Namespace) -> None:
    """Main function to run LLMTime Reimplementation experiments."""
    model_name = args.model
    dataset_name = args.dataset
    sub_category = args.subcat
    truncation = args.truncate

    # 1.1 Load model (note the causal lm !)


    if 'gpt' in model_name:
        model: transformers.GPT2Model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name)
        model.to('cuda')
        # 1.2.1 Load and initialize the models' tokenizer and prepare tokenizer for batch-encoding plus.
        tokenizer = transformers.GPT2TokenizerFast.from_pretrained(model_name, token=os.getenv('HF_TOKEN'))
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name, device_map='auto', torch_dtype=torch.float16)
        tokenizer = transformers.LlamaTokenizerFast.from_pretrained(model_name, token=os.getenv('HF_TOKEN'))

    # 1.2.2 In case the fast-tokenizer has no padding, set the padding manually
    set_padding_or_none(tokenizer, set_padding="eos")

    # 2. Load dataset
    dataset, target = get_dataset(dataset_name=dataset_name, sub_category=sub_category)

    # Create train, validation, test split
    param_sets, train_sets, test_sets = create_train_test_split(dataset["train"], dataset["test"], target=target)

    # TODO: Create train / test split according to paper
    if truncation == "before":
        logging.warning("Truncating before pre-processing will affect the 'range' of the scaled values!")
        # TODO: Implement maximum length split.

    seperator = ' ,' if 'gpt' in model_name else ','
    number_repr = [f' {i}' for i in range(10)] if 'gpt' in model_name else [f'{i}' for i in range(10)]
    form = ' {}' if 'gpt' in model_name else '{}'
    # TODO: beter document / configure the usage of different encoding strategies.
    seperator_token_id, parameter_token_id, numerical_token_mask = get_token_masks(
        seperator = seperator,
        padding = '',
        numerical_encodings = number_repr,
        tokenizer=tokenizer
    )

    # Step 1, hyper-parameter search to get reasonable hyper-parameters
    study = perform_hyper_parameter_tuning(
        dataset_name=f"{dataset_name}_{sub_category}",
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        data_sets=param_sets,
        allowable_token_mask=numerical_token_mask,
        seperator_token_id=seperator_token_id,
        seperator=seperator,
        padding_token_id=parameter_token_id,
        search_space=SEARCH_SPACES[model_name],
        form=form,
    )

    best_nll_parameters = study.best_params
    # Step 2, store hyper-parameter for future reference
    #   automatically done by the hyper
    mapping_function =  base_transformation(precision=best_nll_parameters['precision'], seperator=' ' if 'gpt' in model_name else '')
    # Step 3, add constraints to generation
    generation_config = GenerationConfig(
        max_new_tokens=80,                     # Limit output (default to 600 for testing)
        do_sample=True,                         # Randomly select
        top_k=10 + 1,                           # Only consider numerical and comma TODO: Add seperator
        eos_token_id=model.config.eos_token_id, # EOS token (note that model is not allowed to generate this token :>
        # top_p=0.97,                             # Limit to output probability of 97%
        return_dict_in_generate=True,           # Ensure that we can retrieve scores in deocidng results
        output_scores=True,                      # Ensure that we don;t discard scored after decoding
        temperature=best_nll_parameters['tau'],
        renormalize_logits=True,
    )

    num_token_ids = get_token_ids_for_numerical(
            number_repr,
            tokenizer)
    sep_token_id = get_token_ids_for_numerical(seperator, tokenizer)
    # Note we don't set the top_k, as we limit the ouput vocabulary using our LogitRescaler.
    # TODO: Create configuration object to reduce number of parameters everywhere
    warper = NumericalLogitsWarper(
        vocabulary_size=model.config.vocab_size,
        numerical_token_ids=num_token_ids,
        padding_token_id=None,
        seperator_token_id=sep_token_id,
        device=model.device
    )
    processed_results = []
    # Prepare generation configuration, to set sampling on, and other samplign techniques
    for train_set, test_set in tqdm.tqdm(zip(train_sets, test_sets), total=len(train_sets)):
        torch.cuda.empty_cache()
        train_sets_tokens, test_sets_tokens, [scaler] = process_sets(
            [(train_set, test_set)],
            precision=best_nll_parameters['precision'],
            tokenizer=tokenizer,  #
            quantile=best_nll_parameters['alpha'],
            beta=best_nll_parameters['beta'],
            seperator=seperator,
            form=form,
        )
        # TODO: Determine whether or not the model should be restricted to generate with
        # Generate multiple responses for each item
        predictions: tp.List[tp.List[torch.LongTensor]]
        scores: tp.List[tp.List[torch.FloatTensor]]
        # TODO: Figure out LLama2 detokenization.
        predictions, scores = get_generated_completions(
            train_sets=train_sets_tokens,
            test_sets=test_sets_tokens,
            model=model,
            logits_warper_constraint=warper,
            generation_config=generation_config,
            seperator_token_id=sep_token_id,
            numerical_token_mask=numerical_token_mask,
            completions=20,
            parallel=10
        )


        results: tp.List[tp.List[np.array]] = post_process_responses(
                predictions=predictions,
                scores=scores,
                scaler=scaler,
                tokenizer=tokenizer,
                mapping_function=mapping_function,
                train_len=len(train_set),
                test_len=len(test_set),
                seperator_id=seperator_token_id,
                seperator=seperator
        )

        processed_results.append(results)
        # TODO: Filter responses to have minimum length during processing.

        # TODO: Can we add streaming into the mix?
    with open('data.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(processed_results, f, pickle.HIGHEST_PROTOCOL)

    # Step 4, generated samples,
    # 3. Pre-process data, and get mapping function to re-construct
    # Note, that the scaler that is returned is scaled on the entire time-series
    scaler, process_values, input_ids = pre_processing.convert_timeseries_to_fixed_precision(
        dataset, tokenizer, target=target
    )

    if truncation == "after":
        raise NotImplementedError("Input truncation based on tokenization length is not yet supported!")
        logging.warning("Truncating will leak information of previous states")
    print(input_ids)



    # TODO: Post-processing
    ...

    # TODO: Write to a file

    ...


    # TODO: Do plotting and evaluation in a new set.


if __name__ == "__main__":
    default_max_history = 400
    default_model = "meta-llama/Llama-2-7b-hf"
    # default_model = 'gpt2'
    default_dataset = "hpc"
    default_subcat = None
    default_truncate = "before"
    # Create argparse parser
    parser = argparse.ArgumentParser(description="Description of your experiment with Zero-Shot-Time")
    subparsers = parser.add_subparsers(dest="action", help="Available actions to run")

    baseline_parser = subparsers.add_parser("baseline", description="Run baseline experiment")

    baseline_parser.add_argument(
        "--model",
        type=str,
        choices=["arima", "nbeats", "nhits"],
        default=default_model,
        help=f"Model name (default: {default_model})",
    )

    llmtime_parser = subparsers.add_parser("llmtime", description="Run LLMTime experiment")

    # Add command line arguments
    llmtime_parser.add_argument(
        "--max_history",
        type=int,
        default=default_max_history,
        help=f"Maximum history length (default: {default_max_history})",
    )
    llmtime_parser.add_argument(
        "--model", type=str, default=default_model, help=f"Model name (default: {default_model})"
    )
    llmtime_parser.add_argument(
        "--dataset", type=str, default=default_dataset, help=f"Dataset name (default: {default_dataset})"
    )

    llmtime_parser.add_argument(
        "--subcat",
        type=str,
        default=default_subcat,
        required=False,
        help="Dataset subcategory (needed for HF dataset) (default: None)",
    )
    llmtime_parser.add_argument(
        "--truncate",
        type=str,
        choices=["before", "after"],
        default=default_truncate,
        help=f"Truncate time series before or after tokenization (default: {default_truncate})",
    )

    # Parse command line arguments
    args = parser.parse_args()

    if args.action == "baseline":
        # Run baseline code
        main_baseline(args)
    if args.action == "llmtime":
        main_llm(args)
