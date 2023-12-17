import typing as tp

import argparse
import logging

import transformers

from zero_shot_time.generation import NumericalLogitsWarper
from zero_shot_time.data import get_dataset, pre_processing
from zero_shot_time.data.splits import create_train_test_split
from zero_shot_time.generation import set_padding_or_none
from zero_shot_time.generation.numerical_logits_warper import get_token_masks
from zero_shot_time.optimization import perform_hyper_parameter_tuning


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
    'llama7b': {
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

def main_llm(args: argparse.Namespace) -> None:
    """Main function to run LLMTime Reimplementation experiments."""
    model_name = args.model
    dataset_name = args.dataset
    sub_category = args.subcat
    truncation = args.truncate

    # 1.1 Load model (note the causal lm !)
    model: transformers.GPT2Model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

    # 1.2.1 Load and initialize the models' tokenizer and prepare tokenizer for batch-encoding plus.
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("distilgpt2")
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

    # TODO: beter document / configure the usage of different encoding strategies.
    seperator_token_id, parameter_token_id, numerical_token_mask = get_token_masks(
        seperator = ' ,',
        padding = '',
        numerical_encodings = [
            f" {i}" for i in range(0, 10)
        ],
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
        padding_token_id=parameter_token_id,
        search_space=SEARCH_SPACES[model_name]
    )

    best_nll_parameters = study.best_params
    # Step 2, store hyper-parameter for future reference
    #   automatically done by the hyper

    # Step 3, generate samples
    generation_args = {
        'do_sample': True,          # Randomly select
        'top_p': 0.9,               # Limit the probability to top 90%
        'temperature': 0.7,         # Rescale factor for probability estimates
    }
    # Note we don't set the top_k, as we limit the ouput vocabulary using our LogitRescaler.
    warper = NumericalLogitsWarper(
        vocabulary_size=model.vocab_size,
        numerical_token_ids=get_numerical_tokenids(
            [f' {i}' for i in range(10)],
            tokenizer),
        padding_token_id=None,
        seperator_token_id=get_seperator_tokenids(' ,', tokenizer),
        device=model.device
    )

    for train_set, test_set in zip(train_sets, test_sets):
        responses: tp.List[tp.List[tp.Any]] = generate_completions(
            train_sets=train_set,
            test_sets=test_set,
            model=model,
            logits_warper_constraint=warper,

        )
        # TODO: Filter responses to have minimum length during processing.

        # TODO: Can we add streaming into the mix?

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


    model.generate(inputs=input_ids, logits_processor=warper, max_length=4096)

    # TODO: Post-processing
    ...

    # TODO: Write to a file

    ...


    # TODO: Do plotting and evaluation in a new set.


if __name__ == "__main__":
    default_max_history = 400
    default_model = "distilgpt2"
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
