import argparse
import logging
import typing as tp

import transformers

from zero_shot_time.data import get_dataset, pre_processing
from zero_shot_time.generation import set_padding_or_none


def main(
    model: str = "distilpgt2",
    dataset: str = "hpc",
    sub_category: tp.Optional[str] = None,
    max_history: int = 400,
    truncation: str = "before",
) -> None:
    # 1.1 Load model
    model: transformers.GPT2Model = transformers.AutoModel.from_pretrained(model)

    # 1.2.1 Load and initialize the models' tokenizer and prepare tokenizer for batch-encoding plus.
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("distilgpt2")
    # 1.2.2 In case the fast-tokenizer has no padding, set the padding manually
    set_padding_or_none(tokenizer, set_padding="eos")

    # 2. Load dataset
    dataset, target = get_dataset(dataset_name=dataset, sub_category=sub_category)

    # TODO: Create train / test split according to paper
    if truncation == "before":
        logging.warning("Truncating before pre-processing will affect the 'range' of the scaled values!")
        # TODO: Implement maximum length split.
    # 3. Pre-process data, and get mapping function to re-construct
    # Note, that the scaler that is returned is scaled on the entire time-series
    scaler, process_values, input_ids = pre_processing.convert_timeseries_to_fixed_precision(
        dataset, tokenizer, target=target
    )

    if truncation == "after":
        raise NotImplementedError("Input truncation based on tokenization length is not yet supported!")
        logging.warning("Truncating will leak information of previous states")
    print(input_ids)


if __name__ == "__main__":
    default_max_history = 400
    default_model = "distilgpt2"
    default_dataset = "hpc"
    default_subcat = None
    default_truncate = "before"
    # Create argparse parser
    parser = argparse.ArgumentParser(description="Description of your experiment with Zero-Shot-Time")

    # Add command line arguments
    parser.add_argument(
        "--max_history",
        type=int,
        default=default_max_history,
        help=f"Maximum history length (default: {default_max_history})",
    )
    parser.add_argument("--model", type=str, default=default_model, help=f"Model name (default: {default_model})")
    parser.add_argument(
        "--dataset", type=str, default=default_dataset, help=f"Dataset name (default: {default_dataset})"
    )

    parser.add_argument(
        "--subcat",
        type=str,
        default=default_subcat,
        required=False,
        help="Dataset subcategory (needed for HF dataset) (default: None)",
    )
    parser.add_argument(
        "--truncate",
        type=str,
        choices=["before", "after"],
        default=default_truncate,
        help=f"Truncate time series before or after tokenization (default: {default_truncate})",
    )

    # Parse command line arguments
    args = parser.parse_args()

    main(
        model=args.model,
        dataset=args.dataset,
        sub_category=args.subcat,
        max_history=args.max_history,
        truncation=args.truncate,
    )
