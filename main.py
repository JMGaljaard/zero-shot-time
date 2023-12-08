import datasets
import transformers
import typing as tp

from zero_shot_time.data import pre_processing


def get_dataset(dataset_name: str, sub_category = None) -> tp.Tuple[datasets.Dataset, str]:
    """Helper function to retrieve huggingface / local datasets.

    Args:
        dataset_name: Name of the (super) dataset.
        sub_category: In case of a nested dataset (e.g. monas_tf), subcategory to load.


    Returns:
        datasets.Dataset object containing the requested data.
        Target column on which to perform the prediction.
    """
    if sub_category is not None:
        dataset = datasets.load_dataset(dataset_name, sub_category)
    if dataset_name == 'hpc':

        dataset, target = datasets.Dataset.from_csv('./data/hpc-jobs.csv'), 'num_jobs'

    return dataset, target


def set_padding_or_none(tokenizer: transformers.PreTrainedTokenizerFast, set_padding='eos'):
    """Helper function to initialze the tokenization model when not set during initialization.

    """

    if tokenizer.pad_token_id is None and set_padding:
        tokenizer.pad_token_id = getattr(tokenizer, f"{set_padding}_token_id")
        tokenizer.pad_token = getattr(tokenizer, f"{set_padding}_token")

    return tokenizer


def main(
        model: str,
        dataset: str
):
    # Load model
    model: transformers.GPT2Model = transformers.AutoModel.from_pretrained('distilgpt2')

    # Initialize tokenizer and prepare tokenizer for batch-encoding plus.
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('distilgpt2')
    set_padding_or_none(tokenizer, set_padding='eos')
    max_context_length = model.config.max_length

    # Load dataset
    dataset, target = get_dataset(dataset)


    scaler, process_values, input_ids = pre_processing.convert_timeseries_to_fixed_precision(dataset, tokenizer, target = target)


    print(input_ids)

if __name__ == '__main__':

    model = 'distilgpt2'
    dataset = 'hpc'

    main(model, dataset)