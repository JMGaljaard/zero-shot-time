import transformers

from zero_shot_time.data import pre_processing, get_dataset
from zero_shot_time.generation.tokenizer import set_padding_or_none


def main(
        model: str = 'distilpgt2',
        dataset: str = 'hpc'
):
    # 1.1 Load model
    model: transformers.GPT2Model = transformers.AutoModel.from_pretrained(model)

    # 1.2.1 Load and initialize the models' tokenizer and prepare tokenizer for batch-encoding plus.
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('distilgpt2')
    # 1.2.2 In case the fast-tokenizer has no padding, set the padding manually
    set_padding_or_none(tokenizer, set_padding='eos')
    max_context_length = model.config.max_length

    # 2. Load dataset
    dataset, target = get_dataset(dataset)

    # 3. Pre-process data, and get mapping function to re-construct
    # Note, that the scaler that is returned is scaled on the entire time-series
    scaler, process_values, input_ids = pre_processing.convert_timeseries_to_fixed_precision(dataset, tokenizer, target = target)


    print(input_ids)

if __name__ == '__main__':


    max_history = 400
    model = 'distilgpt2'
    dataset = 'hpc'

    main(model, dataset)