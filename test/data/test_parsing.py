import unittest

import numpy as np
import parameterized.parameterized as parameterized
import transformers

from zero_shot_time.data import pre_processing, get_dataset
from zero_shot_time.data.post_processing import convert_tokens_to_timeseries, base_transformation
from zero_shot_time.data.pre_processing import stringify_values, map_substring_to_tokens, tokenize_values
from zero_shot_time.generation.tokenizer import set_padding_or_none


class TestStringRepresentationGPT2(unittest.TestCase):
    """Basic stringifying test-cases to ensure that behavior of implementation matches that as described in the original
    work.

    """

    def setUp(self) -> None:
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('distilgpt2')
        set_padding_or_none(self.tokenizer)
        hpc_dataset, target = get_dataset('hpc', path='../data/hpc-jobs.csv')
        self.hpc_dataset = hpc_dataset
        self.target = target

    @parameterized.expand([
        [0.123, ' 1 2'],
        [1.23, ' 1 2 3'],
        [12.3, ' 1 2 3 0'],
        [123.0, ' 1 2 3 0 0']
    ])
    def test_stringify_with_fixed_precision(self, input, output, precision=2):
        transforms = np.array([input])
        _, [stringified] = stringify_values(transforms, value_mapper=map_substring_to_tokens)
        joined = ''.join(stringified)
        self.assertEquals(joined, output)

    @parameterized.expand([
        [0.123, ' 1 2, '],
        [1.23, ' 1 2 3, '],
        [12.3, ' 1 2 3 0, '],
        [123.0, ' 1 2 3 0 0, ']
    ])
    def test_tokenization_with_fixed_precision(self, input, output, precision=2):
        transforms = np.array([input])
        _, stringified_values = stringify_values(transforms, value_mapper=map_substring_to_tokens)

        time_series_tokens = tokenize_values(stringified_values, self.tokenizer)

        detokenized = self.tokenizer.decode(time_series_tokens)

        # Note that we assume that the tokenization is performed by
        self.assertEquals(detokenized, output)



    def test_detokenization(self):
        PRECISION = 7
        scaler, process_values, input_ids = pre_processing.convert_timeseries_to_fixed_precision(
            self.hpc_dataset,
            self.tokenizer,
             target=self.target,
             precision=7
        )
        reconstructed_values = scaler.inverse_transform(process_values[None, :]).flatten()
        test = convert_tokens_to_timeseries(input_ids, self.tokenizer, base_transformation(precision=PRECISION))
        converted_test = scaler.inverse_transform(test[None, :]).flatten()

        rmse_construction_error = np.sqrt(np.mean(np.square(converted_test - reconstructed_values)))
        mape_construction_error = np.mean(np.abs((converted_test - reconstructed_values) / reconstructed_values))

        self.assertLess(rmse_construction_error, 0.005)
        self.assertLess(mape_construction_error, 0.0001)
