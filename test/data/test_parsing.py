import unittest


class TestStringRepresentation(unittest.TestCase):



    def test_tokenization(self):
        pass



    def test_detokenization(self):
        scaler, process_values, input_ids = pre_processing.convert_timeseries_to_fixed_precision(dataset, tokenizer,
                                                                                                 target=target,
                                                                                                 precision=5)
        reconstructed_values = scaler.inverse_transform(process_values[None, :]).flatten()
        test = convert_tokens_to_timeseries(input_ids, tokenizer, base_transformation(precision=5))
        converted_test = scaler.inverse_transform(test[None, :]).flatten()

        rmse_construction_error = np.sqrt(np.mean(np.square(converted_test - reconstructed_values)))
        mape_construction_error = np.mean(np.abs((converted_test - reconstructed_values) / reconstructed_values))
        print(rmse_construction_error, mape_construction_error)


def test_max_length(self):
        pass
