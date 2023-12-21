
import typing as tp

import darts
import darts.datasets as ds
import datasets
import numpy as np


def get_dataset(dataset_name: str, sub_category=None, path=None) -> tp.Tuple[tp.Union[datasets.Dataset, np.array], tp.Optional[str]]:
    """Helper function to retrieve huggingface / local datasets.

    Args:
        dataset_name: Name of the (super) dataset.
        sub_category: In case of a nested dataset (e.g. monas_tf), subcategory to load.


    Returns:
        datasets.Dataset object containing the requested data.
        Target column on which to perform the prediction.
    """
    if dataset_name == 'monash_tsf' and sub_category is not None:
        dataset, target = datasets.load_dataset(dataset_name, sub_category), "target"

    if dataset_name == 'darts' and sub_category is not None:
        data_lookup = {
            'beer': ds.AusBeerDataset,
            'airpassenger': ds.AirPassengersDataset
        }
        dataset = data_lookup[sub_category]().load()

        full_array = dataset.data_array().data.flatten()
        return full_array, None

    if dataset_name == "hpc":
        path = path or "./data/hpc-jobs.csv"
        dataset, target = datasets.Dataset.from_csv(path), "num_jobs"


    return dataset, target
