import typing as tp

import datasets

import zero_shot_time.data.pre_processing as pre_processing


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
