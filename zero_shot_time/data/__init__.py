import typing as tp

import datasets


def get_dataset(dataset_name: str, sub_category=None, path=None) -> tp.Tuple[datasets.Dataset, str]:
    """Helper function to retrieve huggingface / local datasets.

    Args:
        dataset_name: Name of the (super) dataset.
        sub_category: In case of a nested dataset (e.g. monas_tf), subcategory to load.


    Returns:
        datasets.Dataset object containing the requested data.
        Target column on which to perform the prediction.
    """
    if sub_category is not None:
        dataset, target = datasets.load_dataset(dataset_name, sub_category), "target"
    if dataset_name == "hpc":
        path = path or "./data/hpc-jobs.csv"
        dataset, target = datasets.Dataset.from_csv(path), "num_jobs"

    return dataset, target
