"""Monash Time Series Forecasting Repository Dataset."""


from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import datasets


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\

"""

_DESCRIPTION = """\
HPC Forecasting dataset
"""

_HOMEPAGE = ""

_LICENSE = ""

_ROOT_URL = "data"


@dataclass
class HPCDatasetBuilderConfig(datasets.BuilderConfig):
    """MonashTSF builder config with some added meta data."""

    file_name: Optional[str] = None
    url: Optional[str] = None
    prediction_length: Optional[int] = None
    item_id_column: Optional[str] = None
    data_column: Optional[str] = None
    target_fields: Optional[List[str]] = None
    feat_dynamic_real_fields: Optional[List[str]] = None
    multivariate: bool = False
    rolling_evaluations: int = 1


class MonashTSF(datasets.GeneratorBasedBuilder):
    """Builder of Monash Time Series Forecasting repository of datasets."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = HPCDatasetBuilderConfig

    BUILDER_CONFIGS = [
        HPCDatasetBuilderConfig(
            name="workload",
            version=VERSION,
            description="Monthly workload prediction on HPC cluster from 2021 to 2022",
            url=None,
            file_name="hpc-jobs.csv",
            data_column="series_type",
        )
    ]

    def _info(self):
        if self.config.multivariate:
            raise ValueError("No HPC dataset has multi-variate output")
        else:
            features = datasets.Features(
                {
                    "start": datasets.Value("timestamp[yy-mm-dd]"),
                    "target": datasets.Sequence(datasets.Value("int32")),
                    "feat_static_cat": datasets.Sequence(datasets.Value("uint64")),
                    # "feat_static_real":  datasets.Sequence(datasets.Value("float32")),
                    # "feat_dynamic_real": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
                    # "feat_dynamic_cat": datasets.Sequence(datasets.Sequence(datasets.Value("uint64"))),
                    # "item_id": datasets.Value("string"),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        urls = f"{_ROOT_URL}/{self.config.file_name}"
        data_dir = dl_manager.download_and_extract(urls)
        file_path = Path(data_dir) / (self.config.file_name.split(".")[0] + ".tsf")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": file_path,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": file_path, "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": file_path,
                    "split": "val",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        loaded_data = pd.read_csv(filepath)

        if self.config.target_fields is not None:
            target_fields = loaded_data[loaded_data[self.config.data_column].isin(self.config.target_fields)]
        else:
            target_fields = loaded_data
        if self.config.feat_dynamic_real_fields is not None:
            feat_dynamic_real_fields = loaded_data[
                loaded_data[self.config.data_column].isin(self.config.feat_dynamic_real_fields)
            ]
        else:
            feat_dynamic_real_fields = None

        for cat, ts in target_fields.iterrows():
            start = ts.get("start_timestamp")
            target = ts.target
            if feat_dynamic_real_fields is not None:
                feat_dynamic_real = np.vstack(feat_dynamic_real_fields.target)
            else:
                feat_dynamic_real = None

            feat_static_cat = [cat]
            if self.config.data_column is not None:
                item_id = f"{ts.series_name}-{ts[self.config.data_column]}"
            else:
                item_id = ts.series_name

            if split in ["train", "val"]:
                offset = forecast_horizon * self.config.rolling_evaluations + forecast_horizon * (split == "train")
                target = target[..., :-offset]
                if feat_dynamic_real is not None:
                    feat_dynamic_real = feat_dynamic_real[..., :-offset]

            yield cat, {
                "start": start,
                "target": target,
                "feat_dynamic_real": feat_dynamic_real,
                "feat_static_cat": feat_static_cat,
                "item_id": item_id,
            }
