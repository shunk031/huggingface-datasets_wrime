import logging
from typing import Final, List, TypedDict

import datasets as ds
import pandas as pd

logger = logging.getLogger(__name__)

_CITATION = """\
@inproceedings{kajiwara-etal-2021-wrime,
    title = "{WRIME}: A New Dataset for Emotional Intensity Estimation with Subjective and Objective Annotations",
    author = "Kajiwara, Tomoyuki  and
      Chu, Chenhui  and
      Takemura, Noriko  and
      Nakashima, Yuta  and
      Nagahara, Hajime",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.169",
    doi = "10.18653/v1/2021.naacl-main.169",
    pages = "2095--2104",
    abstract = "We annotate 17,000 SNS posts with both the writer{'}s subjective emotional intensity and the reader{'}s objective one to construct a Japanese emotion analysis dataset. In this study, we explore the difference between the emotional intensity of the writer and that of the readers with this dataset. We found that the reader cannot fully detect the emotions of the writer, especially anger and trust. In addition, experimental results in estimating the emotional intensity show that it is more difficult to estimate the writer{'}s subjective labels than the readers{'}. The large gap between the subjective and objective emotions imply the complexity of the mapping from a post to the subjective emotion intensities, which also leads to a lower performance with machine learning models.",
}
"""

_DESCRIPTION = """\
WRIME dataset is a new dataset for emotional intensity estimation with subjective and objective annotations.
"""

_HOMEPAGE = "https://github.com/ids-cv/wrime"

_LICENSE = """\
- The dataset is available for research purposes only.
- Redistribution of the dataset is prohibited.
"""


class URLs(TypedDict):
    ver1: str
    ver2: str


_URLS: URLs = {
    "ver1": "https://raw.githubusercontent.com/ids-cv/wrime/master/wrime-ver1.tsv",
    "ver2": "https://raw.githubusercontent.com/ids-cv/wrime/master/wrime-ver2.tsv",
}


def _fix_typo_in_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # ref. https://github.com/ids-cv/wrime/pull/4
    df = df.rename(
        columns={
            "Reader2_Saddness": "Reader2_Sadness",
            "Reader3_Saddness": "Reader3_Sadness",
        }
    )
    return df


def _convert_column_name(df: pd.DataFrame) -> pd.DataFrame:

    # ['Sentence', 'UserID', 'Datetime', 'Train/Dev/Test', 'Writer_Joy', ...]
    # -> ['sentence', 'userid', 'datetime', 'train/dev/test', 'writer_joy', ...]
    df.columns = df.columns.str.lower()

    # ['avg. readers_joy', 'avg. readers_sadness', 'avg. readers_anticipation', ...]
    # -> ['avg_readers_joy', 'avg_readers_sadness', 'avg_readers_anticipation', ...]
    df.columns = df.columns.str.replace(". ", "_")

    return df


def _load_tsv(tsv_path: str) -> pd.DataFrame:
    logger.info(f"Load TSV file from {tsv_path}")
    df = pd.read_csv(tsv_path, delimiter="\t")

    # some preprocessing
    df = _fix_typo_in_dataset(df)
    df = _convert_column_name(df)

    return df


EIGHT_EMOTIONS: Final[List[str]] = [
    "joy",
    "sadness",
    "anticipation",
    "surprise",
    "anger",
    "fear",
    "disgust",
    "trust",
]


class WrimeDataset(ds.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ds.BuilderConfig(
            name="ver1",
            version=ds.Version("1.0.0"),
            description="WRIME dataset ver. 1",
        ),
        ds.BuilderConfig(
            name="ver2",
            version=ds.Version("2.0.0"),
            description="WRIME dataset ver. 2",
        ),
    ]

    def __info(self, emotions: List[str]) -> ds.DatasetInfo:
        features_dict = {
            "sentence": ds.Value("string"),
            "user_id": ds.Value("string"),
            "datetime": ds.Value("string"),
        }

        readers = [f"reader{i}" for i in range(1, 4)] + ["avg_readers"]
        for k in ["writer"] + readers:
            features_dict[k] = {emotion: ds.Value("int8") for emotion in emotions}  # type: ignore
        features = ds.Features(features_dict)

        return ds.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _info(self) -> ds.DatasetInfo:

        if self.config.version.major == 1:  # type: ignore
            # Ver.1: 80人の筆者から収集した43,200件の投稿に感情強度をラベル付け
            return self.__info(emotions=EIGHT_EMOTIONS)

        elif self.config.version.major == 2:  # type: ignore
            # Ver.2: 60人の筆者から収集した35,000件の投稿（Ver.1のサブセット）に感情極性を追加でラベル付け
            return self.__info(emotions=EIGHT_EMOTIONS + ["sentiment"])

        else:
            raise ValueError(f"Invalid dataset version: {self.config.version}")

    def _split_generators(self, dl_manager: ds.DownloadManager):
        wrime_datasets = dl_manager.download_and_extract(_URLS)
        major_version_name = f"ver{self.config.version.major}"  # type: ignore

        wrime_df = _load_tsv(tsv_path=wrime_datasets[major_version_name])
        tng_wrime_df = wrime_df[wrime_df["train/dev/test"] == "train"]
        dev_wrime_df = wrime_df[wrime_df["train/dev/test"] == "dev"]
        tst_wrime_df = wrime_df[wrime_df["train/dev/test"] == "test"]

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,  # type: ignore
                gen_kwargs={"df": tng_wrime_df},
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,  # type: ignore
                gen_kwargs={"df": dev_wrime_df},
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,  # type: ignore
                gen_kwargs={"df": tst_wrime_df},
            ),
        ]

    def __generate_examples(self, df: pd.DataFrame, emotions: List[str]):
        for i in range(len(df)):
            row_df = df.iloc[i]

            example_dict = {
                "sentence": row_df["sentence"],
                "user_id": row_df["userid"],
                "datetime": row_df["datetime"],
            }

            readers = [f"reader{i}" for i in range(1, 4)] + ["avg_readers"]
            for k in ["writer"] + readers:
                example_dict[k] = {
                    emotion: row_df[f"{k}_{emotion}"] for emotion in emotions
                }
            yield i, example_dict

    def _generate_examples(self, df: pd.DataFrame):  # type: ignore[override]

        if self.config.version.major == 1:  # type: ignore
            yield from self.__generate_examples(
                df,
                emotions=EIGHT_EMOTIONS,
            )
        elif self.config.version.major == 2:  # type: ignore
            yield from self.__generate_examples(
                df,
                emotions=EIGHT_EMOTIONS + ["sentiment"],
            )
        else:
            raise ValueError(f"Invalid dataset version: {self.config.version}")
