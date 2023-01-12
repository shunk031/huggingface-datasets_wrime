import logging
from typing import TypedDict

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


def _load_tsv(tsv_path: str) -> pd.DataFrame:
    logger.info(f"Load TSV file from {tsv_path}")
    df = pd.read_csv(tsv_path, delimiter="\t")

    df = _fix_typo_in_dataset(df)

    return df


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

    def _info(self) -> ds.DatasetInfo:
        features = ds.Features(
            {
                "sentence": ds.Value("string"),
                "user_id": ds.Value("string"),
                "datetime": ds.Value("string"),
                "writer": {
                    "joy": ds.Value("uint8"),
                    "sadness": ds.Value("uint8"),
                    "anticipation": ds.Value("uint8"),
                    "surprise": ds.Value("uint8"),
                    "anger": ds.Value("uint8"),
                    "fear": ds.Value("uint8"),
                    "disgust": ds.Value("uint8"),
                    "trust": ds.Value("uint8"),
                },
                "reader1": {
                    "joy": ds.Value("uint8"),
                    "sadness": ds.Value("uint8"),
                    "anticipation": ds.Value("uint8"),
                    "surprise": ds.Value("uint8"),
                    "anger": ds.Value("uint8"),
                    "fear": ds.Value("uint8"),
                    "disgust": ds.Value("uint8"),
                    "trust": ds.Value("uint8"),
                },
                "reader2": {
                    "joy": ds.Value("uint8"),
                    "sadness": ds.Value("uint8"),
                    "anticipation": ds.Value("uint8"),
                    "surprise": ds.Value("uint8"),
                    "anger": ds.Value("uint8"),
                    "fear": ds.Value("uint8"),
                    "disgust": ds.Value("uint8"),
                    "trust": ds.Value("uint8"),
                },
                "reader3": {
                    "joy": ds.Value("uint8"),
                    "sadness": ds.Value("uint8"),
                    "anticipation": ds.Value("uint8"),
                    "surprise": ds.Value("uint8"),
                    "anger": ds.Value("uint8"),
                    "fear": ds.Value("uint8"),
                    "disgust": ds.Value("uint8"),
                    "trust": ds.Value("uint8"),
                },
                "avg_readers": {
                    "joy": ds.Value("uint8"),
                    "sadness": ds.Value("uint8"),
                    "anticipation": ds.Value("uint8"),
                    "surprise": ds.Value("uint8"),
                    "anger": ds.Value("uint8"),
                    "fear": ds.Value("uint8"),
                    "disgust": ds.Value("uint8"),
                    "trust": ds.Value("uint8"),
                },
            }
        )
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: ds.DownloadManager):
        wrime_datasets = dl_manager.download_and_extract(_URLS)
        major_version_name = f"ver{self.config.version.major}"  # type: ignore

        wrime_df = _load_tsv(tsv_path=wrime_datasets[major_version_name])
        tng_wrime_df = wrime_df[wrime_df["Train/Dev/Test"] == "train"]
        dev_wrime_df = wrime_df[wrime_df["Train/Dev/Test"] == "dev"]
        tst_wrime_df = wrime_df[wrime_df["Train/Dev/Test"] == "test"]

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

    def _generate_examples(  # type: ignore[override]
        self,
        df: pd.DataFrame,
    ):
        for i in range(len(df)):
            row_df = df.iloc[i]

            example_dict = {
                "sentence": row_df["Sentence"],
                "user_id": row_df["UserID"],
                "datetime": row_df["Datetime"],
            }

            example_dict["writer"] = {
                "joy": row_df["Writer_Joy"],
                "sadness": row_df["Writer_Sadness"],
                "anticipation": row_df["Writer_Anticipation"],
                "surprise": row_df["Writer_Surprise"],
                "anger": row_df["Writer_Anger"],
                "fear": row_df["Writer_Fear"],
                "disgust": row_df["Writer_Disgust"],
                "trust": row_df["Writer_Trust"],
            }

            for reader_num in range(1, 4):
                example_dict[f"reader{reader_num}"] = {
                    "joy": row_df[f"Reader{reader_num}_Joy"],
                    "sadness": row_df[f"Reader{reader_num}_Sadness"],
                    "anticipation": row_df[f"Reader{reader_num}_Anticipation"],
                    "surprise": row_df[f"Reader{reader_num}_Surprise"],
                    "anger": row_df[f"Reader{reader_num}_Anger"],
                    "fear": row_df[f"Reader{reader_num}_Fear"],
                    "disgust": row_df[f"Reader{reader_num}_Disgust"],
                    "trust": row_df[f"Reader{reader_num}_Trust"],
                }

            example_dict["avg_readers"] = {
                "joy": row_df["Avg. Readers_Joy"],
                "sadness": row_df["Avg. Readers_Sadness"],
                "anticipation": row_df["Avg. Readers_Anticipation"],
                "surprise": row_df["Avg. Readers_Surprise"],
                "anger": row_df["Avg. Readers_Anger"],
                "fear": row_df["Avg. Readers_Fear"],
                "disgust": row_df["Avg. Readers_Disgust"],
                "trust": row_df["Avg. Readers_Trust"],
            }

            yield i, example_dict
