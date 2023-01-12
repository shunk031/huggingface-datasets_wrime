import datasets as ds
import pytest


@pytest.fixture
def dataset_path() -> str:
    return "wrime.py"


@pytest.mark.parametrize(
    "dataset_name, expected_train_num_rows, expected_val_num_rows, expected_test_num_rows,",
    (
        ("ver1", 40000, 1200, 2000),
        ("ver2", 30000, 2500, 2500),
    ),
)
def test_load_dataset(
    dataset_path: str,
    dataset_name: str,
    expected_train_num_rows: int,
    expected_val_num_rows: int,
    expected_test_num_rows: int,
) -> None:

    dataset = ds.load_dataset(path=dataset_path, name=dataset_name)

    assert dataset["train"].num_rows == expected_train_num_rows  # type: ignore
    assert dataset["validation"].num_rows == expected_val_num_rows  # type: ignore
    assert dataset["test"].num_rows == expected_test_num_rows  # type: ignore

    writer_readers = [
        "writer",
        "reader1",
        "reader2",
        "reader3",
        "avg_readers",
    ]
    expected_keys = ["sentence", "user_id", "datetime"] + writer_readers

    for split in ["train", "validation", "test"]:
        split_dataset = dataset[split]  # type: ignore

        for data in split_dataset:
            assert len(data.keys()) == len(expected_keys)
            for expected_key in expected_keys:
                assert expected_key in data.keys()

            for k in writer_readers:
                if dataset_name == "ver1":
                    assert len(data[k]) == 8  # 8 感情強度
                elif dataset_name == "ver2":
                    assert len(data[k]) == 8 + 1  # 8 感情強度 + 1 感情極性
                else:
                    raise ValueError(f"Invalid dataset version: {dataset_name}")
