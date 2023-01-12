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
